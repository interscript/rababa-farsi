"""GRPO on Persian G2P v1 — homograph-weighted gold reward, no teacher.

Motivated by the cross-language RAFT negative result (Persian 76.85 tie,
Arabic 1.183 tie): positive-only updates cannot sharpen a posterior
split between legal readings. GRPO samples G=8 candidates per prompt,
rewards = -homograph-weighted word CER vs gold (deterministic oracle),
group-normalized advantages update ALL samples (negative gradient on
wrong readings), KL leash to the frozen v1 reference.

Usage:
    modal run --detach train_persian_grpo.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import modal

datasets_volume = modal.Volume.from_name("persian-g2p-datasets", create_if_missing=True)
checkpoints_volume = modal.Volume.from_name("persian-g2p-checkpoints", create_if_missing=True)

RUN = "persian_g2p_grpo/run-001"
BASE = "/checkpoints/persian_g2p/run-001/best"
DATA = Path("/datasets/persian-g2p")
STEPS = 800
GROUP = 8
PROMPTS_PER_STEP = 2
GRAD_ACCUM = 8
TEMP = 1.0
LR = 1e-5
KL_BETA = 0.05
MAX_LEN = 256
DEV_N = 500
HG_WEIGHT = 3.0
HG_MIN_FREQ = 4
EVAL_EVERY = 100
SAVE_EVERY = 100

SENTENCEBENCH_URL = (
    "https://huggingface.co/datasets/MahtaFetrat/SentenceBench/resolve/main/SentenceBench.csv"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.5.1", "transformers==4.46.3", "editdistance", "pandas", "tqdm")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)

app = modal.App("persian-g2p-grpo", image=image)


def build_homograph_lexicon(rows: list[dict]) -> set[str]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        sw, tw = row["src"].split(), row["tgt"].split()
        if len(sw) != len(tw):
            continue
        for w, p in zip(sw, tw):
            counts.setdefault(w, {})
            counts[w][p] = counts[w].get(p, 0) + 1
    return {
        w for w, prons in counts.items()
        if sum(prons.values()) >= HG_MIN_FREQ and len(prons) >= 2
    }


def hg_flags(src: str, gold: str, lex: set[str]) -> list[bool]:
    sw, gw = src.split(), gold.split()
    if len(sw) != len(gw):
        return [False] * len(gw)
    return [w in lex for w in sw]


def word_reward(pred: str, gold: str, flags: list[bool]) -> float:
    import editdistance

    pw, gw = pred.split(), gold.split()
    n = max(len(pw), len(gw))
    s = tot = 0.0
    for i in range(n):
        w = gw[i] if i < len(gw) else ""
        p = pw[i] if i < len(pw) else ""
        weight = HG_WEIGHT if (i < len(flags) and flags[i]) else 1.0
        s += weight * editdistance.eval(p, w) / max(1, len(w))
        tot += weight
    return s / tot if tot else 1.0


@app.function(
    gpu="A100",
    timeout=11 * 60 * 60,
    volumes={"/datasets": datasets_volume, "/checkpoints": checkpoints_volume},
)
def run() -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    datasets_volume.reload()
    checkpoints_volume.reload()
    run_dir = Path("/checkpoints") / RUN
    if (run_dir / "EVAL_DONE").exists():
        return {"status": "already-done"}
    run_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [json.loads(l) for l in (DATA / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    dev_rows = [json.loads(l) for l in (DATA / "val.jsonl").read_text(encoding="utf-8").splitlines()][:DEV_N]
    hg_lex = build_homograph_lexicon(train_rows)
    print(f"[lexicon] {len(hg_lex)} homographs", flush=True)

    rng = random.Random(43)
    pool_rows = train_rows[:200_000]
    rng.shuffle(pool_rows)
    pool = pool_rows[:40_000]
    print(f"[data] pool={len(pool)} dev={len(dev_rows)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE)
    policy = AutoModelForSeq2SeqLM.from_pretrained(BASE).to("cuda")
    ref = AutoModelForSeq2SeqLM.from_pretrained(BASE).to("cuda").eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    device = next(policy.parameters()).device
    opt = torch.optim.AdamW(policy.parameters(), lr=LR, weight_decay=0.01)

    def dev_reward() -> float:
        rs = []
        policy.eval()
        with torch.no_grad():
            for i in range(0, len(dev_rows), 32):
                batch = dev_rows[i : i + 32]
                enc = tokenizer([r["src"] for r in batch], return_tensors="pt", padding=True,
                                truncation=True, max_length=MAX_LEN).to(device)
                with torch.autocast("cuda", torch.bfloat16):
                    gen = policy.generate(**enc, max_new_tokens=2 * MAX_LEN, num_beams=1)
                outs = tokenizer.batch_decode(gen, skip_special_tokens=True)
                for o, r in zip(outs, batch):
                    rs.append(word_reward(o, r["tgt"], hg_flags(r["src"], r["tgt"], hg_lex)))
        return sum(rs) / len(rs)

    def token_logprobs(model, src_texts, tgt_texts):
        enc = tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_LEN).to(device)
        labels = tokenizer(tgt_texts, return_tensors="pt", padding=True, truncation=True,
                           max_length=2 * MAX_LEN)
        lab = labels["input_ids"].to(device)
        attn = labels["attention_mask"].to(device)
        with torch.autocast("cuda", torch.bfloat16):
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                           labels=lab).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        tgt = lab[:, 1:].unsqueeze(-1)
        lp = torch.gather(logprobs[:, :-1], 2, tgt).squeeze(-1)
        mask = attn[:, 1:].float()
        return (lp * mask).sum(dim=1), mask.sum(dim=1)

    state_path = run_dir / "state.json"
    metrics_path = run_dir / "metrics.jsonl"
    start_step, best_dev = 0, None
    if state_path.exists():
        st = json.loads(state_path.read_text())
        start_step, best_dev = st["step"], st.get("best_dev")
        ckpt = run_dir / "policy_last"
        if ckpt.is_dir():
            policy.load_state_dict(AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).state_dict())
            if (ckpt / "optimizer.pt").exists():
                opt.load_state_dict(torch.load(str(ckpt / "optimizer.pt"), map_location="cpu"))
            print(f"[resume] step {start_step}", flush=True)
    if (run_dir / "FINISH_ONLY").exists():
        start_step = STEPS
        print("[finish-only] skipping remaining steps; benchmarking saved best", flush=True)
    if best_dev is None:
        best_dev = dev_reward()
        print(f"[dev] v1 baseline reward={best_dev:.4%}", flush=True)

    import statistics

    ptr = 0
    policy.train()
    for step in range(start_step, STEPS):
        for _ in range(GRAD_ACCUM):
            batch_rows = []
            while len(batch_rows) < PROMPTS_PER_STEP:
                batch_rows.append(pool[ptr % len(pool)])
                ptr += 1
            srcs = [r["src"] for r in batch_rows]
            golds = [r["tgt"] for r in batch_rows]

            enc = tokenizer(srcs, return_tensors="pt", padding=True, truncation=True,
                            max_length=MAX_LEN).to(device)
            with torch.no_grad():
                with torch.autocast("cuda", torch.bfloat16):
                    gen = policy.generate(**enc, max_new_tokens=2 * MAX_LEN, num_beams=1,
                                          do_sample=True, temperature=TEMP,
                                          num_return_sequences=GROUP)
            outs = tokenizer.batch_decode(gen, skip_special_tokens=True)

            rewards = []
            for gi, r in enumerate(batch_rows):
                flags = hg_flags(r["src"], r["tgt"], hg_lex)
                for k in range(GROUP):
                    rewards.append(-word_reward(outs[gi * GROUP + k], r["tgt"], flags))
            mu = statistics.mean(rewards)
            sd = statistics.pstdev(rewards) + 1e-4
            advs = [(x - mu) / sd for x in rewards]

            rep_srcs = [s for s in srcs for _ in range(GROUP)]
            lp_pol, n_tok = token_logprobs(policy, rep_srcs, outs)
            with torch.no_grad():
                lp_ref, _ = token_logprobs(ref, rep_srcs, outs)
            kl = (lp_pol - lp_ref).clamp(min=-10, max=10)
            A = torch.tensor(advs, device=device, dtype=torch.float32)
            loss = -((A / n_tok.clamp(min=1)) * lp_pol).mean() + KL_BETA * kl.mean()
            (loss / GRAD_ACCUM).backward()
            if step % 25 == 0:
                print(f"[step {step}] loss={loss.item():.4f} mean_r={mu:.4f} best={best_dev:.4%}", flush=True)

        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        policy.train()

        if (step + 1) % EVAL_EVERY == 0 or step + 1 == STEPS:
            d = dev_reward()
            print(f"[dev] step {step+1}: reward={d:.4%} (best={best_dev:.4%})", flush=True)
            if d < best_dev:
                best_dev = d
                best_dir = run_dir / "best"
                best_dir.mkdir(parents=True, exist_ok=True)
                policy.save_pretrained(str(best_dir))
                tokenizer.save_pretrained(str(best_dir))
            with metrics_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step + 1, "dev_reward": d, "best_dev": best_dev}) + "\n")
            checkpoints_volume.commit()

        if (step + 1) % SAVE_EVERY == 0 or step + 1 == STEPS:
            last = run_dir / "policy_last"
            last.mkdir(parents=True, exist_ok=True)
            policy.save_pretrained(str(last))
            tokenizer.save_pretrained(str(last))
            torch.save(opt.state_dict(), str(last / "optimizer.pt"))
            state_path.write_text(json.dumps({"step": step + 1, "best_dev": best_dev}))
            checkpoints_volume.commit()

    # ---- final one-shot SentenceBench (measured once) ----
    import csv
    import urllib.request

    eval_model = AutoModelForSeq2SeqLM.from_pretrained(str(run_dir / "best")).to(device)
    eval_model.eval()
    local_csv = Path("/tmp/SentenceBench.csv")
    urllib.request.urlretrieve(SENTENCEBENCH_URL, local_csv)
    with local_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    hg_rows = [r for r in rows if (r.get("homograph word") or "").strip() and (r.get("pronunciation") or "").strip()]

    results: dict = {"run": RUN, "best_dev": best_dev}
    srcs_hg = [r["grapheme"].strip() for r in hg_rows]
    for beam in (4, 1):
        preds = []
        with torch.no_grad():
            for i in range(0, len(srcs_hg), 64):
                batch = srcs_hg[i : i + 64]
                enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                max_length=MAX_LEN).to(device)
                with torch.autocast("cuda", torch.bfloat16):
                    gen = eval_model.generate(**enc, max_new_tokens=MAX_LEN, num_beams=beam)
                preds.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
        correct = correct_norm = total = unaligned = 0
        for row, pred in zip(hg_rows, preds):
            g_words = row["grapheme"].strip().split()
            p_words = pred.strip().split()
            positions = [i for i, w in enumerate(g_words) if row["homograph word"].strip() in w]
            if not positions:
                continue
            pos = positions[0]
            if pos >= len(p_words):
                unaligned += 1
                continue
            pron = row["pronunciation"].strip()
            pp = p_words[pos]
            correct += int(pp == pron)
            correct_norm += int(pp == pron or pp.rstrip("e") == pron or pp == pron.rstrip("e"))
            total += 1
        results[f"sb_beam{beam}"] = {
            "ha_exact": correct / max(1, total), "ha_ezafe_norm": correct_norm / max(1, total),
            "n": total,
        }
        print(f"[sb] beam={beam} exact={correct / max(1, total):.4%} "
              f"ezafe-norm={correct_norm / max(1, total):.4%} ({correct_norm}/{total})", flush=True)

    (run_dir / "EVAL_DONE").touch()
    checkpoints_volume.commit()
    return results


@app.function(volumes={"/checkpoints": checkpoints_volume})
def mark_finish() -> str:
    p = Path("/checkpoints") / RUN / "FINISH_ONLY"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    checkpoints_volume.commit()
    return str(p)


@app.local_entrypoint()
def mark_finish_only():
    print(mark_finish.remote())


@app.local_entrypoint()
def main():
    run.remote()

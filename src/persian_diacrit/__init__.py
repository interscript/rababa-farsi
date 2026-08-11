"""Persian/Farsi diacritization — add harakat (short vowels) and detect ezafe."""

from .constants import HARAQAT_LIST, NUM_HARAQAT
from .encoder import PersianEncoder
from .model import PersianDiacritModel
from .dataset import PersianDataset, collate_batch
from .evaluate import haraqat_der, ezafe_metrics

__version__ = "0.1.0"

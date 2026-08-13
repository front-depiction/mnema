"""mnema: constant-size semantic memory. state = fold(entries); supersession = subtraction."""

from .store import Entry, Store, merge

__all__ = ["Entry", "Store", "merge"]
__version__ = "0.1.0"

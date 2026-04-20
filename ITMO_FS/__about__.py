from pathlib import Path

__all__ = ["__title__", "__uri__", "__version__"]

__title__ = "ITMO_FS"
__uri__ = "https://github.com/ctlab/ITMO_FS"

__version__ = Path(__file__).with_name("VERSION").read_text(encoding="utf-8").strip()

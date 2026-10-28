"""Build the 1024-pixel training cache: python -m filseg.prepare --cache DIR"""
import argparse
from pathlib import Path

from .data import build_cache, data_root

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    a = ap.parse_args()
    if (a.cache / "meta.json").exists():
        print("cache exists:", a.cache)
    else:
        build_cache(data_root(), a.cache)
        print("cache built:", a.cache)

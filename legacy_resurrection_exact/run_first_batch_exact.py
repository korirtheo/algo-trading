"""Run the first-batch resurrection under the exact canonical execution model."""
from pathlib import Path
import shutil

import research.legacy_long_resurrection.run_baseline as rb
from legacy_resurrection_exact.execution import install

install(rb)

import research.legacy_long_resurrection.run_standalone as standalone
import research.legacy_long_resurrection.optimize_first_batch as search

OUT=Path("results/legacy_resurrection_exact_20261004")


def _copy(name):
    src=Path("results/legacy_long_resurrection_baseline_20261004")/name
    if src.exists():
        OUT.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src,OUT/name)


def main():
    standalone.main()
    _copy("standalone_summary.json")
    search.main()
    _copy("first_batch_causal_search.json")


if __name__=="__main__":
    main()

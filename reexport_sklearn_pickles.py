"""
Re-save sklearn pickle artifacts with the currently installed scikit-learn version.
Eliminates InconsistentVersionWarning when loading models trained on sklearn 1.7.x.
"""

import pickle
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
PICKLE_DIRS = [
    PROJECT_ROOT / "models",
    PROJECT_ROOT / "bert_model_product",
    PROJECT_ROOT / "bert_model_issue",
]


def reexport_pickle(path: Path) -> None:
    with open(path, "rb") as f:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            obj = pickle.load(f)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[OK] Re-exported {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    count = 0
    for directory in PICKLE_DIRS:
        if not directory.is_dir():
            print(f"[WARN] Skipping missing directory: {directory.name}/")
            continue
        for path in sorted(directory.glob("*.pkl")):
            reexport_pickle(path)
            count += 1
    print(f"\n[OK] Re-exported {count} pickle file(s). Restart the Flask app to load them.")


if __name__ == "__main__":
    main()

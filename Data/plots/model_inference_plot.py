import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    try:
        return Path(__file__).resolve().parents[2]
    except NameError:
        return Path.cwd()


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Data.plots.plots import get_default_model_inference_plot_path as get_default_plot_path
from Data.plots.plots import model_inference_main, plot_model_inference


if __name__ == "__main__":
    model_inference_main()

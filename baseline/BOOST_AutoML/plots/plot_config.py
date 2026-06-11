import re
from pathlib import Path


PLOTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PLOTS_DIR.parent
RESULTS_DIR = PROJECT_ROOT / "Results"
ABLATION_RESULTS_DIR = RESULTS_DIR / "Ablation Study"
FIGURES_DIR = PLOTS_DIR / "figures"
METHOD_PANELS_DIR = FIGURES_DIR / "method_comparison"
ABLATION_FIGURES_DIR = FIGURES_DIR / "ablation_study"
EXPORT_SUFFIXES = (".png",)


def slugify(value: str) -> str:
    lowered = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lowered)
    return re.sub(r"_+", "_", normalized).strip("_")


def save_figure(fig, output_dir: Path, stem: str, dpi: int = 300) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []
    for suffix in EXPORT_SUFFIXES:
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        saved_paths.append(path)
    return saved_paths

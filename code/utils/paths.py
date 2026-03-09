from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
CODE_DIR = PROJECT_ROOT / "code"
DATASETS_DIR = PROJECT_ROOT / "datasets"
CONFIGS_DIR = PROJECT_ROOT / "configs"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
VIDEOS_DIR = OUTPUTS_DIR / "videos"
LOGS_DIR = OUTPUTS_DIR / "logs"

SCENE_DATASETS_DIR = DATASETS_DIR / "scene_datasets"
POINTNAV_DATA_DIR = DATASETS_DIR / "pointnav"
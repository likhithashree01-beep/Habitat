from typing import Optional

from habitat.config.default import get_config
from omegaconf import OmegaConf

from code.utils.paths import POINTNAV_DATA_DIR, SCENE_DATASETS_DIR


def resolve_pointnav_split() -> str:
    base = POINTNAV_DATA_DIR / "gibson"
    for split in ("val_mini", "val", "train"):
        candidate = base / split / f"{split}.json.gz"
        if candidate.exists():
            return split
    raise FileNotFoundError(
        f"No PointNav Gibson episode file found under {base}"
    )


def make_pointnav_gibson_config(split: Optional[str] = None, semantic: bool = False):
    selected_split = split or resolve_pointnav_split()

    cfg = get_config("benchmark/nav/pointnav/pointnav_gibson.yaml")
    OmegaConf.set_readonly(cfg, False)
    OmegaConf.set_struct(cfg, False)

    cfg.habitat.dataset.data_path = str(
        POINTNAV_DATA_DIR / "gibson" / "{split}" / "{split}.json.gz"
    )
    cfg.habitat.dataset.split = selected_split
    cfg.habitat.dataset.scenes_dir = str(SCENE_DATASETS_DIR)

    if semantic:
        cfg.habitat.simulator.agents.main_agent.sim_sensors.semantic_sensor = {
            "type": "HabitatSimSemanticSensor",
            "height": 256,
            "width": 256,
            "position": [0.0, 1.25, 0.0],
            "orientation": [0.0, 0.0, 0.0],
            "hfov": 90,
            "sensor_subtype": "PINHOLE",
        }

    OmegaConf.set_struct(cfg, True)
    OmegaConf.set_readonly(cfg, True)
    return cfg, selected_split

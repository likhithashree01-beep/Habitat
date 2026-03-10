import numpy as np
import imageio.v2 as imageio
import habitat_sim

from code.utils.paths import DATASETS_DIR, VIDEOS_DIR


def to_rgb8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def depth_to_uint8(depth: np.ndarray, clip_max_m: float = 10.0) -> np.ndarray:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    d = depth.astype(np.float32, copy=False)
    d = np.clip(d, 0.0, clip_max_m) / float(clip_max_m)
    d8 = (d * 255.0).astype(np.uint8)
    return np.repeat(d8[..., None], 3, axis=2)


def semantic_to_color(sem: np.ndarray) -> np.ndarray:
    if sem.ndim == 3 and sem.shape[-1] == 1:
        sem = sem[..., 0]
    x = sem.astype(np.uint32, copy=False)
    r = (x * 97) & 255
    g = (x * 57) & 255
    b = (x * 137) & 255
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def main():
    dataset_cfg = DATASETS_DIR / "replica_cad" / "replicaCAD.scene_dataset_config.json"
    if not dataset_cfg.exists():
        raise FileNotFoundError(f"Missing: {dataset_cfg}")

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = str(dataset_cfg)
    sim_cfg.scene_id = "apt_1"
    sim_cfg.enable_physics = True

    height, width = 256, 256
    hfov = 90.0

    def make_sensor(uuid, sensor_type):
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid = uuid
        spec.sensor_type = sensor_type
        spec.resolution = [height, width]
        spec.position = [0.0, 1.25, 0.0]
        spec.hfov = hfov
        spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        return spec

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        make_sensor("rgb", habitat_sim.SensorType.COLOR),
        make_sensor("depth", habitat_sim.SensorType.DEPTH),
        make_sensor("semantic", habitat_sim.SensorType.SEMANTIC),
    ]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    agent = sim.initialize_agent(0)

    try:
        if sim.pathfinder.is_loaded:
            state = agent.get_state()
            state.position = sim.pathfinder.get_random_navigable_point()
            agent.set_state(state)
    except Exception:
        pass

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEOS_DIR / "replicacad_rgb_depth_semantic.mp4"
    writer = imageio.get_writer(out_path, fps=30)

    try:
        for _ in range(300):
            obs = sim.get_sensor_observations()
            combo = np.concatenate(
                [
                    to_rgb8(obs["rgb"]),
                    depth_to_uint8(obs["depth"]),
                    semantic_to_color(obs["semantic"]),
                ],
                axis=1,
            )
            writer.append_data(combo)
            sim.step(np.random.choice(["move_forward", "turn_left", "turn_right"]))
    finally:
        writer.close()
        sim.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()

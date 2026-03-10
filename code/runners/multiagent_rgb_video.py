import imageio.v2 as imageio
import numpy as np

import habitat_sim
from habitat_sim.agent import ActionSpec, ActuationSpec, AgentConfiguration, AgentState
from habitat_sim.sensor import CameraSensorSpec, SensorSubType, SensorType

from code.utils.paths import DATASETS_DIR, VIDEOS_DIR


def make_rgb_sensor(uuid: str, height: int = 256, width: int = 256) -> CameraSensorSpec:
    spec = CameraSensorSpec()
    spec.uuid = uuid
    spec.sensor_type = SensorType.COLOR
    spec.sensor_subtype = SensorSubType.PINHOLE
    spec.resolution = [height, width]
    spec.position = [0.0, 1.5, 0.0]
    spec.hfov = 90.0
    return spec


def make_agent_config(sensor_uuid: str) -> AgentConfiguration:
    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [make_rgb_sensor(sensor_uuid)]
    agent_cfg.action_space = {
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=15.0)),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=15.0)),
    }
    return agent_cfg


def initialize_agent_on_navmesh(sim: habitat_sim.Simulator, agent_id: int, yaw_deg: float):
    state = AgentState()
    if sim.pathfinder.is_loaded:
        state.position = sim.pathfinder.get_random_navigable_point()
    else:
        state.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sim.initialize_agent(agent_id, state)
    agent = sim.get_agent(agent_id)
    current_state = agent.get_state()
    yaw = np.deg2rad(yaw_deg)
    current_state.rotation = habitat_sim.utils.common.quat_from_angle_axis(
        yaw, np.array([0.0, 1.0, 0.0])
    )
    agent.set_state(current_state)
    return agent


def main():
    scene_path = DATASETS_DIR / "test_scenes" / "skokloster-castle.glb"
    if not scene_path.exists():
        raise FileNotFoundError(
            f"Missing scene: {scene_path}. Run notebooks/01_env_setup.ipynb first."
        )

    np.random.seed(7)

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    sim_cfg.enable_physics = False

    cfg = habitat_sim.Configuration(
        sim_cfg,
        [make_agent_config("rgb"), make_agent_config("rgb")],
    )
    sim = habitat_sim.Simulator(cfg)

    agent0 = initialize_agent_on_navmesh(sim, 0, yaw_deg=0.0)
    agent1 = initialize_agent_on_navmesh(sim, 1, yaw_deg=180.0)

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEOS_DIR / "multiagent_rgb_agents_0_1.mp4"
    writer = imageio.get_writer(out_path, fps=10)

    actions = ["move_forward", "turn_left", "turn_right"]

    try:
        for step_idx in range(120):
            if step_idx > 0:
                agent0.act(actions[step_idx % len(actions)])
                agent1.act(actions[(step_idx + 1) % len(actions)])

            observations = sim.get_sensor_observations(agent_ids=[0, 1])
            frame0 = observations[0]["rgb"]
            frame1 = observations[1]["rgb"]

            if frame0.shape[-1] == 4:
                frame0 = frame0[:, :, :3]
            if frame1.shape[-1] == 4:
                frame1 = frame1[:, :, :3]

            combo = np.concatenate([frame0, frame1], axis=1)
            writer.append_data(combo.astype(np.uint8))

        print("Saved:", out_path)
        print("Agent 0 final position:", agent0.get_state().position)
        print("Agent 1 final position:", agent1.get_state().position)
    finally:
        writer.close()
        sim.close()


if __name__ == "__main__":
    main()

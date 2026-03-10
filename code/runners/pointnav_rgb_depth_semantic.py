import numpy as np
import imageio.v2 as imageio

import habitat

from code.utils.paths import VIDEOS_DIR
from code.utils.pointnav import make_pointnav_gibson_config


def depth_to_uint8(depth: np.ndarray, clip_max_m: float = 10.0) -> np.ndarray:
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    d = depth.astype(np.float32, copy=False)
    if np.nanmax(d) <= 1.0 + 1e-6:
        d = np.clip(d, 0.0, 1.0)
    else:
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


def ensure_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.dtype == np.uint8:
        return rgb
    return np.clip(rgb, 0, 255).astype(np.uint8)


def main():
    cfg, split = make_pointnav_gibson_config(semantic=True)

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEOS_DIR / f"pointnav_gibson_{split}_rgb_depth_semantic.mp4"

    env = habitat.Env(config=cfg)
    obs = env.reset()
    print("Obs keys:", obs.keys())

    writer = imageio.get_writer(out_path, fps=10)

    try:
        frames_written = 0
        while frames_written < 200:
            combo = np.concatenate(
                [
                    ensure_uint8_rgb(obs["rgb"]),
                    depth_to_uint8(obs["depth"]),
                    semantic_to_color(obs["semantic"]),
                ],
                axis=1,
            )
            writer.append_data(combo)
            frames_written += 1

            step_out = env.step(env.action_space.sample())
            if bool(getattr(env, "episode_over", False)):
                obs = env.reset()
            else:
                obs = step_out[0] if isinstance(step_out, tuple) else step_out
    finally:
        writer.close()
        env.close()

    print("Saved:", out_path)


if __name__ == "__main__":
    main()

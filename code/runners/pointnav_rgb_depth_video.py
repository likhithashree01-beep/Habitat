import os

import imageio
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import track

import habitat
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from code.utils.paths import VIDEOS_DIR
from code.utils.pointnav import make_pointnav_gibson_config

console = Console()


def pick_key(obs, needle: str) -> str:
    needle = needle.lower()
    for key in obs.keys():
        if needle in key.lower():
            return key
    raise KeyError(f"Couldn't find '{needle}' in obs keys: {list(obs.keys())}")


def depth_to_vis(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth).squeeze()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    if hi - lo < 1e-6:
        norm = np.zeros_like(d, dtype=np.float32)
    else:
        norm = np.clip((d - lo) / (hi - lo), 0.0, 1.0)

    vis = (norm * 255).astype(np.uint8)
    return np.repeat(vis[..., None], 3, axis=2)


def main():
    cfg, split = make_pointnav_gibson_config()

    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = VIDEOS_DIR / f"pointnav_gibson_{split}_rgb_depth.mp4"

    console.print(
        Panel.fit(
            f"[bold]PointNav Gibson ({split})[/bold]\n"
            f"Saving RGB|Depth video -> [cyan]{out_path}[/cyan]",
            border_style="green",
        )
    )

    env = habitat.Env(config=cfg)
    obs = env.reset()

    rgb_key = pick_key(obs, "rgb")
    depth_key = pick_key(obs, "depth")
    console.print(
        f"Obs keys: rgb=[yellow]{rgb_key}[/yellow], depth=[yellow]{depth_key}[/yellow]"
    )

    follower = ShortestPathFollower(env.sim, goal_radius=0.2, return_one_hot=False)
    writer = imageio.get_writer(out_path, fps=10)

    try:
        for _ in track(range(300), description="Recording episode..."):
            rgb = obs[rgb_key]
            depth = obs[depth_key]

            if rgb.shape[-1] == 4:
                rgb = rgb[:, :, :3]
            if rgb.dtype != np.uint8:
                rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            frame = np.concatenate([rgb, depth_to_vis(depth)], axis=1)
            writer.append_data(frame)

            goal_pos = env.current_episode.goals[0].position
            action = follower.get_next_action(goal_pos)
            if action is None:
                break

            step_out = env.step(action)
            obs = step_out[0] if isinstance(step_out, tuple) else step_out

            if bool(getattr(env, "episode_over", False)):
                break
    finally:
        writer.close()
        env.close()

    console.print(f"[bold green]Saved:[/bold green] {out_path}")


if __name__ == "__main__":
    main()

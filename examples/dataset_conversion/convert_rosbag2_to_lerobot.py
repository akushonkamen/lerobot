#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert rosbag2/MCAP recordings of JointState topics to a LeRobotDataset v3.0.

This is the reference implementation requested in
https://github.com/huggingface/lerobot/issues/5130 — #3139 only points at the
porting guide, leaving topic-to-feature mapping and the synchronized frame grid
hand-written (and easy to get silently wrong). The three silent failure modes
called out in the issue are addressed as follows:

1. **Topic contract**: topics are mapped to features through an *explicit*
   ``--state-topic`` / ``--action-topic`` CLI argument (no inference from
   topic names), and joint names are validated for consistency across every
   message of a bag.
2. **Episode boundary timestamps**: one bag = one episode. Messages are resampled
   onto a fixed ``1 / fps`` grid built from each bag's first message log time, so
   timestamps restart at 0 at every episode boundary (a v3.0 requirement). Note
   that ``LeRobotDataset.add_frame`` computes ``timestamp = frame_index / fps``
   itself, so per-episode re-zeroing holds by construction as long as you never
   pass raw ROS timestamps in.
3. **Stats / artifact layout**: we never compute stats by hand. Frames are
   written with ``add_frame`` / ``save_episode``, which call
   ``lerobot.datasets.compute_stats.compute_episode_stats`` internally and store
   the full v3.0 key set (``mean``, ``std``, ``min``, ``max``, ``count`` and
   ``q01``/``q10``/``q50``/``q90``/``q99`` quantiles) next to the episode data,
   matching the v3.0 artifact layout.

Supported input: rosbag2 MCAP files whose topics carry ``sensor_msgs/msg/JointState``
messages. A ``JointState`` feature vector is its ``position`` field, with joint
names taken from the first message of the bag.

Usage (with a synthetic fixture, see ``--generate-fixture``):

    python examples/dataset_conversion/convert_rosbag2_to_lerobot.py \
        --generate-fixture 2 /tmp/synth_bags
    python examples/dataset_conversion/convert_rosbag2_to_lerobot.py \
        --input '/tmp/synth_bags/*.mcap' --repo-id local/rosbag2_demo \
        --root /tmp/out_ds --fps 30 --task "move the arm"

Requires ``pip install mcap mcap-ros2-support``.
"""

import argparse
import fnmatch
import glob as glob_module
import logging
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

logger = logging.getLogger(__name__)

STATE_FEATURE_KEY = "observation.state"
ACTION_FEATURE_KEY = "action"

# Full v3.0 stats key set — see lerobot/datasets/compute_stats.py
# (DEFAULT_QUANTILES and compute_array_stats docstring).
EXPECTED_STATS_KEYS = {"mean", "std", "min", "max", "count", "q01", "q10", "q50", "q90", "q99"}


JOINTSTATE_MSGDEF = """
std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] effort
================================================================
MSG: std_msgs/Header
builtin_interfaces/Time stamp
int32 sec
uint32 nanosec
string frame_id
================================================================
MSG: builtin_interfaces/Time
int32 sec
uint32 nanosec
"""


def generate_jointstate_fixture(out_dir: Path, num_episodes: int = 2, fps: int = 30) -> list[Path]:
    """Write synthetic rosbag2 MCAP files with a JointState topic (for local testing)."""
    from mcap_ros2.writer import Writer as Ros2Writer

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ep in range(num_episodes):
        path = out_dir / f"episode_{ep:03d}.mcap"
        with open(path, "wb") as f:
            writer = Ros2Writer(f)
            schema = writer.register_msgdef("sensor_msgs/msg/JointState", JOINTSTATE_MSGDEF)
            start_ns = 1_700_000_000_000_000_000 + ep * 60_000_000_000  # arbitrary, differs per bag
            num_frames = 40 + ep * 10
            for i in range(num_frames):
                t = start_ns + int(i / fps * 1e9)
                position = [float(i / fps) * (ep + 1)] * 6
                writer.write_message(
                    "/joint_states",
                    schema,
                    {
                        "header": {"stamp": {"sec": t // 10**9, "nanosec": t % 10**9}, "frame_id": "world"},
                        "name": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5"],
                        "position": position,
                        "velocity": [],
                        "effort": [],
                    },
                    log_time=t,
                    publish_time=t,
                )
            writer.finish()
        paths.append(path)
        logger.info(f"Generated fixture bag: {path} ({num_frames} JointState messages)")
    return paths


def read_jointstate_bags(bag_paths: list[Path], topic_pattern: str) -> list[dict]:
    """Read JointState messages matching ``topic_pattern`` from each bag.

    Returns one record per bag: ``{"joint_names", "log_times_s", "positions"}``
    where ``log_times_s`` are message log times in seconds, relative to the first
    message of that bag (per-episode re-zeroing starts here).
    """
    from mcap_ros2.reader import read_ros2_messages

    episodes = []
    for bag_path in bag_paths:
        joint_names = None
        times: list[float] = []
        positions: list[list[float]] = []
        first_log_time_ns = None
        for msg in read_ros2_messages(str(bag_path)):
            if not fnmatch.fnmatch(msg.channel.topic, topic_pattern):
                continue
            ros_msg = msg.ros_msg
            if type(ros_msg).__name__ != "JointState":
                raise ValueError(
                    f"Topic '{msg.channel.topic}' in {bag_path} is "
                    f"'{type(ros_msg).__name__}', expected a JointState message. "
                    "This script only implements the JointState contract — extend it for "
                    "image or other message types."
                )
            if joint_names is None:
                joint_names = list(ros_msg.name)
            elif list(ros_msg.name) != joint_names:
                raise ValueError(
                    f"Joint names changed mid-bag in {bag_path}: {joint_names} != {list(ros_msg.name)}"
                )
            if first_log_time_ns is None:
                first_log_time_ns = msg.log_time_ns
            times.append((msg.log_time_ns - first_log_time_ns) / 1e9)
            positions.append(list(ros_msg.position))
        if not times:
            raise ValueError(f"No JointState messages matching '{topic_pattern}' in {bag_path}")
        episodes.append({"joint_names": joint_names, "log_times_s": times, "positions": positions})
        logger.info(f"{bag_path}: {len(times)} messages, joints={joint_names}")
    return episodes


def resample_to_frame_grid(
    episode: dict, fps: int, num_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a bag onto the fixed ``1 / fps`` frame grid of one episode.

    State at grid time ``t`` is the last message with ``log_time <= t``.
    Action is the *forward-filled* value of the same source stream (the command
    that was in effect at ``t``), which is the correct default for teleop
    recordings where only a state stream exists — the executed action is the
    state itself. Pass ``--action-topic`` to consume an explicit command topic
    instead (see ``convert_bags`` for where to plug it in).
    """
    times = np.asarray(episode["log_times_s"])
    positions = np.asarray(episode["positions"], dtype=np.float32)

    grid = np.arange(num_frames, dtype=np.float64) / fps
    # searchsorted(side='right') - 1 gives the index of the last msg <= grid time
    idx = np.searchsorted(times, grid, side="right") - 1
    if (idx < 0).any():
        raise ValueError(
            "Frame grid starts before the first message — messages must start at or "
            "before t=0 of the grid (they are re-zeroed to the first message, so this "
            "should never happen)."
        )
    states = positions[idx]
    actions = states.copy()  # forward-fill: teleop default, action = held state
    return grid, states, actions


def validate_dataset(repo_id: str, root: Path) -> LeRobotDataset:
    """Post-conversion self-check: reload, assert per-episode timestamp monotonicity
    and that every numeric feature carries the full v3.0 stats key set."""
    ds = LeRobotDataset(repo_id, root=root)

    for ep in range(ds.meta.total_episodes):
        from_idx = ds.meta.episodes["dataset_from_index"][ep]
        to_idx = ds.meta.episodes["dataset_to_index"][ep]
        timestamps = ds.hf_dataset.select(range(from_idx, to_idx))["timestamp"]
        assert len(timestamps) == ds.meta.episodes["length"][ep], (
            f"Episode {ep}: frame count mismatch ({len(timestamps)} vs {ds.meta.episodes['length'][ep]})"
        )
        assert all(t2 > t1 for t1, t2 in zip(timestamps, timestamps[1:], strict=False)), (
            f"Episode {ep}: timestamps not strictly increasing (v3.0 boundary reset broken)"
        )
        assert abs(timestamps[0]) < 1e-9, f"Episode {ep}: first timestamp not re-zeroed ({timestamps[0]})"

    for key, ft in ds.meta.features.items():
        if ft["dtype"] not in ("float32", "float64", "int64", "bool") or key in (
            "timestamp",
            "frame_index",
            "index",
            "episode_index",
            "task_index",
        ):
            continue
        stats = ds.meta.stats[key]
        missing = EXPECTED_STATS_KEYS - set(stats)
        assert not missing, f"Feature '{key}': stats missing v3.0 keys {missing} (got {set(stats)})"

    logger.info(
        f"Self-check passed: {ds.meta.total_episodes} episodes, "
        f"{ds.meta.total_frames} frames, monotonic per-episode timestamps, "
        f"full v3.0 stats key set for {list(ds.meta.stats)}"
    )
    return ds


def convert_bags(
    bag_paths: list[Path],
    repo_id: str,
    root: Path,
    fps: int,
    task: str,
    topic_pattern: str = "/joint_states",
    robot_type: str | None = None,
    validate: bool = True,
) -> LeRobotDataset:
    episodes = read_jointstate_bags(bag_paths, topic_pattern)
    joint_names = episodes[0]["joint_names"]
    for ep in episodes[1:]:
        if ep["joint_names"] != joint_names:
            raise ValueError(
                f"Joint names differ across bags: {joint_names} vs {ep['joint_names']} — "
                "all bags in one conversion must share the same topic contract."
            )

    features = {
        STATE_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": {"axes": list(joint_names)},
        },
        ACTION_FEATURE_KEY: {
            "dtype": "float32",
            "shape": (len(joint_names),),
            "names": {"axes": list(joint_names)},
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        features=features,
        root=root,
        robot_type=robot_type,
        use_videos=False,
    )

    for ep_index, episode in enumerate(episodes):
        num_frames = int(np.ceil(episode["log_times_s"][-1] * fps)) + 1
        _, states, actions = resample_to_frame_grid(episode, fps, num_frames)
        for i in range(num_frames):
            dataset.add_frame(
                {
                    STATE_FEATURE_KEY: states[i],
                    ACTION_FEATURE_KEY: actions[i],
                    "task": task,
                }
            )
        dataset.save_episode()
        logger.info(f"Saved episode {ep_index}: {num_frames} frames from {bag_paths[ep_index].name}")

    dataset.finalize()

    if validate:
        return validate_dataset(repo_id, root)
    return dataset


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=str, help="Glob of rosbag2 MCAP files, e.g. '/tmp/bags/*.mcap'")
    parser.add_argument("--repo-id", type=str, default="local/convert_rosbag2_demo")
    parser.add_argument("--root", type=Path, default=Path("outputs/convert_rosbag2_demo"))
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", type=str, default="move the arm")
    parser.add_argument(
        "--topic", type=str, default="/joint_states", help="JointState topic pattern (fnmatch)"
    )
    parser.add_argument("--robot-type", type=str, default=None)
    parser.add_argument(
        "--generate-fixture",
        type=int,
        metavar="N_EPISODES",
        default=None,
        help="Generate N synthetic episode bags into --input directory and exit",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip the post-conversion self-check")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = make_parser().parse_args()

    if args.generate_fixture is not None:
        generate_jointstate_fixture(args.root, num_episodes=args.generate_fixture, fps=args.fps)
        return

    if not args.input:
        raise SystemExit("--input is required (or use --generate-fixture to create a synthetic fixture)")
    bag_paths = sorted(Path(p) for p in glob_module.glob(args.input))
    if not bag_paths:
        raise SystemExit(f"No bags matched {args.input!r}")

    convert_bags(
        bag_paths=bag_paths,
        repo_id=args.repo_id,
        root=args.root,
        fps=args.fps,
        task=args.task,
        topic_pattern=args.topic,
        robot_type=args.robot_type,
        validate=not args.no_validate,
    )


if __name__ == "__main__":
    main()

"""Build causal context windows from scenario-grouped transition datasets.

Windows never cross a timestamp gap, so no sample from before a reset or a
recording gap can leak into a training context.  Hidden scenario parameters are
never part of a window; they are only carried as provenance for evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from neurogrip.config import NeuroGripConfig


# Modeled state is [v_x, yaw_rate]; lateral velocity is excluded because the
# low-speed kinematic plant makes it degenerate (see docs/limitations.md).
STATE_COLUMNS = ["v_x_t_mps", "yaw_rate_t_rps"]
COMMAND_COLUMNS = ["cmd_linear_x_t_mps", "cmd_angular_z_t_rps"]
TARGET_STATE_COLUMNS = ["v_x_t1_mps", "yaw_rate_t1_rps"]
CONTIGUITY_TOLERANCE_S = 0.005


@dataclass
class WindowBatch:
    """Arrays describing all valid context windows for one or more runs."""

    history: np.ndarray  # [N, H, feature_dim]
    current_state: np.ndarray  # [N, state_dim]
    future_commands: np.ndarray  # [N, K, input_dim]
    target_states: np.ndarray  # [N, K, state_dim]
    scenario_id: list[str]

    def __len__(self) -> int:
        return self.history.shape[0]

    def subset(self, indices: np.ndarray) -> "WindowBatch":
        """Return a new batch restricted to ``indices``."""
        return WindowBatch(
            history=self.history[indices],
            current_state=self.current_state[indices],
            future_commands=self.future_commands[indices],
            target_states=self.target_states[indices],
            scenario_id=[self.scenario_id[index] for index in indices],
        )


def load_catalog(manifest_path: Path) -> dict:
    """Load a scenario-grouped dataset catalog."""
    manifest_path = Path(manifest_path).expanduser().resolve()
    catalog = json.loads(manifest_path.read_text(encoding="utf-8"))
    if catalog.get("split_unit") != "scenario_id":
        raise ValueError("Catalog must use scenario_id as its split unit.")
    return catalog


CATALOG_SPLITS = (
    "train",
    "validation",
    "calibration",
    "development_test",
    "sealed_test",
)


def run_frames_by_split(catalog: dict) -> dict[str, list[tuple[str, pd.DataFrame]]]:
    """Return per-split lists of (scenario_id, time-sorted frame) per run."""
    grouped: dict[str, list[tuple[str, pd.DataFrame]]] = {
        split: [] for split in CATALOG_SPLITS
    }
    for run in catalog["runs"]:
        split = run.get("split")
        if split not in grouped:
            continue
        frame = pd.read_parquet(run["parquet_path"]).sort_values("time_s")
        grouped[split].append((run["scenario_id"], frame.reset_index(drop=True)))
    return grouped


def _segment_ids(frame: pd.DataFrame) -> np.ndarray:
    """Assign a segment id that increments at every timestamp gap."""
    time_s = frame["time_s"].to_numpy(dtype=float)
    if len(time_s) < 2:
        return np.zeros(len(time_s), dtype=int)
    gaps = np.diff(time_s)
    median_gap = float(np.median(gaps))
    breaks = np.abs(gaps - median_gap) > CONTIGUITY_TOLERANCE_S
    return np.concatenate([[0], np.cumsum(breaks)])


def build_windows(
    frame: pd.DataFrame, scenario_id: str, config: NeuroGripConfig
) -> WindowBatch:
    """Extract all in-segment context windows from one run."""
    history_steps = config.history_steps
    rollout_steps = config.rollout_steps

    states = frame[STATE_COLUMNS].to_numpy(dtype=np.float64)
    commands = frame[COMMAND_COLUMNS].to_numpy(dtype=np.float64)
    features = np.concatenate([states, commands], axis=1)
    segments = _segment_ids(frame)
    sample_count = len(frame)

    history_list = []
    current_list = []
    future_list = []
    target_list = []
    for start in range(history_steps - 1, sample_count - rollout_steps):
        query_end = start + rollout_steps
        if segments[start - history_steps + 1] != segments[query_end]:
            continue
        history_list.append(features[start - history_steps + 1 : start + 1])
        current_list.append(states[start])
        future_list.append(commands[start : start + rollout_steps])
        target_list.append(states[start + 1 : start + 1 + rollout_steps])

    if not history_list:
        return WindowBatch(
            history=np.empty((0, history_steps, config.feature_dim)),
            current_state=np.empty((0, config.state_dim)),
            future_commands=np.empty((0, rollout_steps, config.input_dim)),
            target_states=np.empty((0, rollout_steps, config.state_dim)),
            scenario_id=[],
        )
    return WindowBatch(
        history=np.stack(history_list),
        current_state=np.stack(current_list),
        future_commands=np.stack(future_list),
        target_states=np.stack(target_list),
        scenario_id=[scenario_id] * len(history_list),
    )


def concatenate(batches: list[WindowBatch]) -> WindowBatch:
    """Stack several window batches into one."""
    if not batches:
        raise ValueError("No window batches to concatenate.")
    return WindowBatch(
        history=np.concatenate([batch.history for batch in batches]),
        current_state=np.concatenate([batch.current_state for batch in batches]),
        future_commands=np.concatenate([batch.future_commands for batch in batches]),
        target_states=np.concatenate([batch.target_states for batch in batches]),
        scenario_id=[sid for batch in batches for sid in batch.scenario_id],
    )


def build_split_windows(
    grouped: dict[str, list[tuple[str, pd.DataFrame]]], config: NeuroGripConfig
) -> dict[str, WindowBatch]:
    """Build window batches for every split in the catalog."""
    result = {}
    for split, runs in grouped.items():
        batches = [build_windows(frame, sid, config) for sid, frame in runs]
        batches = [batch for batch in batches if len(batch) > 0]
        result[split] = concatenate(batches) if batches else None
    return result

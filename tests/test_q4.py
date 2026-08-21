"""Focused event-level checks for Question 4 multi-UAV controls."""

from __future__ import annotations

import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np

EVENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVENT_ROOT))

from src.q3 import _sampled_metrics, _settings_margin  # noqa: E402
from src.q4 import (  # noqa: E402
    Q4Strategy,
    UAVControl,
    _bomb_geometries,
    _feasibility_violation,
)


def _config() -> dict[str, object]:
    return json.loads((EVENT_ROOT / "configs/q4.json").read_text(encoding="utf-8"))


def test_each_uav_uses_its_own_initial_position_and_ballistics() -> None:
    config = _config()
    controls = (
        UAVControl(0.0, 70.0, 0.0, 1.0),
        UAVControl(np.pi / 2.0, 80.0, 2.0, 0.5),
        UAVControl(np.pi, 90.0, 3.0, 2.0),
    )
    bombs = _bomb_geometries(Q4Strategy(controls), 9.8, config)

    assert np.allclose(bombs[0].release_point, [17800.0, 0.0, 1800.0])
    assert np.allclose(bombs[1].release_point, [12000.0, 1560.0, 1400.0])
    assert np.allclose(bombs[2].release_point, [5730.0, -3000.0, 700.0])
    assert bombs[0].explosion_point[2] == 1800.0 - 4.9
    assert bombs[1].explosion_point[2] == 1400.0 - 4.9 * 0.5**2
    assert bombs[2].explosion_point[2] == 700.0 - 4.9 * 2.0**2


def test_feasible_strategy_has_no_constraint_violation() -> None:
    config = _config()
    strategy = Q4Strategy(
        (
            UAVControl(3.1, 70.0, 0.0, 2.5),
            UAVControl(3.0, 100.0, 1.0, 2.0),
            UAVControl(2.8, 120.0, 2.0, 3.0),
        )
    )

    assert _feasibility_violation(strategy, 9.8, config) == 0.0


def test_joint_proxy_is_invariant_to_uav_label_order() -> None:
    config = _config()
    strategy = Q4Strategy(
        (
            UAVControl(3.08, 70.0, 0.0, 2.5),
            UAVControl(3.0, 90.0, 1.0, 2.5),
            UAVControl(2.8, 110.0, 2.0, 3.0),
        )
    )
    bombs = _bomb_geometries(strategy, 9.8, config)
    settings = {"surface_angles": 32, "surface_levels": 5}
    durations = []
    for order in permutations(bombs):
        margin = _settings_margin(order, settings, refine=False)
        duration, _ = _sampled_metrics(order, margin, 0.25)
        durations.append(duration)

    assert np.allclose(durations, durations[0], atol=1e-12, rtol=0.0)

"""Focused event-level checks for Question 5 multi-missile geometry."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

EVENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVENT_ROOT))

from src.q4 import Q4Strategy, UAVControl  # noqa: E402
from src.q4 import _bomb_geometries as q4_bomb_geometries  # noqa: E402
from src.q5 import (  # noqa: E402
    MISSILE_NAMES,
    Q5Strategy,
    UAVPlan,
    _bomb_geometries,
    _feasibility_violation,
    _find_intervals,
    _formal_evaluation,
    _missile_initials,
    _q4_anchor_strategy,
    missile_hit_time,
    missile_position,
)


def _config() -> dict[str, object]:
    return json.loads((EVENT_ROOT / "configs/q5.json").read_text(encoding="utf-8"))


def test_arbitrary_missile_trajectory_reaches_origin_at_its_own_hit_time() -> None:
    config = _config()
    hits = []
    for name, initial in zip(MISSILE_NAMES, _missile_initials(config), strict=True):
        hit = missile_hit_time(initial)
        hits.append(hit)
        assert np.allclose(missile_position(initial, 0.0), initial)
        assert np.allclose(missile_position(initial, hit), np.zeros(3), atol=1e-10)
        assert name in {"M1", "M2", "M3"}

    assert hits[0] > hits[1] > hits[2]


def test_five_uav_parameterization_enforces_release_spacing() -> None:
    config = _config()
    plan = UAVPlan(3.0, 100.0, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0)
    strategy = Q5Strategy((plan, plan, plan, plan, plan))
    bombs = _bomb_geometries(strategy, config)

    assert len(bombs) == 15
    assert plan.release_times_s == (1.0, 2.0, 3.0)
    assert _feasibility_violation(strategy, config) == 0.0


def test_q5_m1_kernel_regresses_formal_q4_strategy() -> None:
    config = _config()
    validation = config["validation"]
    strategy = Q4Strategy(
        tuple(UAVControl(*(float(value) for value in row)) for row in validation["q4_strategy"])
    )
    q4_config = {
        "gravity": config["gravity"],
        "uavs": {name: config["uavs"][name] for name in ("FY1", "FY2", "FY3")},
    }
    bombs = q4_bomb_geometries(strategy, float(config["gravity"]), q4_config)
    intervals = _find_intervals(
        bombs,
        _missile_initials(config)[0],
        config["formal_evaluation"],
        refine=True,
    )
    duration = sum(interval.duration for interval in intervals)

    assert (
        abs(duration - validation["q4_regression_expected_s"])
        <= validation["q4_regression_tolerance_s"]
    )


def test_q4_anchor_keeps_all_three_verified_m1_bombs_exactly() -> None:
    config = _config()
    dummy = {
        (3, 1): np.array([3.0, 100.0, 5.0, 2.0]),
        (4, 2): np.array([3.2, 110.0, 7.0, 2.5]),
    }
    strategy = _q4_anchor_strategy(dummy, config)
    bombs = _bomb_geometries(strategy, config)

    for uav_index, row in enumerate(config["validation"]["q4_strategy"]):
        theta, speed, release, fuse = row
        plan = strategy.plans[uav_index]
        anchor = bombs[3 * uav_index]
        assert (plan.theta_rad, plan.speed_m_s) == (theta, speed)
        assert anchor.release_time_s == release
        assert anchor.fuse_delay_s == fuse


def test_candidate_ranking_can_skip_expensive_deletion_recomputation() -> None:
    config = _config()
    plan = UAVPlan(3.0, 100.0, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0)
    strategy = Q5Strategy((plan, plan, plan, plan, plan))

    evaluation = _formal_evaluation(strategy, config, compute_deletion=False)

    assert evaluation.deletion_losses_s == ((0.0, 0.0, 0.0),) * 15

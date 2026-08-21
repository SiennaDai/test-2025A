"""Focused event-level checks for Question 3 joint coverage semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EVENT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVENT_ROOT))

from src.q3 import (  # noqa: E402
    BombGeometry,
    Q3Strategy,
    _bomb_geometries,
    _find_joint_intervals,
    _sampled_metrics,
    _settings_margin,
    joint_coverage_margin,
)


def _bomb(index: int, explosion_time: float, point: np.ndarray) -> BombGeometry:
    return BombGeometry(
        bomb=index,
        release_time_s=explosion_time,
        fuse_delay_s=0.0,
        explosion_time_s=explosion_time,
        release_point=tuple(point),
        explosion_point=tuple(point),
    )


def test_event_is_only_a_split_point_when_root_occurs_after_activation() -> None:
    bomb = _bomb(1, 1.0, np.zeros(3))
    intervals = _find_joint_intervals(
        lambda time: 1.2 - time,
        (bomb,),
        scan_step=0.1,
        root_tolerance=1e-10,
    )

    assert intervals[0].start == 1.2
    assert intervals[0].start != bomb.explosion_time_s


def test_two_incomplete_clouds_can_jointly_cover_all_sampled_rays() -> None:
    time = 4.768
    center = np.array([17625.525268056128, 10.240444866222791, 1762.640057120904])
    bombs = tuple(
        _bomb(index, time, center + np.array([0.0, sign * 9.65, 0.0]))
        for index, sign in enumerate((-1.0, 1.0), 1)
    )
    settings = {
        "surface_angles": 360,
        "surface_levels": 41,
        "continuous_refinement": True,
        "refinement_starts": 1,
        "refinement_maxfev": 50,
    }
    individual = [joint_coverage_margin(time, (bomb,), **settings) for bomb in bombs]
    joint = joint_coverage_margin(time, bombs, **settings)

    assert all(margin > 0.0 for margin in individual)
    assert joint <= 0.0


def test_inactive_third_bomb_does_not_shift_absolute_time_sampling() -> None:
    strategy = Q3Strategy(
        3.1047922065753095,
        75.53740845287045,
        0.00008659727834377369,
        0.0001070796609000733,
        0.0000817521129715,
        2.7648584468708908,
        2.58163903013058,
        9.334353521045797,
    )
    first_two = _bomb_geometries(strategy, 9.8)[:2]
    far_point = np.array([100000.0, 100000.0, 100000.0])
    third_early = _bomb(3, 12.0, far_point)
    third_late = _bomb(3, 17.0, far_point)
    settings = {"surface_angles": 36, "surface_levels": 5}

    durations = []
    for third in (third_early, third_late):
        bombs = (*first_two, third)
        margin = _settings_margin(bombs, settings, refine=False)
        duration, _ = _sampled_metrics(bombs, margin, 0.1)
        durations.append(duration)

    assert durations[0] == durations[1]

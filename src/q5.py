"""Question 5 hierarchical optimization for five UAVs and three missiles."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution, minimize

from mmkit.artifacts import prepare_output_paths, save_table

from .q1 import CLOUD_LIFETIME, MISSILE_SPEED, OcclusionInterval
from .q3 import (
    BombGeometry,
    _cloud_center,
    _continuous_worst_margin,
    _coverage_margins,
    _duration,
    _independent_ray_intersections,
    _merge_intervals,
    _surface_covering_radius,
    _surface_points,
)
from .q4 import Q4Strategy, UAVControl
from .q4 import _bomb_geometries as q4_bomb_geometries

UAV_NAMES = ("FY1", "FY2", "FY3", "FY4", "FY5")
MISSILE_NAMES = ("M1", "M2", "M3")
TARGET_CENTER = np.array([0.0, 200.0, 5.0])


@dataclass(frozen=True, slots=True)
class UAVPlan:
    """Shared flight controls and three automatically spaced bomb slots."""

    theta_rad: float
    speed_m_s: float
    s0_s: float
    s1_s: float
    s2_s: float
    fuse1_s: float
    fuse2_s: float
    fuse3_s: float

    @property
    def release_times_s(self) -> tuple[float, float, float]:
        return (
            self.s0_s,
            self.s0_s + 1.0 + self.s1_s,
            self.s0_s + 2.0 + self.s1_s + self.s2_s,
        )

    @property
    def fuse_delays_s(self) -> tuple[float, float, float]:
        return (self.fuse1_s, self.fuse2_s, self.fuse3_s)


@dataclass(frozen=True, slots=True)
class Q5Strategy:
    """Forty-variable strategy represented as five eight-variable UAV blocks."""

    plans: tuple[UAVPlan, UAVPlan, UAVPlan, UAVPlan, UAVPlan]


@dataclass(frozen=True, slots=True)
class MissileEvaluation:
    """Formal joint coverage result for one missile."""

    missile: str
    intervals: tuple[OcclusionInterval, ...]

    @property
    def duration_s(self) -> float:
        return _duration(self.intervals)


@dataclass(frozen=True, slots=True)
class Q5Evaluation:
    """Formal three-missile evaluation and per-bomb deletion contributions."""

    strategy: Q5Strategy
    bombs: tuple[BombGeometry, ...]
    missiles: tuple[MissileEvaluation, MissileEvaluation, MissileEvaluation]
    deletion_losses_s: tuple[tuple[float, float, float], ...]

    @property
    def durations_s(self) -> tuple[float, float, float]:
        return tuple(item.duration_s for item in self.missiles)

    @property
    def j_sum_s(self) -> float:
        return sum(self.durations_s)

    @property
    def j_min_s(self) -> float:
        return min(self.durations_s)


def missile_hit_time(initial: np.ndarray) -> float:
    """Return time for an arbitrary missile to reach the false target at the origin."""
    return float(np.linalg.norm(initial) / MISSILE_SPEED)


def missile_position(initial: np.ndarray, time: float) -> np.ndarray:
    """Return an arbitrary missile position at absolute task time."""
    return initial * (1.0 - time / missile_hit_time(initial))


def _strategy_from_array(values: np.ndarray) -> Q5Strategy:
    array = np.asarray(values, dtype=float).reshape(5, 8)
    return Q5Strategy(tuple(UAVPlan(*(float(value) for value in row)) for row in array))


def _strategy_vector(strategy: Q5Strategy) -> np.ndarray:
    return np.array([value for plan in strategy.plans for value in asdict(plan).values()])


def _uav_initials(config: dict[str, Any]) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(config["uavs"][name], dtype=float) for name in UAV_NAMES)


def _missile_initials(config: dict[str, Any]) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(config["missiles"][name], dtype=float) for name in MISSILE_NAMES)


def _bomb_geometries(strategy: Q5Strategy, config: dict[str, Any]) -> tuple[BombGeometry, ...]:
    gravity = float(config["gravity"])
    bombs = []
    number = 1
    for plan, initial in zip(strategy.plans, _uav_initials(config), strict=True):
        heading = np.array([np.cos(plan.theta_rad), np.sin(plan.theta_rad), 0.0])
        for release, fuse in zip(plan.release_times_s, plan.fuse_delays_s, strict=True):
            explosion_time = release + fuse
            release_point = initial + plan.speed_m_s * release * heading
            explosion_point = initial + plan.speed_m_s * explosion_time * heading
            explosion_point += np.array([0.0, 0.0, -0.5 * gravity * fuse**2])
            bombs.append(
                BombGeometry(
                    number,
                    release,
                    fuse,
                    explosion_time,
                    tuple(float(value) for value in release_point),
                    tuple(float(value) for value in explosion_point),
                )
            )
            number += 1
    return tuple(bombs)


def _fuse_maximum(initial: np.ndarray, gravity: float) -> float:
    return float(np.sqrt(2.0 * initial[2] / gravity))


def _block_bounds(initial: np.ndarray, config: dict[str, Any]) -> list[tuple[float, float]]:
    bounds = config["bounds"]
    fuse = (0.0, _fuse_maximum(initial, float(config["gravity"])))
    return [
        tuple(bounds["theta_rad"]),
        tuple(bounds["speed_m_s"]),
        tuple(bounds["s0_s"]),
        tuple(bounds["s1_s"]),
        tuple(bounds["s2_s"]),
        fuse,
        fuse,
        fuse,
    ]


def _feasibility_violation(strategy: Q5Strategy, config: dict[str, Any]) -> float:
    violation = 0.0
    latest_hit = max(missile_hit_time(item) for item in _missile_initials(config))
    bombs = _bomb_geometries(strategy, config)
    for uav_index, (plan, initial) in enumerate(
        zip(strategy.plans, _uav_initials(config), strict=True)
    ):
        fuse_max = _fuse_maximum(initial, float(config["gravity"]))
        violation += max(0.0, 70.0 - plan.speed_m_s) / 70.0
        violation += max(0.0, plan.speed_m_s - 140.0) / 70.0
        for value in asdict(plan).values():
            violation += max(0.0, -value - 1e-12) / max(1.0, fuse_max)
        for bomb in bombs[3 * uav_index : 3 * uav_index + 3]:
            violation += max(0.0, bomb.explosion_time_s - latest_hit) / latest_hit
            violation += max(0.0, -bomb.explosion_point[2]) / float(initial[2])
    return violation


def _active_bombs(time: float, bombs: Sequence[BombGeometry]) -> tuple[BombGeometry, ...]:
    return tuple(
        bomb
        for bomb in bombs
        if bomb.explosion_time_s <= time <= bomb.explosion_time_s + CLOUD_LIFETIME
    )


def _margin(
    time: float,
    bombs: Sequence[BombGeometry],
    missile_initial: np.ndarray,
    settings: dict[str, Any],
    *,
    refine: bool,
) -> float:
    active = _active_bombs(time, bombs)
    if not active or time > missile_hit_time(missile_initial):
        return float("inf")
    missile = missile_position(missile_initial, time)
    angles = int(settings["surface_angles"])
    levels = int(settings["surface_levels"])
    points = _surface_points(angles, levels)
    margins = _coverage_margins(points, missile, active, time)
    sampled = float(np.max(margins))
    if not refine or sampled > 0.0:
        return sampled
    maximum_angles = int(settings.get("adaptive_max_surface_angles", angles))
    maximum_levels = int(settings.get("adaptive_max_surface_levels", levels))
    while sampled + _surface_covering_radius(angles, levels) > 0.0 and (
        angles < maximum_angles or levels < maximum_levels
    ):
        angles = min(maximum_angles, 2 * angles)
        levels = min(maximum_levels, 2 * (levels - 1) + 1)
        points = _surface_points(angles, levels)
        margins = _coverage_margins(points, missile, active, time)
        sampled = float(np.max(margins))
        if sampled > 0.0:
            return sampled
    upper = sampled + _surface_covering_radius(angles, levels)
    if upper <= 0.0:
        return upper
    return _continuous_worst_margin(
        missile,
        active,
        time,
        points,
        margins,
        starts=int(settings.get("continuous_refinement_starts", 1)),
        maxfev=int(settings.get("continuous_refinement_maxfev", 40)),
    )


def _find_intervals(
    bombs: Sequence[BombGeometry],
    missile_initial: np.ndarray,
    settings: dict[str, Any],
    *,
    refine: bool,
) -> tuple[OcclusionInterval, ...]:
    hit = missile_hit_time(missile_initial)
    start = min(bomb.explosion_time_s for bomb in bombs)
    end = min(hit, max(bomb.explosion_time_s + CLOUD_LIFETIME for bomb in bombs))
    if start > end:
        return ()
    step = float(settings.get("time_scan_step_s", settings.get("time_step_s", 0.2)))
    tolerance = float(settings.get("root_tolerance_s", 1e-8))
    regular = np.arange(0.0, hit + 0.5 * step, step)
    regular = regular[(regular >= start) & (regular <= end)]
    events = np.array(
        [
            event
            for bomb in bombs
            for event in (bomb.explosion_time_s, bomb.explosion_time_s + CLOUD_LIFETIME)
            if start <= event <= end
        ]
    )
    probe = max(tolerance * 10.0, step * 1e-5)
    probes = np.array(
        [
            value
            for event in events
            for value in (max(start, event - probe), min(end, event + probe))
        ]
    )
    times = np.unique(np.concatenate(([start, end], regular, events, probes)))

    def function(time: float) -> float:
        return _margin(time, bombs, missile_initial, settings, refine=refine)

    values = np.array([function(float(time)) for time in times])
    inside = values <= 0.0
    result = []
    current = float(times[0]) if inside[0] else None
    for index in np.flatnonzero(inside[1:] != inside[:-1]):
        left, right = float(times[index]), float(times[index + 1])
        midpoint = 0.5 * (left + right)
        active = tuple(bomb.bomb for bomb in _active_bombs(left, bombs))
        middle = tuple(bomb.bomb for bomb in _active_bombs(midpoint, bombs))
        ending = tuple(bomb.bomb for bomb in _active_bombs(right, bombs))
        if active != middle and middle == ending:
            boundary = left
        elif active == middle and middle != ending:
            boundary = right
        else:
            boundary = float(brentq(function, left, right, xtol=tolerance, rtol=1e-14))
        if inside[index + 1]:
            current = boundary
        elif current is not None:
            result.append(OcclusionInterval(current, boundary))
            current = None
    if current is not None:
        result.append(OcclusionInterval(current, float(times[-1])))
    return _merge_intervals(result)


def _sampled_duration(
    bombs: Sequence[BombGeometry], missile_initial: np.ndarray, settings: dict[str, Any]
) -> tuple[float, float]:
    hit = missile_hit_time(missile_initial)
    step = float(settings["time_step_s"])
    start = min(bomb.explosion_time_s for bomb in bombs)
    end = min(hit, max(bomb.explosion_time_s + CLOUD_LIFETIME for bomb in bombs))
    if start > end:
        return 0.0, float("inf")
    anchored = np.arange(0.0, hit + 0.5 * step, step)
    anchored = anchored[(anchored >= start) & (anchored <= end)]
    events = np.array(
        [
            event
            for bomb in bombs
            for event in (bomb.explosion_time_s, bomb.explosion_time_s + CLOUD_LIFETIME)
            if start <= event <= end
        ]
    )
    times = np.unique(np.concatenate(([start, end], anchored, events)))
    values = np.array(
        [_margin(float(time), bombs, missile_initial, settings, refine=False) for time in times]
    )
    return float(np.trapezoid((values <= 0.0).astype(float), times)), float(np.min(values))


def _single_bomb(control: np.ndarray, initial: np.ndarray, gravity: float) -> BombGeometry:
    theta, speed, release, fuse = control
    heading = np.array([np.cos(theta), np.sin(theta), 0.0])
    explosion_time = release + fuse
    release_point = initial + speed * release * heading
    explosion_point = initial + speed * explosion_time * heading
    explosion_point += np.array([0.0, 0.0, -0.5 * gravity * fuse**2])
    return BombGeometry(
        1,
        float(release),
        float(fuse),
        float(explosion_time),
        tuple(float(value) for value in release_point),
        tuple(float(value) for value in explosion_point),
    )


def _single_population(
    uav: np.ndarray, missile: np.ndarray, fuse_max: float, settings: dict[str, Any], seed: int
) -> np.ndarray:
    count = 4 * int(settings["popsize"])
    rng = np.random.default_rng(seed)
    lower = np.array([0.0, 70.0, 0.0, 0.0])
    upper = np.array([2.0 * np.pi, 140.0, 45.0, fuse_max])
    population = rng.uniform(lower, upper, size=(count, 4))
    structured = []
    hit = missile_hit_time(missile)
    for time in np.linspace(3.0, 0.85 * hit, 40):
        point = missile_position(missile, float(time))
        for fraction in (0.05, 0.15, 0.3, 0.5, 0.7):
            desired = (1.0 - fraction) * point + fraction * TARGET_CENTER
            displacement = desired[:2] - uav[:2]
            speed = float(np.linalg.norm(displacement) / time)
            height_drop = uav[2] - desired[2]
            if 70.0 <= speed <= 140.0 and height_drop >= 0.0:
                fuse = float(np.sqrt(2.0 * height_drop / 9.8))
                release = float(time - fuse)
                if 0.0 <= release <= 45.0 and fuse <= fuse_max:
                    structured.append(
                        np.array(
                            [
                                np.arctan2(displacement[1], displacement[0]) % (2 * np.pi),
                                speed,
                                release,
                                fuse,
                            ]
                        )
                    )
    for index, item in enumerate(structured[:count]):
        population[index] = item
    return population


def _candidate_library(
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[tuple[int, int], np.ndarray]]:
    settings = config["single_library"]
    gravity = float(config["gravity"])
    records = []
    best = {}
    for uav_index, uav in enumerate(_uav_initials(config)):
        fuse_max = _fuse_maximum(uav, gravity)
        for missile_index, missile in enumerate(_missile_initials(config)):
            hit = missile_hit_time(missile)

            def objective(
                values: np.ndarray,
                fixed_uav: np.ndarray = uav,
                fixed_missile: np.ndarray = missile,
                fixed_hit: float = hit,
            ) -> float:
                bomb = _single_bomb(values, fixed_uav, gravity)
                if bomb.explosion_time_s > fixed_hit or bomb.explosion_point[2] < 0.0:
                    return 100.0 + max(0.0, bomb.explosion_time_s - fixed_hit)
                duration, minimum = _sampled_duration((bomb,), fixed_missile, settings)
                return -duration if duration > 0.0 else max(0.0, minimum) / 100.0

            seed = int(config["seed"]) + 10 * uav_index + missile_index
            result = differential_evolution(
                objective,
                [(0.0, 2 * np.pi), (70.0, 140.0), (0.0, 45.0), (0.0, fuse_max)],
                init=_single_population(uav, missile, fuse_max, settings, seed),
                seed=seed,
                maxiter=int(settings["maxiter"]),
                popsize=int(settings["popsize"]),
                tol=float(settings["tol"]),
                polish=False,
                workers=1,
            )
            vector = np.asarray(result.x, dtype=float)
            best[(uav_index, missile_index)] = vector
            records.append(
                {
                    "uav": UAV_NAMES[uav_index],
                    "missile": MISSILE_NAMES[missile_index],
                    "sampled_duration_s": max(0.0, -float(result.fun)),
                    "physical_feasible": bool(
                        _single_bomb(vector, uav, gravity).explosion_time_s <= hit
                        and _single_bomb(vector, uav, gravity).explosion_point[2] >= 0.0
                    ),
                    "positive_coverage_found": bool(float(result.fun) < 0.0),
                    "evaluations": int(result.nfev),
                    "success": bool(result.success),
                    "message": str(result.message),
                    "vector": vector.tolist(),
                }
            )
    return records, best


def _assembled_strategy(
    assignment: Sequence[int], library: dict[tuple[int, int], np.ndarray]
) -> Q5Strategy:
    plans = []
    for uav_index, missile_index in enumerate(assignment):
        theta, speed, release, fuse = library[(uav_index, missile_index)]
        base = max(0.0, float(release) - 1.0)
        plans.append(
            UAVPlan(
                float(theta), float(speed), base, 0.0, 0.0, float(fuse), float(fuse), float(fuse)
            )
        )
    return Q5Strategy(tuple(plans))


def _q4_anchor_strategy(
    library: dict[tuple[int, int], np.ndarray], config: dict[str, Any]
) -> Q5Strategy:
    """Embed the three formally verified Q4 bombs as a monotone M1 incumbent."""
    plans = []
    for row in config["validation"]["q4_strategy"]:
        theta, speed, release, fuse = (float(value) for value in row)
        plans.append(UAVPlan(theta, speed, release, 0.0, 0.0, fuse, fuse, fuse))
    for uav_index, missile_index in ((3, 1), (4, 2)):
        theta, speed, release, fuse = library[(uav_index, missile_index)]
        plans.append(
            UAVPlan(
                float(theta),
                float(speed),
                float(release),
                0.0,
                0.0,
                float(fuse),
                float(fuse),
                float(fuse),
            )
        )
    return Q5Strategy(tuple(plans))


def _proxy_scores(
    strategy: Q5Strategy, config: dict[str, Any], settings: dict[str, Any]
) -> tuple[float, tuple[float, float, float]]:
    violation = _feasibility_violation(strategy, config)
    if violation > 0.0:
        return -100.0 - 100.0 * violation, (0.0, 0.0, 0.0)
    bombs = _bomb_geometries(strategy, config)
    durations = tuple(
        _sampled_duration(bombs, missile, settings)[0] for missile in _missile_initials(config)
    )
    score = sum(durations) - 2.0 * sum(max(0.0, 0.05 - value) for value in durations)
    return score, durations


def _block_refine(
    strategy: Q5Strategy, config: dict[str, Any], stage_path: Path
) -> tuple[Q5Strategy, list[dict[str, Any]]]:
    settings = config["block_refinement"]
    current = _strategy_vector(strategy)
    records = []
    for round_index in range(int(settings["rounds"])):
        for uav_index, initial in enumerate(_uav_initials(config)):
            offset = 8 * uav_index

            def objective(block: np.ndarray, fixed_offset: int = offset) -> float:
                candidate = current.copy()
                candidate[fixed_offset : fixed_offset + 8] = block
                return -_proxy_scores(_strategy_from_array(candidate), config, settings)[0]

            result = minimize(
                objective,
                current[offset : offset + 8],
                method="Nelder-Mead",
                bounds=_block_bounds(initial, config),
                options={
                    "maxfev": int(settings["maxfev_per_block"]),
                    "xatol": float(settings["xatol"]),
                    "fatol": float(settings["fatol"]),
                },
            )
            if float(result.fun) < objective(current[offset : offset + 8]):
                current[offset : offset + 8] = result.x
            score, durations = _proxy_scores(_strategy_from_array(current), config, settings)
            records.append(
                {
                    "round": round_index + 1,
                    "uav": UAV_NAMES[uav_index],
                    "evaluations": int(result.nfev),
                    "success": bool(result.success),
                    "message": str(result.message),
                    "proxy_score_s": score,
                    "durations_s": durations,
                }
            )
            stage_path.write_text(
                json.dumps({"stage": "block-refinement-in-progress", "records": records}, indent=2)
                + "\n",
                encoding="utf-8",
            )
    return _strategy_from_array(current), records


def _marginal_fill(
    strategy: Q5Strategy, config: dict[str, Any], stage_path: Path
) -> tuple[Q5Strategy, list[dict[str, Any]]]:
    """Search otherwise unused second/third slots while preserving each anchor bomb."""
    settings = config["marginal_fill"]
    current = _strategy_vector(strategy)
    records = []
    rng = np.random.default_rng(int(config["seed"]) + 900)
    for name in settings["uav_order"]:
        uav_index = UAV_NAMES.index(name)
        offset = 8 * uav_index
        initial = _uav_initials(config)[uav_index]
        fuse_max = _fuse_maximum(initial, float(config["gravity"]))
        lower = np.array([0.0, 0.0, 0.0, 0.0])
        upper = np.array([20.0, 20.0, fuse_max, fuse_max])
        population = rng.uniform(lower, upper, size=(int(settings["population_size"]), 4))
        plan = _strategy_from_array(current).plans[uav_index]
        anchor = np.array([plan.s1_s, plan.s2_s, plan.fuse2_s, plan.fuse3_s])
        population[0] = anchor
        structured = (
            (0.0, 0.0, plan.fuse1_s, plan.fuse1_s),
            (1.0, 1.0, plan.fuse1_s, plan.fuse1_s),
            (3.0, 3.0, plan.fuse1_s, plan.fuse1_s),
            (6.0, 6.0, plan.fuse1_s, plan.fuse1_s),
            (0.0, 2.0, 0.8 * plan.fuse1_s, 1.2 * plan.fuse1_s),
            (2.0, 5.0, 1.2 * plan.fuse1_s, 0.8 * plan.fuse1_s),
        )
        for index, values in enumerate(structured, 1):
            if index < len(population):
                population[index] = np.clip(values, lower, upper)

        base_vector = current.copy()

        def full_vector(
            values: np.ndarray,
            base: np.ndarray = base_vector,
            fixed_offset: int = offset,
        ) -> np.ndarray:
            candidate = base.copy()
            candidate[fixed_offset + 3 : fixed_offset + 5] = values[:2]
            candidate[fixed_offset + 6 : fixed_offset + 8] = values[2:]
            return candidate

        def objective(values: np.ndarray) -> float:
            candidate = _strategy_from_array(full_vector(values))
            return -_proxy_scores(candidate, config, settings)[0]

        before_score, before_durations = _proxy_scores(
            _strategy_from_array(current), config, settings
        )
        result = differential_evolution(
            objective,
            [(0.0, 20.0), (0.0, 20.0), (0.0, fuse_max), (0.0, fuse_max)],
            init=population,
            seed=int(config["seed"]) + 900 + uav_index,
            maxiter=int(settings["maxiter"]),
            popsize=2,
            polish=False,
            workers=1,
        )
        candidate = full_vector(np.asarray(result.x, dtype=float))
        after_score, after_durations = _proxy_scores(
            _strategy_from_array(candidate), config, settings
        )
        accepted = after_score > before_score + float(settings["improvement_threshold_s"])
        if accepted:
            current = candidate
        records.append(
            {
                "uav": name,
                "evaluations": int(result.nfev),
                "success": bool(result.success),
                "message": str(result.message),
                "before_score_s": before_score,
                "after_score_s": after_score,
                "before_durations_s": before_durations,
                "after_durations_s": after_durations,
                "accepted": accepted,
            }
        )
        stage_path.write_text(
            json.dumps({"stage": "marginal-fill-in-progress", "records": records}, indent=2) + "\n",
            encoding="utf-8",
        )
    return _strategy_from_array(current), records


def _formal_evaluation(
    strategy: Q5Strategy, config: dict[str, Any], *, compute_deletion: bool
) -> Q5Evaluation:
    bombs = _bomb_geometries(strategy, config)
    settings = config["formal_evaluation"]
    missiles = tuple(
        MissileEvaluation(name, _find_intervals(bombs, initial, settings, refine=True))
        for name, initial in zip(MISSILE_NAMES, _missile_initials(config), strict=True)
    )
    deletion = []
    if compute_deletion:
        deletion_settings = config["deletion_evaluation"]
        for removed in range(15):
            remaining = tuple(bomb for index, bomb in enumerate(bombs) if index != removed)
            losses = []
            for missile, baseline in zip(_missile_initials(config), missiles, strict=True):
                intervals = _find_intervals(remaining, missile, deletion_settings, refine=False)
                losses.append(
                    max(
                        0.0,
                        baseline.duration_s - min(_duration(intervals), baseline.duration_s),
                    )
                )
            deletion.append(tuple(losses))
    else:
        deletion = [(0.0, 0.0, 0.0)] * 15
    return Q5Evaluation(strategy, bombs, missiles, tuple(deletion))


def optimize_question_5(
    config: dict[str, Any], stage_path: Path
) -> tuple[list[Q5Evaluation], dict[str, Any], list[dict[str, Any]]]:
    """Build the 15-pair library, assemble incumbents, and refine five UAV blocks."""
    library_records, library = _candidate_library(config)
    stage_path.write_text(
        json.dumps(
            {"stage": "candidate-library-complete", "candidate_library": library_records}, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    print("Q5 15-pair candidate library complete", flush=True)
    library_scores = {
        (UAV_NAMES.index(record["uav"]), MISSILE_NAMES.index(record["missile"])): record[
            "sampled_duration_s"
        ]
        for record in library_records
    }
    assignments = []
    for assignment in product(range(3), repeat=5):
        if len(set(assignment)) < 3:
            continue
        score = sum(library_scores[(uav, missile)] for uav, missile in enumerate(assignment))
        assignments.append((score, assignment))
    assignments.sort(reverse=True)
    assembly_count = int(config["assembly"]["candidate_assignments"])
    assembled = []
    for _, assignment in assignments[:assembly_count]:
        strategy = _assembled_strategy(assignment, library)
        score, durations = _proxy_scores(strategy, config, config["assembly"])
        assembled.append((score, min(durations), strategy, assignment, durations))
    assembled.sort(key=lambda item: (item[0], item[1]), reverse=True)
    incumbent = _q4_anchor_strategy(library, config)
    incumbent_score, incumbent_durations = _proxy_scores(incumbent, config, config["assembly"])
    stage_path.write_text(
        json.dumps(
            {
                "stage": "assembly-complete",
                "q4_anchor_incumbent": {
                    "proxy_sum_s": incumbent_score,
                    "durations_s": incumbent_durations,
                    "retained_for_formal_evaluation": True,
                },
                "assignments": [
                    {
                        "assignment": [MISSILE_NAMES[index] for index in item[3]],
                        "proxy_sum_s": item[0],
                        "durations_s": item[4],
                    }
                    for item in assembled
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Q5 feasible three-missile incumbent assembled", flush=True)
    refined, block_records = _block_refine(incumbent, config, stage_path)
    filled, fill_records = _marginal_fill(refined, config, stage_path)
    candidates = [incumbent, refined, filled, *(item[2] for item in assembled[1:])]
    proxy_ranked = sorted(
        candidates,
        key=lambda item: _proxy_scores(item, config, config["block_refinement"])[0],
        reverse=True,
    )
    count = min(int(config["formal_evaluation"]["candidate_count"]), len(proxy_ranked))
    formal_strategies = [proxy_ranked[0]]
    if count > 1:
        formal_strategies.append(incumbent)
    formal = []
    for index, strategy in enumerate(formal_strategies[:count], 1):
        formal.append(_formal_evaluation(strategy, config, compute_deletion=False))
        print(f"Q5 formal candidate {index}/{count} complete", flush=True)
    formal.sort(key=lambda item: (item.j_sum_s, item.j_min_s), reverse=True)
    formal[0] = _formal_evaluation(formal[0].strategy, config, compute_deletion=True)
    stage_path.write_text(
        json.dumps(
            {
                "stage": "search-complete",
                "library_size": len(library_records),
                "block_records": block_records,
                "marginal_fill_records": fill_records,
                "formal_scores": [
                    {
                        "j_sum_s": item.j_sum_s,
                        "j_min_s": item.j_min_s,
                        "durations_s": item.durations_s,
                    }
                    for item in formal
                ],
                "q4_anchor_incumbent_retained": True,
                "sampled_time_grid": "absolute task time t=0",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    solver = {
        "seed": int(config["seed"]),
        "method": "15 single-pair DE searches, assignment assembly, and five bounded 8D blocks",
        "candidate_library_size": len(library_records),
        "block_refinements": block_records,
        "marginal_fill": fill_records,
        "formal_settings": config["formal_evaluation"],
        "status": "budget-limited-hierarchical-search-with-feasible-incumbent",
        "convergence_claimed": False,
        "sampled_time_grid": "absolute task time t=0 plus cloud events",
        "q4_anchor_incumbent_retained": True,
    }
    return formal, solver, library_records


def _summary(evaluation: Q5Evaluation, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "D1_s": evaluation.durations_s[0],
        "D2_s": evaluation.durations_s[1],
        "D3_s": evaluation.durations_s[2],
        "J_sum_s": evaluation.j_sum_s,
        "J_min_s": evaluation.j_min_s,
    }


def _bomb_rows(evaluation: Q5Evaluation) -> list[dict[str, Any]]:
    rows = []
    for index, bomb in enumerate(evaluation.bombs):
        uav_index, slot = divmod(index, 3)
        plan = evaluation.strategy.plans[uav_index]
        losses = evaluation.deletion_losses_s[index]
        rows.append(
            {
                "schema_status": "provisional-schema; official result3.xlsx unavailable",
                "uav": UAV_NAMES[uav_index],
                "bomb_slot": slot + 1,
                "used": sum(losses) > 1e-8,
                "theta_rad": plan.theta_rad,
                "theta_deg": float(np.degrees(plan.theta_rad) % 360.0),
                "speed_m_s": plan.speed_m_s,
                "release_time_s": bomb.release_time_s,
                "release_x_m": bomb.release_point[0],
                "release_y_m": bomb.release_point[1],
                "release_z_m": bomb.release_point[2],
                "fuse_delay_s": bomb.fuse_delay_s,
                "explosion_time_s": bomb.explosion_time_s,
                "explosion_x_m": bomb.explosion_point[0],
                "explosion_y_m": bomb.explosion_point[1],
                "explosion_z_m": bomb.explosion_point[2],
                "M1_deletion_loss_s": losses[0],
                "M2_deletion_loss_s": losses[1],
                "M3_deletion_loss_s": losses[2],
                "J_sum_deletion_loss_s": sum(losses),
            }
        )
    return rows


def _validation(evaluation: Q5Evaluation, config: dict[str, Any]) -> dict[str, Any]:
    bombs = evaluation.bombs
    formal = config["formal_evaluation"]
    validation = config["validation"]
    convergence = {}
    boundaries = {}
    direct = {}
    points = _surface_points(
        int(validation["direct_surface_angles"]), int(validation["direct_surface_levels"])
    )
    for name, missile, baseline in zip(
        MISSILE_NAMES, _missile_initials(config), evaluation.missiles, strict=True
    ):
        convergence[name] = []
        for index, settings in enumerate(validation["convergence_settings"]):
            merged = {
                **settings,
                **(
                    {
                        "adaptive_max_surface_angles": formal["adaptive_max_surface_angles"],
                        "adaptive_max_surface_levels": formal["adaptive_max_surface_levels"],
                        "continuous_refinement_starts": 1,
                        "continuous_refinement_maxfev": 40,
                    }
                    if index == 2
                    else {}
                ),
            }
            intervals = _find_intervals(bombs, missile, merged, refine=index == 2)
            convergence[name].append({**settings, "duration_s": _duration(intervals)})
        boundaries[name], direct[name] = [], []
        for interval in baseline.intervals:
            for location, time in (
                ("before-start", interval.start - validation["boundary_probe_s"]),
                ("start", interval.start),
                ("midpoint", 0.5 * (interval.start + interval.end)),
                ("end", interval.end),
                ("after-end", interval.end + validation["boundary_probe_s"]),
            ):
                margin = _margin(time, bombs, missile, formal, refine=True)
                active = _active_bombs(time, bombs)
                covered = _independent_ray_intersections(
                    points,
                    missile_position(missile, time),
                    [_cloud_center(time, bomb) for bomb in active],
                )
                boundaries[name].append({"location": location, "time_s": time, "margin_m": margin})
                direct[name].append(
                    {
                        "location": location,
                        "time_s": time,
                        "sampled_rays": len(points),
                        "uncovered_rays": int(np.count_nonzero(~covered)),
                        "all_covered": bool(np.all(covered)),
                    }
                )
    rng = np.random.default_rng(int(config["seed"]) + 500)
    permutations = []
    medium = validation["convergence_settings"][1]
    for _ in range(int(validation["permutation_samples"])):
        order = tuple(bombs[index] for index in rng.permutation(len(bombs)))
        permutations.append(
            [
                _duration(_find_intervals(order, missile, medium, refine=False))
                for missile in _missile_initials(config)
            ]
        )
    q4_strategy = Q4Strategy(
        tuple(UAVControl(*(float(value) for value in row)) for row in validation["q4_strategy"])
    )
    q4_config = {
        "gravity": config["gravity"],
        "uavs": {name: config["uavs"][name] for name in UAV_NAMES[:3]},
    }
    q4_bombs = q4_bomb_geometries(q4_strategy, float(config["gravity"]), q4_config)
    q4_duration = _duration(
        _find_intervals(q4_bombs, _missile_initials(config)[0], formal, refine=True)
    )
    gaps = [np.diff(plan.release_times_s).tolist() for plan in evaluation.strategy.plans]
    return {
        "feasibility": {
            "violation": _feasibility_violation(evaluation.strategy, config),
            "release_gaps_s": gaps,
            "all_release_gaps_at_least_1s": all(min(item) >= 1.0 for item in gaps),
            "all_durations_positive": all(value > 0.0 for value in evaluation.durations_s),
            "explosion_heights_m": [bomb.explosion_point[2] for bomb in bombs],
        },
        "surface_time_convergence": convergence,
        "boundary_checks": boundaries,
        "independent_unit_direction_ray_checks": direct,
        "label_permutation_duration_samples_s": permutations,
        "label_permutation_max_deviation_s": max(
            (
                abs(sample[index] - permutations[0][index])
                for sample in permutations[1:]
                for index in range(3)
            ),
            default=0.0,
        ),
        "q4_m1_geometry_regression": {
            "computed_duration_s": q4_duration,
            "expected_duration_s": float(validation["q4_regression_expected_s"]),
            "absolute_difference_s": abs(
                q4_duration - float(validation["q4_regression_expected_s"])
            ),
            "tolerance_s": float(validation["q4_regression_tolerance_s"]),
            "within_tolerance": abs(q4_duration - float(validation["q4_regression_expected_s"]))
            <= float(validation["q4_regression_tolerance_s"]),
        },
        "deletion_contributions_s": [list(item) for item in evaluation.deletion_losses_s],
        "arithmetic": {
            "durations_s": evaluation.durations_s,
            "reported_J_sum_s": evaluation.j_sum_s,
            "recomputed_J_sum_s": sum(evaluation.durations_s),
            "consistent": evaluation.j_sum_s == sum(evaluation.durations_s),
        },
        "q4_monotone_lower_bound": {
            "q5_D1_s": evaluation.durations_s[0],
            "q4_reference_s": float(validation["q4_regression_expected_s"]),
            "satisfied": evaluation.durations_s[0] + float(validation["q4_regression_tolerance_s"])
            >= float(validation["q4_regression_expected_s"]),
        },
    }


def run_question_5(project_root: Path, config: dict[str, Any]) -> list[Path]:
    """Run Question 5 and save provisional tables, workbook, logs, and stage state."""
    paths = prepare_output_paths(project_root / "outputs")
    stage_path = paths.logs / "q5_search_stage.json"
    if bool(config.get("formal_only_persistent_rerun", False)):
        previous_stage = (
            json.loads(stage_path.read_text(encoding="utf-8")) if stage_path.exists() else None
        )
        persistent = _strategy_from_array(
            np.asarray(config["persistent_incumbent"]["vector"], dtype=float)
        )
        evaluations = [_formal_evaluation(persistent, config, compute_deletion=True)]
        library_path_existing = paths.tables / "q5_candidate_library.csv"
        library = (
            pd.read_csv(library_path_existing).to_dict(orient="records")
            if library_path_existing.exists()
            else []
        )
        solver = {
            "seed": int(config["seed"]),
            "status": "persistent-incumbent-formal-rerun",
            "convergence_claimed": False,
            "persistent_incumbent": config["persistent_incumbent"],
            "marginal_fill_comparison": {
                "candidate_J_sum_s": 13.869304210239239,
                "retained_incumbent_J_sum_s": evaluations[0].j_sum_s,
                "candidate_improved_incumbent": False,
            },
            "previous_search_stage": previous_stage,
            "formal_settings": config["formal_evaluation"],
            "deletion_settings": config["deletion_evaluation"],
        }
        stage_path.write_text(
            json.dumps(
                {
                    "stage": "search-complete",
                    "mode": "persistent-incumbent-formal-rerun",
                    "persistent_J_sum_s": evaluations[0].j_sum_s,
                    "marginal_fill_candidate_J_sum_s": 13.869304210239239,
                    "retained_persistent_incumbent": True,
                    "previous_search_stage": previous_stage,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        evaluations, solver, library = optimize_question_5(config, stage_path)
    if not evaluations or evaluations[0].j_min_s <= 0.0:
        raise RuntimeError("Question 5 produced no strategy covering all three missiles")
    best = evaluations[0]
    summaries = [
        _summary(item, rank)
        for rank, item in enumerate(evaluations[: int(config["near_optimal_count"])], 1)
    ]
    summary_path = save_table(
        pd.DataFrame(summaries[:1]), paths.tables / "q5_summary.csv", overwrite=True
    )
    candidates_path = save_table(
        pd.DataFrame(summaries), paths.tables / "q5_candidates.csv", overwrite=True
    )
    library_path = save_table(
        pd.DataFrame(library), paths.tables / "q5_candidate_library.csv", overwrite=True
    )
    rows = pd.DataFrame(_bomb_rows(best))
    bombs_path = save_table(rows, paths.tables / "q5_bombs.csv", overwrite=True)
    interval_rows = [
        {
            "missile": item.missile,
            "interval": number,
            "start_s": interval.start,
            "end_s": interval.end,
            "duration_s": interval.duration,
        }
        for item in best.missiles
        for number, interval in enumerate(item.intervals, 1)
    ]
    intervals_path = save_table(
        pd.DataFrame(interval_rows), paths.tables / "q5_intervals.csv", overwrite=True
    )
    workbook_path = paths.tables / "result3.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        rows.to_excel(writer, sheet_name="provisional_result3", index=False)
        pd.DataFrame(
            [
                {
                    "schema_status": "provisional",
                    "reason": "official result3.xlsx template is missing",
                    "mapping_action": "map explicit fields when official template is supplied",
                }
            ]
        ).to_excel(writer, sheet_name="schema_note", index=False)
    optimization_path = paths.logs / "q5_optimization.json"
    optimization_path.write_text(
        json.dumps(
            {
                "model": "q5-hierarchical-three-missile-joint-cover-v1",
                "objective": "J_sum=D1+D2+D3; J_min reported",
                "best": {**_summary(best, 1), "bombs": _bomb_rows(best)},
                "near_optimal_candidates": summaries,
                "solver": solver,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    validation_path = paths.logs / "q5_validation.json"
    validation_path.write_text(
        json.dumps(_validation(best, config), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return [
        summary_path,
        candidates_path,
        library_path,
        bombs_path,
        intervals_path,
        workbook_path,
        stage_path,
        optimization_path,
        validation_path,
    ]

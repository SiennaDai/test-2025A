"""Question 3 joint line-of-sight coverage by three smoke clouds."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq, differential_evolution, minimize

from mmkit.artifacts import prepare_output_paths, save_table

from .q1 import (
    CLOUD_DESCENT_SPEED,
    CLOUD_LIFETIME,
    CLOUD_RADIUS,
    TARGET_HEIGHT,
    TARGET_RADIUS,
    UAV_INITIAL,
    OcclusionInterval,
    _dense_target_surface,
    missile_position,
)
from .q2 import Q2Strategy, missile_hit_time
from .q2 import _formal_evaluation as q2_formal_evaluation
from .q2 import explosion_point as single_explosion_point
from .q2 import release_point as single_release_point


@dataclass(frozen=True, slots=True)
class Q3Strategy:
    """Shared FY1 flight controls and three release/fuse controls."""

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
        """Return release times under the automatic one-second-gap parameterization."""
        return (
            self.s0_s,
            self.s0_s + 1.0 + self.s1_s,
            self.s0_s + 2.0 + self.s1_s + self.s2_s,
        )

    @property
    def fuse_delays_s(self) -> tuple[float, float, float]:
        """Return the three independently selected fuse delays."""
        return (self.fuse1_s, self.fuse2_s, self.fuse3_s)

    @property
    def explosion_times_s(self) -> tuple[float, float, float]:
        """Return explosion times; their ordering is deliberately unconstrained."""
        return tuple(
            release + fuse
            for release, fuse in zip(self.release_times_s, self.fuse_delays_s, strict=True)
        )


@dataclass(frozen=True, slots=True)
class BombGeometry:
    """Derived geometry for one bomb."""

    bomb: int
    release_time_s: float
    fuse_delay_s: float
    explosion_time_s: float
    release_point: tuple[float, float, float]
    explosion_point: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Q3Evaluation:
    """Formal joint and conservative evaluation of one three-bomb strategy."""

    strategy: Q3Strategy
    bombs: tuple[BombGeometry, ...]
    joint_intervals: tuple[OcclusionInterval, ...]
    individual_strict_intervals: tuple[tuple[OcclusionInterval, ...], ...]
    conservative_union_intervals: tuple[OcclusionInterval, ...]
    deletion_durations_s: tuple[float, float, float]
    ordered_incremental_durations_s: tuple[float, float, float]

    @property
    def joint_duration_s(self) -> float:
        return _duration(self.joint_intervals)

    @property
    def conservative_duration_s(self) -> float:
        return _duration(self.conservative_union_intervals)


@dataclass(frozen=True, slots=True)
class CoverageAssessment:
    """Adaptive surface certificate or boundary-level unresolved assessment."""

    status: str
    signed_margin_m: float
    sampled_lower_bound_m: float
    lipschitz_upper_bound_m: float
    surface_angles: int
    surface_levels: int
    refinement_trace: tuple[dict[str, float | int], ...]


def _strategy_from_array(values: np.ndarray) -> Q3Strategy:
    return Q3Strategy(*(float(value) for value in values))


def _bomb_strategies(strategy: Q3Strategy) -> tuple[Q2Strategy, ...]:
    return tuple(
        Q2Strategy(
            strategy.theta_rad,
            strategy.speed_m_s,
            release_time,
            fuse,
        )
        for release_time, fuse in zip(strategy.release_times_s, strategy.fuse_delays_s, strict=True)
    )


def _bomb_geometries(strategy: Q3Strategy, gravity: float) -> tuple[BombGeometry, ...]:
    rows = []
    for index, bomb in enumerate(_bomb_strategies(strategy), 1):
        release = single_release_point(bomb)
        explosion = single_explosion_point(bomb, gravity)
        rows.append(
            BombGeometry(
                bomb=index,
                release_time_s=bomb.release_time_s,
                fuse_delay_s=bomb.fuse_delay_s,
                explosion_time_s=bomb.explosion_time_s,
                release_point=tuple(float(value) for value in release),
                explosion_point=tuple(float(value) for value in explosion),
            )
        )
    return tuple(rows)


def _feasibility_violation(strategy: Q3Strategy, gravity: float) -> float:
    hit_time = missile_hit_time()
    violation = 0.0
    values = asdict(strategy).values()
    if any(value < 0.0 for value in values):
        violation += sum(max(0.0, -value) for value in values)
    for bomb in _bomb_strategies(strategy):
        violation += max(0.0, bomb.explosion_time_s - hit_time) / hit_time
        height = float(single_explosion_point(bomb, gravity)[2])
        violation += max(0.0, -height) / float(UAV_INITIAL[2])
    return violation


def _cloud_center(time: float, bomb: BombGeometry) -> np.ndarray:
    return np.asarray(bomb.explosion_point) + np.array(
        [0.0, 0.0, -CLOUD_DESCENT_SPEED * (time - bomb.explosion_time_s)]
    )


def _active_bombs(time: float, bombs: Sequence[BombGeometry]) -> tuple[BombGeometry, ...]:
    return tuple(
        bomb
        for bomb in bombs
        if bomb.explosion_time_s <= time <= bomb.explosion_time_s + CLOUD_LIFETIME
    )


@lru_cache(maxsize=24)
def _surface_points(angle_count: int, level_count: int) -> np.ndarray:
    points = _dense_target_surface(angle_count, level_count)
    points.setflags(write=False)
    return points


def _point_cloud_distances(
    points: np.ndarray, missile: np.ndarray, cloud: np.ndarray
) -> np.ndarray:
    """Return cloud-center distances to missile-to-target-point segments."""
    directions = points - missile
    squared_lengths = np.einsum("ij,ij->i", directions, directions)
    parameters = ((cloud - missile) @ directions.T) / squared_lengths
    parameters = np.clip(parameters, 0.0, 1.0)
    closest = missile + parameters[:, None] * directions
    return np.linalg.norm(closest - cloud, axis=1)


def _coverage_margins(
    points: np.ndarray,
    missile: np.ndarray,
    active_bombs: Sequence[BombGeometry],
    time: float,
) -> np.ndarray:
    """Return pointwise best-cloud margins; nonpositive means jointly covered."""
    if not active_bombs:
        return np.full(len(points), np.inf)
    distances = np.vstack(
        [
            _point_cloud_distances(points, missile, _cloud_center(time, bomb))
            for bomb in active_bombs
        ]
    )
    return np.min(distances, axis=0) - CLOUD_RADIUS


def _target_point(surface: str, first: float, second: float) -> np.ndarray:
    if surface == "side":
        angle, height = first, second
        radius = TARGET_RADIUS
        z = height
    else:
        angle, radius = first, second
        z = 0.0 if surface == "bottom" else TARGET_HEIGHT
    return np.array([radius * np.cos(angle), 200.0 + radius * np.sin(angle), z], dtype=float)


def _continuous_worst_margin(
    missile: np.ndarray,
    active: Sequence[BombGeometry],
    time: float,
    sampled_points: np.ndarray,
    sampled_margins: np.ndarray,
    *,
    starts: int,
    maxfev: int,
) -> float:
    """Refine the worst sampled ray continuously on side and both caps."""
    block_size = len(sampled_points) // 3
    order_parts = [
        offset + np.argsort(sampled_margins[offset : offset + block_size])[-starts:]
        for offset in (0, block_size, 2 * block_size)
    ]
    order = np.concatenate(order_parts)
    seeds: list[tuple[str, np.ndarray]] = []
    for index in order:
        point = sampled_points[int(index)]
        angle = float(np.arctan2(point[1] - 200.0, point[0]) % (2.0 * np.pi))
        radial = float(np.clip(np.hypot(point[0], point[1] - 200.0), 0.0, TARGET_RADIUS))
        seeds.append(("side", np.array([angle, float(np.clip(point[2], 0, TARGET_HEIGHT))])))
        if abs(point[2]) < 1e-9:
            seeds.append(("bottom", np.array([angle, radial])))
        if abs(point[2] - TARGET_HEIGHT) < 1e-9:
            seeds.append(("top", np.array([angle, radial])))

    best = float(np.max(sampled_margins))
    for surface, seed in seeds:
        upper_second = TARGET_HEIGHT if surface == "side" else TARGET_RADIUS

        def negative_margin(parameters: np.ndarray, fixed_surface: str = surface) -> float:
            point = _target_point(fixed_surface, float(parameters[0]), float(parameters[1]))
            margin = _coverage_margins(point[None, :], missile, active, time)[0]
            return -float(margin)

        result = minimize(
            negative_margin,
            seed,
            method="Nelder-Mead",
            bounds=[(0.0, 2.0 * np.pi), (0.0, upper_second)],
            options={"maxfev": maxfev, "xatol": 1e-9, "fatol": 1e-9},
        )
        best = max(best, -float(result.fun))
    return best


def _surface_covering_radius(angle_count: int, level_count: int) -> float:
    """Return a Euclidean covering-radius bound for the sampled cylinder surface."""
    angular_half_chord = 2.0 * TARGET_RADIUS * np.sin(np.pi / (2.0 * angle_count))
    linear_half_step = max(TARGET_HEIGHT, TARGET_RADIUS) / (2.0 * (level_count - 1))
    return float(np.hypot(angular_half_chord, linear_half_step))


def joint_coverage_margin(
    time: float,
    bombs: Sequence[BombGeometry],
    *,
    surface_angles: int,
    surface_levels: int,
    continuous_refinement: bool,
    refinement_starts: int = 0,
    refinement_maxfev: int = 0,
    adaptive_max_surface_angles: int | None = None,
    adaptive_max_surface_levels: int | None = None,
) -> float:
    """Return max-over-rays min-over-active-clouds coverage margin."""
    return joint_coverage_assessment(
        time,
        bombs,
        surface_angles=surface_angles,
        surface_levels=surface_levels,
        continuous_refinement=continuous_refinement,
        refinement_starts=refinement_starts,
        refinement_maxfev=refinement_maxfev,
        adaptive_max_surface_angles=adaptive_max_surface_angles,
        adaptive_max_surface_levels=adaptive_max_surface_levels,
    ).signed_margin_m


def joint_coverage_assessment(
    time: float,
    bombs: Sequence[BombGeometry],
    *,
    surface_angles: int,
    surface_levels: int,
    continuous_refinement: bool,
    refinement_starts: int = 0,
    refinement_maxfev: int = 0,
    adaptive_max_surface_angles: int | None = None,
    adaptive_max_surface_levels: int | None = None,
) -> CoverageAssessment:
    """Adaptively certify covered/uncovered rays using a Lipschitz grid bound."""
    active = _active_bombs(time, bombs)
    if not active:
        return CoverageAssessment(
            "uncovered-no-active-cloud",
            float("inf"),
            float("inf"),
            float("inf"),
            surface_angles,
            surface_levels,
            (),
        )
    missile = missile_position(time)
    max_angles = adaptive_max_surface_angles or surface_angles
    max_levels = adaptive_max_surface_levels or surface_levels
    current_angles = surface_angles
    current_levels = surface_levels
    trace: list[dict[str, float | int]] = []
    while True:
        points = _surface_points(current_angles, current_levels)
        margins = _coverage_margins(points, missile, active, time)
        sampled_maximum = float(np.max(margins))
        covering_radius = _surface_covering_radius(current_angles, current_levels)
        upper_bound = sampled_maximum + covering_radius
        trace.append(
            {
                "surface_angles": current_angles,
                "surface_levels": current_levels,
                "covering_radius_m": covering_radius,
                "sampled_lower_bound_m": sampled_maximum,
                "lipschitz_upper_bound_m": upper_bound,
            }
        )
        if not continuous_refinement or sampled_maximum > 0.0:
            status = "uncovered-certified" if sampled_maximum > 0.0 else "sampled-covered"
            return CoverageAssessment(
                status,
                sampled_maximum,
                sampled_maximum,
                upper_bound,
                current_angles,
                current_levels,
                tuple(trace),
            )
        if upper_bound <= 0.0:
            return CoverageAssessment(
                "covered-certified",
                upper_bound,
                sampled_maximum,
                upper_bound,
                current_angles,
                current_levels,
                tuple(trace),
            )
        if current_angles >= max_angles and current_levels >= max_levels:
            break
        current_angles = min(max_angles, current_angles * 2)
        current_levels = min(max_levels, 2 * (current_levels - 1) + 1)

    local_margin = _continuous_worst_margin(
        missile,
        active,
        time,
        points,
        margins,
        starts=refinement_starts,
        maxfev=refinement_maxfev,
    )
    status = "uncovered-certified-local-witness" if local_margin > 0.0 else "boundary-uncertain"
    return CoverageAssessment(
        status,
        local_margin,
        sampled_maximum,
        upper_bound,
        current_angles,
        current_levels,
        tuple(trace),
    )


def _merge_intervals(intervals: Sequence[OcclusionInterval]) -> tuple[OcclusionInterval, ...]:
    if not intervals:
        return ()
    ordered = sorted(intervals, key=lambda item: item.start)
    merged = [ordered[0]]
    for interval in ordered[1:]:
        previous = merged[-1]
        if interval.start <= previous.end + 1e-9:
            merged[-1] = OcclusionInterval(previous.start, max(previous.end, interval.end))
        else:
            merged.append(interval)
    return tuple(merged)


def _duration(intervals: Sequence[OcclusionInterval]) -> float:
    return sum(interval.duration for interval in intervals)


def _find_joint_intervals(
    margin: Callable[[float], float],
    bombs: Sequence[BombGeometry],
    *,
    scan_step: float,
    root_tolerance: float,
) -> tuple[OcclusionInterval, ...]:
    """Find union intervals, preserving cloud activation/deactivation breakpoints."""
    start = min(bomb.explosion_time_s for bomb in bombs)
    end = min(missile_hit_time(), max(bomb.explosion_time_s + CLOUD_LIFETIME for bomb in bombs))
    regular = np.linspace(start, end, int(np.ceil((end - start) / scan_step)) + 1)
    events = [
        event
        for bomb in bombs
        for event in (bomb.explosion_time_s, bomb.explosion_time_s + CLOUD_LIFETIME)
        if start <= event <= end
    ]
    event_probes = []
    probe = max(root_tolerance * 10.0, scan_step * 1e-5)
    for event in events:
        event_probes.extend((max(start, event - probe), min(end, event + probe)))
    times = np.unique(np.concatenate((regular, np.asarray(events), np.asarray(event_probes))))
    margins = np.array([margin(float(time)) for time in times])
    inside = margins <= 0.0
    intervals: list[OcclusionInterval] = []
    current = float(times[0]) if inside[0] else None
    for index in np.flatnonzero(inside[1:] != inside[:-1]):
        left, right = float(times[index]), float(times[index + 1])
        midpoint = 0.5 * (left + right)
        left_active = tuple(bomb.bomb for bomb in _active_bombs(left, bombs))
        middle_active = tuple(bomb.bomb for bomb in _active_bombs(midpoint, bombs))
        right_active = tuple(bomb.bomb for bomb in _active_bombs(right, bombs))
        if left_active != middle_active and middle_active == right_active:
            boundary = left
        elif left_active == middle_active and middle_active != right_active:
            boundary = right
        else:
            boundary = float(brentq(margin, left, right, xtol=root_tolerance, rtol=1e-14))
        if inside[index + 1]:
            current = boundary
        elif current is not None:
            intervals.append(OcclusionInterval(current, boundary))
            current = None
    if current is not None:
        intervals.append(OcclusionInterval(current, float(times[-1])))
    return _merge_intervals(intervals)


def _settings_margin(
    bombs: Sequence[BombGeometry], settings: dict[str, Any], *, refine: bool
) -> Callable[[float], float]:
    return lambda time: joint_coverage_margin(
        time,
        bombs,
        surface_angles=int(settings["surface_angles"]),
        surface_levels=int(settings["surface_levels"]),
        continuous_refinement=refine,
        refinement_starts=int(settings.get("continuous_refinement_starts", 0)),
        refinement_maxfev=int(settings.get("continuous_refinement_maxfev", 0)),
        adaptive_max_surface_angles=int(
            settings.get("adaptive_max_surface_angles", settings["surface_angles"])
        ),
        adaptive_max_surface_levels=int(
            settings.get("adaptive_max_surface_levels", settings["surface_levels"])
        ),
    )


def _sampled_metrics(
    bombs: Sequence[BombGeometry], margin: Callable[[float], float], time_step: float
) -> tuple[float, float]:
    start = min(bomb.explosion_time_s for bomb in bombs)
    end = min(missile_hit_time(), max(bomb.explosion_time_s + CLOUD_LIFETIME for bomb in bombs))
    anchored = np.arange(0.0, missile_hit_time() + 0.5 * time_step, time_step)
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
    margins = np.array([margin(float(time)) for time in times])
    inside = (margins <= 0.0).astype(float)
    return float(np.trapezoid(inside, times)), float(np.min(margins))


def _objective(
    values: np.ndarray, gravity: float, settings: dict[str, Any], *, roots: bool
) -> float:
    strategy = _strategy_from_array(values)
    violation = _feasibility_violation(strategy, gravity)
    if violation > 0.0:
        return 100.0 + 100.0 * violation
    bombs = _bomb_geometries(strategy, gravity)
    margin = _settings_margin(bombs, settings, refine=False)
    duration, minimum = _sampled_metrics(bombs, margin, float(settings["time_step_s"]))
    if duration <= 0.0:
        return max(minimum, 0.0) / 100.0
    if not roots:
        return -duration
    intervals = _find_joint_intervals(
        margin,
        bombs,
        scan_step=float(settings["time_step_s"]),
        root_tolerance=1e-8,
    )
    return -_duration(intervals)


def _bounds(config: dict[str, Any]) -> list[tuple[float, float]]:
    bounds = config["bounds"]
    fuse = tuple(bounds["fuse_delay_s"])
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


def _warm_vector(config: dict[str, Any]) -> np.ndarray:
    warm = config["q2_warm_start"]
    return np.array(
        [
            warm["theta_rad"],
            warm["speed_m_s"],
            warm["s0_s"],
            warm["s1_s"],
            warm["s2_s"],
            *warm["fuse_delays_s"],
        ],
        dtype=float,
    )


def _initial_population(config: dict[str, Any]) -> np.ndarray:
    bounds = _bounds(config)
    settings = config["global_search"]
    rng = np.random.default_rng(int(config["seed"]))
    count = int(settings["popsize"]) * len(bounds)
    lower = np.array([item[0] for item in bounds])
    upper = np.array([item[1] for item in bounds])
    population = rng.uniform(lower, upper, size=(count, len(bounds)))
    warm = _warm_vector(config)
    structured = []
    fuse_families = (
        (2.4968, 2.7, 0.5),
        (2.4968, 1.5, 0.5),
        (2.4968, 3.5, 4.5),
        (2.4968, 2.7, 2.0),
    )
    for speed in (70.0, 90.0, 110.0, 140.0):
        for theta_offset, fuses in zip((-0.03, -0.01, 0.01, 0.03), fuse_families, strict=True):
            structured.append(
                np.array([warm[0] + theta_offset, speed, 0.0, 0.0, 0.0, *fuses], dtype=float)
            )
    structured.insert(0, warm)
    for index, candidate in enumerate(structured[: len(population)]):
        population[index] = np.clip(candidate, lower, upper)
    return population


def _deduplicate(values: Sequence[np.ndarray]) -> list[np.ndarray]:
    unique: list[np.ndarray] = []
    for value in values:
        if not any(np.linalg.norm(value - existing) < 1e-7 for existing in unique):
            unique.append(np.asarray(value, dtype=float))
    return unique


def _individual_strict_intervals(
    strategy: Q3Strategy, gravity: float, config: dict[str, Any]
) -> tuple[tuple[OcclusionInterval, ...], ...]:
    q2_config = {
        "formal_evaluation": {
            "time_scan_step_s": config["formal_evaluation"]["time_scan_step_s"],
            "root_tolerance_s": config["formal_evaluation"]["root_tolerance_s"],
            "rim_samples": 2048,
        }
    }
    return tuple(
        q2_formal_evaluation(bomb, gravity, q2_config).strict_intervals
        for bomb in _bomb_strategies(strategy)
    )


def _formal_evaluation(strategy: Q3Strategy, config: dict[str, Any]) -> Q3Evaluation:
    gravity = float(config["gravity"])
    bombs = _bomb_geometries(strategy, gravity)
    settings = config["formal_evaluation"]
    margin = _settings_margin(bombs, settings, refine=True)
    joint = _find_joint_intervals(
        margin,
        bombs,
        scan_step=float(settings["time_scan_step_s"]),
        root_tolerance=float(settings["root_tolerance_s"]),
    )
    individual = _individual_strict_intervals(strategy, gravity, config)
    conservative = _merge_intervals([interval for group in individual for interval in group])
    deletion = []
    for removed in range(3):
        remaining = tuple(bomb for index, bomb in enumerate(bombs) if index != removed)
        remaining_margin = _settings_margin(remaining, settings, refine=False)
        intervals = _find_joint_intervals(
            remaining_margin,
            remaining,
            scan_step=float(settings["time_scan_step_s"]),
            root_tolerance=float(settings["root_tolerance_s"]),
        )
        deletion.append(min(_duration(intervals), _duration(joint)))
    first_duration = _duration(individual[0])
    first_two = bombs[:2]
    first_two_intervals = _find_joint_intervals(
        _settings_margin(first_two, settings, refine=True),
        first_two,
        scan_step=float(settings["time_scan_step_s"]),
        root_tolerance=float(settings["root_tolerance_s"]),
    )
    first_two_duration = _duration(first_two_intervals)
    incremental = (
        first_duration,
        first_two_duration - first_duration,
        _duration(joint) - first_two_duration,
    )
    return Q3Evaluation(
        strategy=strategy,
        bombs=bombs,
        joint_intervals=joint,
        individual_strict_intervals=individual,
        conservative_union_intervals=conservative,
        deletion_durations_s=tuple(deletion),
        ordered_incremental_durations_s=incremental,
    )


def optimize_question_3(
    config: dict[str, Any], *, stage_log_path: Path | None = None
) -> tuple[list[Q3Evaluation], dict[str, Any]]:
    """Run seeded eight-dimensional global search and multi-start local refinement."""
    gravity = float(config["gravity"])
    global_settings = config["global_search"]
    result = differential_evolution(
        lambda values: _objective(values, gravity, global_settings, roots=False),
        _bounds(config),
        seed=int(config["seed"]),
        init=_initial_population(config),
        maxiter=int(global_settings["maxiter"]),
        popsize=int(global_settings["popsize"]),
        tol=float(global_settings["tol"]),
        polish=bool(global_settings["polish"]),
        workers=1,
    )
    print(
        "Q3 global search complete; "
        f"evaluations={result.nfev}; sampled_duration={max(0.0, -float(result.fun)):.6f}s"
    )
    order = np.argsort(result.population_energies)
    persistent_incumbent = np.asarray(config["persistent_incumbent"]["vector"], dtype=float)
    candidates = [result.x, _warm_vector(config), persistent_incumbent]
    candidates.extend(np.asarray(result.population)[order[:10]])
    if stage_log_path is not None:
        stage_log_path.write_text(
            json.dumps(
                {
                    "stage": "global-search-complete",
                    "seed": int(config["seed"]),
                    "success": bool(result.success),
                    "message": str(result.message),
                    "evaluations": int(result.nfev),
                    "sampled_duration_s": max(0.0, -float(result.fun)),
                    "ranked_candidate_vectors": [
                        np.asarray(item, dtype=float).tolist() for item in candidates
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    local_results = []
    local_settings = config["local_refinement"]
    for start in _deduplicate(candidates)[: int(local_settings["starts"])]:
        local = minimize(
            lambda values: _objective(values, gravity, local_settings, roots=True),
            start,
            method="Nelder-Mead",
            bounds=_bounds(config),
            options={
                "maxfev": int(local_settings["maxfev"]),
                "xatol": float(local_settings["xatol"]),
                "fatol": float(local_settings["fatol"]),
            },
        )
        local_results.append(local)
        candidates.append(local.x)
        print(
            "Q3 local refinement complete; "
            f"start={len(local_results)}; evaluations={local.nfev}; "
            f"duration={max(0.0, -float(local.fun)):.6f}s"
        )
        if stage_log_path is not None:
            stage_log_path.write_text(
                json.dumps(
                    {
                        "stage": "local-refinement-in-progress",
                        "completed_starts": len(local_results),
                        "planned_starts": int(local_settings["starts"]),
                        "candidate_vectors": [
                            np.asarray(item, dtype=float).tolist() for item in candidates
                        ],
                        "local_results": [
                            {
                                "success": bool(item.success),
                                "message": str(item.message),
                                "evaluations": int(item.nfev),
                                "duration_s": max(0.0, -float(item.fun)),
                                "vector": np.asarray(item.x, dtype=float).tolist(),
                            }
                            for item in local_results
                        ],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    feasible = [
        item
        for item in _deduplicate(candidates)
        if _feasibility_violation(_strategy_from_array(item), gravity) <= 1e-12
    ]
    ranked = sorted(
        feasible,
        key=lambda values: _objective(values, gravity, local_settings, roots=True),
    )
    count = min(int(config["formal_evaluation"]["candidate_count"]), len(ranked))
    formal = []
    for index, item in enumerate(ranked[:count], 1):
        formal.append(_formal_evaluation(_strategy_from_array(item), config))
        print(f"Q3 formal candidate complete; candidate={index}/{count}")
    formal.sort(key=lambda item: item.joint_duration_s, reverse=True)
    if stage_log_path is not None:
        stage_log_path.write_text(
            json.dumps(
                {
                    "stage": "search-complete",
                    "seed": int(config["seed"]),
                    "global_success": bool(result.success),
                    "global_message": str(result.message),
                    "global_evaluations": int(result.nfev),
                    "completed_local_starts": len(local_results),
                    "formal_candidate_count": len(formal),
                    "formal_candidates": [
                        _strategy_summary(item, rank) for rank, item in enumerate(formal, 1)
                    ],
                    "best_formal_joint_duration_s": (
                        formal[0].joint_duration_s if formal else None
                    ),
                    "sampled_time_grid": "absolute task time t=0",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    log = {
        "seed": int(config["seed"]),
        "variable_order": [
            "theta_rad",
            "speed_m_s",
            "s0_s",
            "s1_s",
            "s2_s",
            "fuse1_s",
            "fuse2_s",
            "fuse3_s",
        ],
        "global_search": {
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "evaluations": int(result.nfev),
            "sampled_joint_duration_s": -float(result.fun),
            "settings": global_settings,
        },
        "candidate_initialization": {
            "mechanism": (
                "Q2 anchor plus structured partial-edge-complement seeds over four speeds, "
                "four heading offsets, and simultaneous/overlapping/staggered fuse families"
            ),
            "interpretation": (
                "later bombs target partial view-domain gaps; no single-cloud relay is assumed"
            ),
            "structured_seed_count": 17,
            "persistent_incumbent": config["persistent_incumbent"],
        },
        "sampled_time_grid": {
            "anchor": "absolute task time t=0",
            "included_points": "fixed-step grid plus window endpoints and cloud events",
            "invariance": (
                "inactive late clouds cannot shift the sampling phase of earlier coverage"
            ),
        },
        "corrected_full_rerun": True,
        "local_refinements": [
            {
                "success": bool(item.success),
                "message": str(item.message),
                "evaluations": int(item.nfev),
                "root_refined_duration_s": -float(item.fun) if item.fun <= 0 else None,
            }
            for item in local_results
        ],
        "formal_settings": config["formal_evaluation"],
        "status": "converged" if result.success else "budget-exhausted-with-feasible-candidates",
        "convergence_claimed": bool(result.success),
        "feasible_candidate_count": len(feasible),
    }
    return formal, log


def _strategy_summary(evaluation: Q3Evaluation, rank: int) -> dict[str, Any]:
    strategy = evaluation.strategy
    individual_total = sum(
        _duration(intervals) for intervals in evaluation.individual_strict_intervals
    )
    return {
        "rank": rank,
        "theta_rad": strategy.theta_rad,
        "theta_deg": float(np.degrees(strategy.theta_rad) % 360.0),
        "speed_m_s": strategy.speed_m_s,
        "release_times_s": json.dumps(strategy.release_times_s),
        "fuse_delays_s": json.dumps(strategy.fuse_delays_s),
        "explosion_times_s": json.dumps(strategy.explosion_times_s),
        "joint_duration_s": evaluation.joint_duration_s,
        "conservative_duration_s": evaluation.conservative_duration_s,
        "joint_synergy_s": evaluation.joint_duration_s - evaluation.conservative_duration_s,
        "sum_individual_strict_duration_s": individual_total,
        "individual_strict_overlap_s": individual_total - evaluation.conservative_duration_s,
    }


def _bomb_rows(evaluation: Q3Evaluation) -> list[dict[str, Any]]:
    rows = []
    individual_durations = [_duration(group) for group in evaluation.individual_strict_intervals]
    for bomb, individual, deleted, incremental in zip(
        evaluation.bombs,
        individual_durations,
        evaluation.deletion_durations_s,
        evaluation.ordered_incremental_durations_s,
        strict=True,
    ):
        rows.append(
            {
                "schema_status": "provisional-schema; official result1.xlsx unavailable",
                "drone": "FY1",
                "bomb": bomb.bomb,
                "theta_rad": evaluation.strategy.theta_rad,
                "theta_deg": float(np.degrees(evaluation.strategy.theta_rad) % 360.0),
                "speed_m_s": evaluation.strategy.speed_m_s,
                "release_time_s": bomb.release_time_s,
                "release_x_m": bomb.release_point[0],
                "release_y_m": bomb.release_point[1],
                "release_z_m": bomb.release_point[2],
                "fuse_delay_s": bomb.fuse_delay_s,
                "explosion_time_s": bomb.explosion_time_s,
                "explosion_x_m": bomb.explosion_point[0],
                "explosion_y_m": bomb.explosion_point[1],
                "explosion_z_m": bomb.explosion_point[2],
                "individual_strict_duration_s": individual,
                "joint_deletion_loss_s": evaluation.joint_duration_s - deleted,
                "ordered_incremental_contribution_s": incremental,
                "minimum_cloud_center_height_m": bomb.explosion_point[2]
                - CLOUD_DESCENT_SPEED * CLOUD_LIFETIME,
            }
        )
    return rows


def _independent_ray_intersections(
    points: np.ndarray, missile: np.ndarray, clouds: Sequence[np.ndarray]
) -> np.ndarray:
    """Independently test segment-sphere intersections through quadratic roots."""
    directions = points - missile
    lengths = np.linalg.norm(directions, axis=1)
    unit_directions = directions / lengths[:, None]
    covered = np.zeros(len(points), dtype=bool)
    for cloud in clouds:
        offset = missile - cloud
        b = 2.0 * (unit_directions @ offset)
        c = float(offset @ offset) - CLOUD_RADIUS**2
        discriminant = b**2 - 4.0 * c
        real = discriminant >= 0.0
        root = np.sqrt(np.maximum(discriminant, 0.0))
        lower = (-b - root) / 2.0
        upper = (-b + root) / 2.0
        covered |= real & (upper >= 0.0) & (lower <= lengths)
    return covered


def _validation_record(evaluation: Q3Evaluation, config: dict[str, Any]) -> dict[str, Any]:
    gravity = float(config["gravity"])
    validation = config["validation"]
    convergence = []
    convergence_settings = validation["convergence_settings"]
    for index, settings in enumerate(convergence_settings):
        use_refinement = index == len(convergence_settings) - 1
        merged = {
            **settings,
            "continuous_refinement_starts": 1,
            "continuous_refinement_maxfev": 50,
            "adaptive_max_surface_angles": (
                int(config["formal_evaluation"]["adaptive_max_surface_angles"])
                if use_refinement
                else int(settings["surface_angles"])
            ),
            "adaptive_max_surface_levels": (
                int(config["formal_evaluation"]["adaptive_max_surface_levels"])
                if use_refinement
                else int(settings["surface_levels"])
            ),
        }
        margin = _settings_margin(evaluation.bombs, merged, refine=use_refinement)
        intervals = _find_joint_intervals(
            margin,
            evaluation.bombs,
            scan_step=float(settings["time_scan_step_s"]),
            root_tolerance=float(config["formal_evaluation"]["root_tolerance_s"]),
        )
        convergence.append(
            {
                **settings,
                "continuous_surface_refinement": use_refinement,
                "joint_duration_s": _duration(intervals),
            }
        )

    boundary = []
    direct = []
    probe = float(validation["boundary_probe_s"])
    points = _surface_points(
        int(validation["direct_surface_angles"]), int(validation["direct_surface_levels"])
    )
    for interval in evaluation.joint_intervals:
        for location, time in (
            ("before-start", interval.start - probe),
            ("start", interval.start),
            ("midpoint", 0.5 * (interval.start + interval.end)),
            ("end", interval.end),
            ("after-end", interval.end + probe),
        ):
            assessment = joint_coverage_assessment(
                time,
                evaluation.bombs,
                surface_angles=int(config["formal_evaluation"]["surface_angles"]),
                surface_levels=int(config["formal_evaluation"]["surface_levels"]),
                continuous_refinement=True,
                refinement_starts=int(config["formal_evaluation"]["continuous_refinement_starts"]),
                refinement_maxfev=int(config["formal_evaluation"]["continuous_refinement_maxfev"]),
                adaptive_max_surface_angles=int(
                    config["formal_evaluation"]["adaptive_max_surface_angles"]
                ),
                adaptive_max_surface_levels=int(
                    config["formal_evaluation"]["adaptive_max_surface_levels"]
                ),
            )
            active = _active_bombs(time, evaluation.bombs)
            covered = _independent_ray_intersections(
                points,
                missile_position(time),
                [_cloud_center(time, bomb) for bomb in active],
            )
            boundary.append(
                {
                    "location": location,
                    "time_s": time,
                    "coverage_status": assessment.status,
                    "worst_margin_m": assessment.signed_margin_m,
                    "sampled_lower_bound_m": assessment.sampled_lower_bound_m,
                    "lipschitz_upper_bound_m": assessment.lipschitz_upper_bound_m,
                    "surface_angles": assessment.surface_angles,
                    "surface_levels": assessment.surface_levels,
                    "adaptive_trace": assessment.refinement_trace,
                }
            )
            direct.append(
                {
                    "location": location,
                    "time_s": time,
                    "sampled_rays": len(points),
                    "uncovered_rays": int(np.count_nonzero(~covered)),
                    "all_covered": bool(np.all(covered)),
                }
            )

    q2 = config["q2_warm_start"]
    q2_strategy = Q2Strategy(q2["theta_rad"], q2["speed_m_s"], 0.0, q2["fuse_delays_s"][0])
    q2_config = {
        "formal_evaluation": {
            "time_scan_step_s": config["formal_evaluation"]["time_scan_step_s"],
            "root_tolerance_s": config["formal_evaluation"]["root_tolerance_s"],
            "rim_samples": 2048,
        }
    }

    q2_reference = q2_formal_evaluation(q2_strategy, gravity, q2_config)
    single_bomb = _bomb_geometries(
        Q3Strategy(
            q2_strategy.theta_rad, q2_strategy.speed_m_s, 0, 0, 0, q2_strategy.fuse_delay_s, 19, 19
        ),
        gravity,
    )[0]
    single_settings = config["formal_evaluation"]
    single_intervals = _find_joint_intervals(
        _settings_margin((single_bomb,), single_settings, refine=True),
        (single_bomb,),
        scan_step=float(single_settings["time_scan_step_s"]),
        root_tolerance=float(single_settings["root_tolerance_s"]),
    )
    permutation_durations = []
    permutation_settings = {
        **validation["convergence_settings"][1],
        "continuous_refinement_starts": 0,
        "continuous_refinement_maxfev": 0,
    }
    for order in permutations(evaluation.bombs):
        intervals = _find_joint_intervals(
            _settings_margin(order, permutation_settings, refine=False),
            order,
            scan_step=float(permutation_settings["time_scan_step_s"]),
            root_tolerance=float(config["formal_evaluation"]["root_tolerance_s"]),
        )
        permutation_durations.append(_duration(intervals))

    complement_time = 4.768
    reference_cloud = np.array([17625.525268056128, 10.240444866222791, 1762.640057120904])
    complement_bombs = tuple(
        BombGeometry(
            bomb=index,
            release_time_s=complement_time,
            fuse_delay_s=0.0,
            explosion_time_s=complement_time,
            release_point=tuple(reference_cloud + np.array([0.0, sign * 9.65, 0.0])),
            explosion_point=tuple(reference_cloud + np.array([0.0, sign * 9.65, 0.0])),
        )
        for index, sign in enumerate((-1.0, 1.0), 1)
    )
    complement_settings = {
        "surface_angles": 360,
        "surface_levels": 41,
        "continuous_refinement_starts": 8,
        "continuous_refinement_maxfev": 100,
    }
    complement_individual = [
        joint_coverage_margin(
            complement_time,
            (bomb,),
            surface_angles=360,
            surface_levels=41,
            continuous_refinement=True,
            refinement_starts=8,
            refinement_maxfev=100,
        )
        for bomb in complement_bombs
    ]
    complement_joint = _settings_margin(complement_bombs, complement_settings, refine=True)(
        complement_time
    )
    return {
        "feasibility": {
            "violation": _feasibility_violation(evaluation.strategy, gravity),
            "release_gaps_s": np.diff(evaluation.strategy.release_times_s).tolist(),
            "explosion_times_s": list(evaluation.strategy.explosion_times_s),
            "explosion_heights_m": [bomb.explosion_point[2] for bomb in evaluation.bombs],
            "cloud_minimum_heights_m": [
                bomb.explosion_point[2] - CLOUD_DESCENT_SPEED * CLOUD_LIFETIME
                for bomb in evaluation.bombs
            ],
        },
        "surface_time_convergence": convergence,
        "boundary_checks": boundary,
        "independent_quadratic_ray_checks": direct,
        "label_permutation_durations_s": permutation_durations,
        "complementarity_construct": {
            "time_s": complement_time,
            "cloud_y_offsets_m": [-9.65, 9.65],
            "individual_worst_margins_m": complement_individual,
            "joint_worst_margin_m": complement_joint,
            "each_individual_incomplete": all(value > 0.0 for value in complement_individual),
            "joint_complete": complement_joint <= 0.0,
        },
        "single_cloud_q2_regression": {
            "q3_joint_duration_s": _duration(single_intervals),
            "q2_strict_duration_s": q2_reference.strict_duration_s,
            "absolute_difference_s": abs(
                _duration(single_intervals) - q2_reference.strict_duration_s
            ),
            "tolerance_s": float(validation["single_cloud_tolerance_s"]),
        },
        "joint_vs_conservative": {
            "joint_duration_s": evaluation.joint_duration_s,
            "conservative_duration_s": evaluation.conservative_duration_s,
            "joint_not_less": evaluation.joint_duration_s + 1e-9
            >= evaluation.conservative_duration_s,
        },
        "deletion": [
            {
                "bomb": index,
                "remaining_duration_s": duration,
                "marginal_loss_s": evaluation.joint_duration_s - duration,
            }
            for index, duration in enumerate(evaluation.deletion_durations_s, 1)
        ],
    }


def _conditional_third_bomb_search(
    baseline: Q3Evaluation, config: dict[str, Any]
) -> tuple[dict[str, Any], Q3Evaluation | None]:
    """Optimize only the third bomb conditional on the first two formal controls."""
    settings = config["conditional_third_bomb_search"]
    gravity = float(config["gravity"])
    baseline_vector = np.array(list(asdict(baseline.strategy).values()), dtype=float)

    def full_vector(values: np.ndarray) -> np.ndarray:
        candidate = baseline_vector.copy()
        candidate[4] = values[0]
        candidate[7] = values[1]
        return candidate

    search_settings = {
        "time_step_s": settings["time_step_s"],
        "surface_angles": settings["surface_angles"],
        "surface_levels": settings["surface_levels"],
    }
    result = differential_evolution(
        lambda values: _objective(full_vector(values), gravity, search_settings, roots=False),
        [tuple(config["bounds"]["s2_s"]), tuple(config["bounds"]["fuse_delay_s"])],
        seed=int(settings["seed"]),
        maxiter=int(settings["maxiter"]),
        popsize=int(settings["popsize"]),
        tol=float(settings["tol"]),
        polish=False,
        workers=1,
    )
    population = np.asarray(result.population)
    order = np.argsort(result.population_energies)
    candidate_values = [result.x]
    candidate_values.extend(population[order[: int(settings["root_candidate_count"])]])
    candidate_values.append(np.array([baseline.strategy.s2_s, baseline.strategy.fuse3_s]))
    root_settings = {
        "time_step_s": config["formal_evaluation"]["time_scan_step_s"],
        "surface_angles": config["formal_evaluation"]["surface_angles"],
        "surface_levels": config["formal_evaluation"]["surface_levels"],
    }
    root_ranked = []
    for values in _deduplicate(candidate_values):
        objective = _objective(full_vector(values), gravity, root_settings, roots=True)
        root_ranked.append(
            {
                "s2_s": float(values[0]),
                "fuse3_s": float(values[1]),
                "root_refined_duration_s": max(0.0, -objective),
                "vector": full_vector(values),
            }
        )
    root_ranked.sort(key=lambda item: item["root_refined_duration_s"], reverse=True)
    formal_candidate = _formal_evaluation(
        _strategy_from_array(np.asarray(root_ranked[0]["vector"])), config
    )
    improvement = formal_candidate.joint_duration_s - baseline.joint_duration_s
    threshold = float(settings["improvement_threshold_s"])
    record = {
        "scope": "theta, speed, and first two bombs fixed at the formal baseline",
        "variable_order": ["s2_s", "fuse3_s"],
        "seed": int(settings["seed"]),
        "search_success": bool(result.success),
        "search_message": str(result.message),
        "search_evaluations": int(result.nfev),
        "settings": settings,
        "root_refined_candidates": [
            {key: value for key, value in item.items() if key != "vector"} for item in root_ranked
        ],
        "formal_candidate": _strategy_summary(formal_candidate, 1),
        "baseline_joint_duration_s": baseline.joint_duration_s,
        "formal_improvement_s": improvement,
        "improvement_threshold_s": threshold,
        "material_improvement_found": improvement > threshold,
        "conclusion": (
            "conditional search found a material third-bomb contribution"
            if improvement > threshold
            else "conditional on the first two bombs, no material third-bomb contribution was found"
        ),
    }
    return record, formal_candidate if improvement > threshold else None


def run_question_3(project_root: Path, config: dict[str, Any]) -> list[Path]:
    """Optimize Question 3 and write tables, provisional workbook, and audit logs."""
    paths = prepare_output_paths(project_root / "outputs")
    stage_log_path = paths.logs / "q3_search_stage.json"
    evaluations, solver = optimize_question_3(config, stage_log_path=stage_log_path)
    if not evaluations:
        raise RuntimeError("Question 3 optimization produced no feasible formal candidate")
    best = evaluations[0]
    conditional_record, conditional_improvement = _conditional_third_bomb_search(best, config)
    if conditional_improvement is not None:
        evaluations = [conditional_improvement, *evaluations]
        evaluations.sort(key=lambda item: item.joint_duration_s, reverse=True)
        best = evaluations[0]
    near_count = min(int(config["near_optimal_count"]), len(evaluations))
    summaries = [
        _strategy_summary(item, rank) for rank, item in enumerate(evaluations[:near_count], 1)
    ]
    summary_path = save_table(
        pd.DataFrame(summaries[:1]), paths.tables / "q3_summary.csv", overwrite=True
    )
    candidates_path = save_table(
        pd.DataFrame(summaries), paths.tables / "q3_candidates.csv", overwrite=True
    )
    bombs = pd.DataFrame(_bomb_rows(best))
    bombs_path = save_table(bombs, paths.tables / "q3_bombs.csv", overwrite=True)
    interval_rows = []
    for criterion, groups in (
        ("joint-union-cover", (best.joint_intervals,)),
        ("conservative-single-cloud-union", (best.conservative_union_intervals,)),
        ("bomb-1-strict", (best.individual_strict_intervals[0],)),
        ("bomb-2-strict", (best.individual_strict_intervals[1],)),
        ("bomb-3-strict", (best.individual_strict_intervals[2],)),
    ):
        for group in groups:
            for number, interval in enumerate(group, 1):
                interval_rows.append(
                    {
                        "criterion": criterion,
                        "interval": number,
                        "start_s": interval.start,
                        "end_s": interval.end,
                        "duration_s": interval.duration,
                    }
                )
    intervals_path = save_table(
        pd.DataFrame(interval_rows), paths.tables / "q3_intervals.csv", overwrite=True
    )
    workbook_path = paths.tables / "result1.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        bombs.to_excel(writer, sheet_name="provisional_result1", index=False)
        pd.DataFrame(
            [
                {
                    "schema_status": "provisional",
                    "reason": "official result1.xlsx template is missing",
                    "mapping_action": (
                        "map these explicit fields when the official template is supplied"
                    ),
                }
            ]
        ).to_excel(writer, sheet_name="schema_note", index=False)

    optimization_path = paths.logs / "q3_optimization.json"
    optimization_path.write_text(
        json.dumps(
            {
                "model": "q3-joint-ray-cover-v1",
                "quantifiers": (
                    "for every target ray, at least one active cloud intersects before target"
                ),
                "best": {**_strategy_summary(best, 1), "bombs": _bomb_rows(best)},
                "near_optimal_candidates": summaries,
                "solver": solver,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    validation_path = paths.logs / "q3_validation.json"
    validation_path.write_text(
        json.dumps(_validation_record(best, config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    conditional_path = paths.logs / "q3_third_bomb_incremental.json"
    conditional_path.write_text(
        json.dumps(conditional_record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [
        summary_path,
        candidates_path,
        bombs_path,
        intervals_path,
        workbook_path,
        stage_log_path,
        optimization_path,
        validation_path,
        conditional_path,
    ]

"""Question 4 optimization for one smoke bomb from each of FY1, FY2, and FY3."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, minimize

from mmkit.artifacts import prepare_output_paths, save_table

from .q1 import CLOUD_DESCENT_SPEED, CLOUD_LIFETIME, OcclusionInterval, missile_position
from .q2 import missile_hit_time
from .q3 import (
    BombGeometry,
    Q3Strategy,
    _active_bombs,
    _cloud_center,
    _duration,
    _find_joint_intervals,
    _independent_ray_intersections,
    _merge_intervals,
    _sampled_metrics,
    _settings_margin,
    _surface_points,
    joint_coverage_assessment,
    joint_coverage_margin,
)
from .q3 import (
    _bomb_geometries as q3_bomb_geometries,
)

UAV_NAMES = ("FY1", "FY2", "FY3")


@dataclass(frozen=True, slots=True)
class UAVControl:
    """Flight and bomb controls for one UAV."""

    theta_rad: float
    speed_m_s: float
    release_time_s: float
    fuse_delay_s: float

    @property
    def explosion_time_s(self) -> float:
        return self.release_time_s + self.fuse_delay_s


@dataclass(frozen=True, slots=True)
class Q4Strategy:
    """Independent four-variable controls for FY1, FY2, and FY3."""

    controls: tuple[UAVControl, UAVControl, UAVControl]


@dataclass(frozen=True, slots=True)
class Q4Evaluation:
    """Formal joint evaluation for one Question 4 strategy."""

    strategy: Q4Strategy
    bombs: tuple[BombGeometry, BombGeometry, BombGeometry]
    joint_intervals: tuple[OcclusionInterval, ...]
    individual_intervals: tuple[tuple[OcclusionInterval, ...], ...]
    conservative_intervals: tuple[OcclusionInterval, ...]
    deletion_durations_s: tuple[float, float, float]

    @property
    def joint_duration_s(self) -> float:
        return _duration(self.joint_intervals)

    @property
    def conservative_duration_s(self) -> float:
        return _duration(self.conservative_intervals)


def _strategy_from_array(values: np.ndarray) -> Q4Strategy:
    array = np.asarray(values, dtype=float).reshape(3, 4)
    return Q4Strategy(tuple(UAVControl(*(float(value) for value in row)) for row in array))


def _strategy_vector(strategy: Q4Strategy) -> np.ndarray:
    return np.array(
        [value for control in strategy.controls for value in asdict(control).values()], dtype=float
    )


def _uav_initials(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(np.asarray(config["uavs"][name], dtype=float) for name in UAV_NAMES)


def _bomb_geometries(
    strategy: Q4Strategy, gravity: float, config: dict[str, Any]
) -> tuple[BombGeometry, BombGeometry, BombGeometry]:
    bombs = []
    for index, (control, initial) in enumerate(
        zip(strategy.controls, _uav_initials(config), strict=True), 1
    ):
        heading = np.array([np.cos(control.theta_rad), np.sin(control.theta_rad), 0.0])
        release = initial + control.speed_m_s * control.release_time_s * heading
        explosion = initial + control.speed_m_s * control.explosion_time_s * heading
        explosion = explosion + np.array([0.0, 0.0, -0.5 * gravity * control.fuse_delay_s**2])
        bombs.append(
            BombGeometry(
                bomb=index,
                release_time_s=control.release_time_s,
                fuse_delay_s=control.fuse_delay_s,
                explosion_time_s=control.explosion_time_s,
                release_point=tuple(float(value) for value in release),
                explosion_point=tuple(float(value) for value in explosion),
            )
        )
    return tuple(bombs)


def _fuse_maxima(config: dict[str, Any]) -> tuple[float, float, float]:
    gravity = float(config["gravity"])
    return tuple(float(np.sqrt(2.0 * initial[2] / gravity)) for initial in _uav_initials(config))


def _bounds(config: dict[str, Any]) -> list[tuple[float, float]]:
    common = config["bounds"]
    bounds = []
    for fuse_maximum in _fuse_maxima(config):
        bounds.extend(
            (
                tuple(common["theta_rad"]),
                tuple(common["speed_m_s"]),
                tuple(common["release_time_s"]),
                (0.0, fuse_maximum),
            )
        )
    return bounds


def _feasibility_violation(strategy: Q4Strategy, gravity: float, config: dict[str, Any]) -> float:
    violation = 0.0
    hit_time = missile_hit_time()
    for control, bomb, fuse_maximum, initial in zip(
        strategy.controls,
        _bomb_geometries(strategy, gravity, config),
        _fuse_maxima(config),
        _uav_initials(config),
        strict=True,
    ):
        violation += max(0.0, 70.0 - control.speed_m_s) / 70.0
        violation += max(0.0, control.speed_m_s - 140.0) / 70.0
        violation += max(0.0, -control.release_time_s) / hit_time
        violation += max(0.0, -control.fuse_delay_s) / fuse_maximum
        violation += max(0.0, control.fuse_delay_s - fuse_maximum) / fuse_maximum
        violation += max(0.0, control.explosion_time_s - hit_time) / hit_time
        violation += max(0.0, -bomb.explosion_point[2]) / float(initial[2])
    return violation


def _objective(
    values: np.ndarray,
    gravity: float,
    config: dict[str, Any],
    settings: dict[str, Any],
    *,
    roots: bool,
) -> float:
    strategy = _strategy_from_array(values)
    violation = _feasibility_violation(strategy, gravity, config)
    if violation > 0.0:
        return 100.0 + 100.0 * violation
    bombs = _bomb_geometries(strategy, gravity, config)
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


def _single_initial_population(
    initial: np.ndarray, fuse_maximum: float, config: dict[str, Any], seed: int
) -> np.ndarray:
    settings = config["single_search"]
    count = 4 * int(settings["popsize"])
    rng = np.random.default_rng(seed)
    lower = np.array([0.0, 70.0, 0.0, 0.0])
    upper = np.array([2.0 * np.pi, 140.0, 30.0, fuse_maximum])
    population = rng.uniform(lower, upper, size=(count, 4))
    toward_origin = float(np.arctan2(-initial[1], -initial[0]) % (2.0 * np.pi))
    structured = []
    for speed in (70.0, 100.0, 140.0):
        for release, fraction in ((0.0, 0.15), (0.0, 0.35), (2.0, 0.25)):
            structured.append(np.array([toward_origin, speed, release, fraction * fuse_maximum]))
    for index, candidate in enumerate(structured):
        population[index] = candidate
    return population


def _single_searches(config: dict[str, Any]) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    gravity = float(config["gravity"])
    settings = config["single_search"]
    best_vectors = []
    records = []
    for index, (name, initial, fuse_maximum) in enumerate(
        zip(UAV_NAMES, _uav_initials(config), _fuse_maxima(config), strict=True)
    ):

        def objective(values: np.ndarray, fixed_index: int = index) -> float:
            controls = [UAVControl(0.0, 70.0, 0.0, 0.0) for _ in UAV_NAMES]
            controls[fixed_index] = UAVControl(*(float(value) for value in values))
            strategy = Q4Strategy(tuple(controls))
            bomb = (_bomb_geometries(strategy, gravity, config)[fixed_index],)
            control = strategy.controls[fixed_index]
            if control.explosion_time_s > missile_hit_time():
                return 100.0 + control.explosion_time_s - missile_hit_time()
            margin = _settings_margin(bomb, settings, refine=False)
            duration, minimum = _sampled_metrics(bomb, margin, float(settings["time_step_s"]))
            return -duration if duration > 0.0 else max(minimum, 0.0) / 100.0

        result = differential_evolution(
            objective,
            [
                tuple(config["bounds"]["theta_rad"]),
                tuple(config["bounds"]["speed_m_s"]),
                tuple(config["bounds"]["release_time_s"]),
                (0.0, fuse_maximum),
            ],
            seed=int(config["seed"]) + index,
            init=_single_initial_population(
                initial, fuse_maximum, config, int(config["seed"]) + index
            ),
            maxiter=int(settings["maxiter"]),
            popsize=int(settings["popsize"]),
            tol=float(settings["tol"]),
            polish=False,
            workers=1,
        )
        best_vectors.append(np.asarray(result.x, dtype=float))
        records.append(
            {
                "uav": name,
                "success": bool(result.success),
                "message": str(result.message),
                "evaluations": int(result.nfev),
                "sampled_duration_s": max(0.0, -float(result.fun)),
                "vector": np.asarray(result.x, dtype=float).tolist(),
            }
        )
    return best_vectors, records


def _joint_initial_population(
    single_vectors: Sequence[np.ndarray], config: dict[str, Any]
) -> np.ndarray:
    bounds = _bounds(config)
    count = len(bounds) * int(config["joint_search"]["popsize"])
    rng = np.random.default_rng(int(config["seed"]) + 100)
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    population = rng.uniform(lower, upper, size=(count, len(bounds)))
    base = np.concatenate(single_vectors)
    population[0] = base
    variants = (
        np.zeros(12),
        np.array([0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 0], dtype=float),
        np.array([0, 0, 0, 0, 0.03, 10, 0, 0, -0.03, -5, 0, 0], dtype=float),
        np.array([0, 0, 0, 0.3, 0, 0, 0, -0.3, 0, 0, 0, 0.2], dtype=float),
    )
    for index, offset in enumerate(variants, 1):
        population[index] = np.clip(base + offset, lower, upper)
    for index in range(5, min(16, count)):
        population[index] = np.clip(base + rng.normal(0.0, 0.2, 12), lower, upper)
    return population


def _deduplicate(candidates: Sequence[np.ndarray]) -> list[np.ndarray]:
    unique = []
    for candidate in candidates:
        if not any(np.linalg.norm(candidate - existing) < 1e-7 for existing in unique):
            unique.append(np.asarray(candidate, dtype=float))
    return unique


def _formal_evaluation(strategy: Q4Strategy, config: dict[str, Any]) -> Q4Evaluation:
    gravity = float(config["gravity"])
    bombs = _bomb_geometries(strategy, gravity, config)
    settings = config["formal_evaluation"]
    joint = _find_joint_intervals(
        _settings_margin(bombs, settings, refine=True),
        bombs,
        scan_step=float(settings["time_scan_step_s"]),
        root_tolerance=float(settings["root_tolerance_s"]),
    )
    individual = []
    for bomb in bombs:
        intervals = _find_joint_intervals(
            _settings_margin((bomb,), settings, refine=True),
            (bomb,),
            scan_step=float(settings["time_scan_step_s"]),
            root_tolerance=float(settings["root_tolerance_s"]),
        )
        individual.append(intervals)
    conservative = _merge_intervals([interval for group in individual for interval in group])
    deletion = []
    for removed in range(3):
        remaining = tuple(bomb for index, bomb in enumerate(bombs) if index != removed)
        intervals = _find_joint_intervals(
            _settings_margin(remaining, settings, refine=False),
            remaining,
            scan_step=float(settings["time_scan_step_s"]),
            root_tolerance=float(settings["root_tolerance_s"]),
        )
        deletion.append(min(_duration(intervals), _duration(joint)))
    return Q4Evaluation(
        strategy=strategy,
        bombs=bombs,
        joint_intervals=joint,
        individual_intervals=tuple(individual),
        conservative_intervals=conservative,
        deletion_durations_s=tuple(deletion),
    )


def optimize_question_4(
    config: dict[str, Any], stage_log_path: Path
) -> tuple[list[Q4Evaluation], dict[str, Any]]:
    """Run single-UAV searches, joint global search, local refinement, and formal ranking."""
    gravity = float(config["gravity"])
    single_vectors, single_records = _single_searches(config)
    stage_log_path.write_text(
        json.dumps(
            {"stage": "single-search-complete", "single_candidates": single_records},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Q4 single-UAV searches complete", flush=True)
    settings = config["joint_search"]
    result = differential_evolution(
        lambda values: _objective(values, gravity, config, settings, roots=False),
        _bounds(config),
        seed=int(config["seed"]) + 10,
        init=_joint_initial_population(single_vectors, config),
        maxiter=int(settings["maxiter"]),
        popsize=int(settings["popsize"]),
        tol=float(settings["tol"]),
        polish=False,
        workers=1,
    )
    order = np.argsort(result.population_energies)
    candidates = [result.x, np.concatenate(single_vectors)]
    candidates.extend(np.asarray(result.population)[order[:10]])
    stage_log_path.write_text(
        json.dumps(
            {
                "stage": "joint-global-complete",
                "single_candidates": single_records,
                "success": bool(result.success),
                "message": str(result.message),
                "evaluations": int(result.nfev),
                "sampled_duration_s": max(0.0, -float(result.fun)),
                "candidate_vectors": [np.asarray(item).tolist() for item in candidates],
                "sampled_time_grid": "absolute task time t=0",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Q4 joint global search complete", flush=True)
    local_results = []
    local_settings = config["local_refinement"]
    for start in _deduplicate(candidates)[: int(local_settings["starts"])]:
        local = minimize(
            lambda values: _objective(values, gravity, config, local_settings, roots=True),
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
        stage_log_path.write_text(
            json.dumps(
                {
                    "stage": "local-refinement-in-progress",
                    "completed_starts": len(local_results),
                    "planned_starts": int(local_settings["starts"]),
                    "local_results": [
                        {
                            "success": bool(item.success),
                            "message": str(item.message),
                            "evaluations": int(item.nfev),
                            "duration_s": max(0.0, -float(item.fun)),
                            "vector": np.asarray(item.x).tolist(),
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
        print(f"Q4 local refinement {len(local_results)} complete", flush=True)
    feasible = [
        item
        for item in _deduplicate(candidates)
        if _feasibility_violation(_strategy_from_array(item), gravity, config) <= 1e-12
    ]
    ranked = sorted(
        feasible,
        key=lambda values: _objective(values, gravity, config, local_settings, roots=True),
    )
    count = min(int(config["formal_evaluation"]["candidate_count"]), len(ranked))
    incumbent = np.concatenate(single_vectors)
    formal_vectors = _deduplicate([*ranked[: max(0, count - 1)], incumbent])[:count]
    formal = []
    for index, candidate in enumerate(formal_vectors, 1):
        formal.append(_formal_evaluation(_strategy_from_array(candidate), config))
        print(f"Q4 formal candidate {index}/{count} complete", flush=True)
    formal.sort(key=lambda item: item.joint_duration_s, reverse=True)
    stage_log_path.write_text(
        json.dumps(
            {
                "stage": "search-complete",
                "single_candidates": single_records,
                "global_success": bool(result.success),
                "global_message": str(result.message),
                "global_evaluations": int(result.nfev),
                "completed_local_starts": len(local_results),
                "formal_candidate_count": len(formal),
                "formal_durations_s": [item.joint_duration_s for item in formal],
                "single_concat_incumbent_retained": True,
                "sampled_time_grid": "absolute task time t=0",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    solver = {
        "seed": int(config["seed"]),
        "single_searches": single_records,
        "joint_global": {
            "success": bool(result.success),
            "message": str(result.message),
            "evaluations": int(result.nfev),
            "sampled_duration_s": max(0.0, -float(result.fun)),
            "settings": settings,
        },
        "local_refinements": [
            {
                "success": bool(item.success),
                "message": str(item.message),
                "evaluations": int(item.nfev),
                "duration_s": max(0.0, -float(item.fun)),
            }
            for item in local_results
        ],
        "candidate_initialization": (
            "per-UAV single candidates plus relay, overlap, heading/speed, "
            "and edge-complement variants"
        ),
        "sampled_time_grid": "absolute task time t=0 plus cloud events",
        "status": "converged" if result.success else "budget-exhausted-with-feasible-candidates",
        "convergence_claimed": bool(result.success),
        "formal_settings": config["formal_evaluation"],
    }
    return formal, solver


def _summary_row(evaluation: Q4Evaluation, rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "joint_duration_s": evaluation.joint_duration_s,
        "conservative_duration_s": evaluation.conservative_duration_s,
        "joint_synergy_s": evaluation.joint_duration_s - evaluation.conservative_duration_s,
        "individual_durations_s": json.dumps(
            [_duration(intervals) for intervals in evaluation.individual_intervals]
        ),
    }


def _uav_rows(evaluation: Q4Evaluation) -> list[dict[str, Any]]:
    rows = []
    for name, control, bomb, individual, deletion in zip(
        UAV_NAMES,
        evaluation.strategy.controls,
        evaluation.bombs,
        evaluation.individual_intervals,
        evaluation.deletion_durations_s,
        strict=True,
    ):
        rows.append(
            {
                "schema_status": "provisional-schema; official result2.xlsx unavailable",
                "uav": name,
                "theta_rad": control.theta_rad,
                "theta_deg": float(np.degrees(control.theta_rad) % 360.0),
                "speed_m_s": control.speed_m_s,
                "release_time_s": control.release_time_s,
                "release_x_m": bomb.release_point[0],
                "release_y_m": bomb.release_point[1],
                "release_z_m": bomb.release_point[2],
                "fuse_delay_s": control.fuse_delay_s,
                "explosion_time_s": control.explosion_time_s,
                "explosion_x_m": bomb.explosion_point[0],
                "explosion_y_m": bomb.explosion_point[1],
                "explosion_z_m": bomb.explosion_point[2],
                "individual_strict_duration_s": _duration(individual),
                "joint_deletion_loss_s": evaluation.joint_duration_s - deletion,
                "minimum_cloud_center_height_m": bomb.explosion_point[2]
                - CLOUD_DESCENT_SPEED * CLOUD_LIFETIME,
            }
        )
    return rows


def _validation_record(evaluation: Q4Evaluation, config: dict[str, Any]) -> dict[str, Any]:
    validation = config["validation"]
    hit_time = missile_hit_time()
    convergence = []
    convergence_settings = validation["convergence_settings"]
    for index, settings in enumerate(convergence_settings):
        refine = index == len(convergence_settings) - 1
        merged = {
            **settings,
            "continuous_refinement_starts": 1,
            "continuous_refinement_maxfev": 50,
            "adaptive_max_surface_angles": (
                config["formal_evaluation"]["adaptive_max_surface_angles"]
                if refine
                else settings["surface_angles"]
            ),
            "adaptive_max_surface_levels": (
                config["formal_evaluation"]["adaptive_max_surface_levels"]
                if refine
                else settings["surface_levels"]
            ),
        }
        intervals = _find_joint_intervals(
            _settings_margin(evaluation.bombs, merged, refine=refine),
            evaluation.bombs,
            scan_step=float(settings["time_scan_step_s"]),
            root_tolerance=float(config["formal_evaluation"]["root_tolerance_s"]),
        )
        convergence.append(
            {**settings, "continuous_refinement": refine, "joint_duration_s": _duration(intervals)}
        )
    boundary = []
    direct = []
    points = _surface_points(
        int(validation["direct_surface_angles"]), int(validation["direct_surface_levels"])
    )
    probe = float(validation["boundary_probe_s"])
    formal = config["formal_evaluation"]
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
                surface_angles=int(formal["surface_angles"]),
                surface_levels=int(formal["surface_levels"]),
                continuous_refinement=True,
                refinement_starts=int(formal["continuous_refinement_starts"]),
                refinement_maxfev=int(formal["continuous_refinement_maxfev"]),
                adaptive_max_surface_angles=int(formal["adaptive_max_surface_angles"]),
                adaptive_max_surface_levels=int(formal["adaptive_max_surface_levels"]),
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
                    "status": assessment.status,
                    "margin_m": assessment.signed_margin_m,
                    "lower_bound_m": assessment.sampled_lower_bound_m,
                    "upper_bound_m": assessment.lipschitz_upper_bound_m,
                    "surface_angles": assessment.surface_angles,
                    "surface_levels": assessment.surface_levels,
                    "trace": assessment.refinement_trace,
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
    permutation_durations = []
    medium = validation["convergence_settings"][1]
    for order in permutations(evaluation.bombs):
        intervals = _find_joint_intervals(
            _settings_margin(order, medium, refine=False),
            order,
            scan_step=float(medium["time_scan_step_s"]),
            root_tolerance=float(formal["root_tolerance_s"]),
        )
        permutation_durations.append(_duration(intervals))
    q3_vector = np.array(
        [
            3.1047922065753095,
            75.53740845287045,
            0.00008659727834377369,
            0.0001070796609000733,
            0.0000817521129715,
            2.7648584468708908,
            2.58163903013058,
            9.334353521045797,
        ]
    )
    q3_bombs = q3_bomb_geometries(Q3Strategy(*q3_vector), float(config["gravity"]))
    q3_intervals = _find_joint_intervals(
        _settings_margin(q3_bombs, formal, refine=True),
        q3_bombs,
        scan_step=float(formal["time_scan_step_s"]),
        root_tolerance=float(formal["root_tolerance_s"]),
    )
    complement_time = 4.768
    center = np.array([17625.525268056128, 10.240444866222791, 1762.640057120904])
    complement_bombs = tuple(
        BombGeometry(
            index,
            complement_time,
            0.0,
            complement_time,
            tuple(center + np.array([0.0, sign * 9.65, 0.0])),
            tuple(center + np.array([0.0, sign * 9.65, 0.0])),
        )
        for index, sign in enumerate((-1.0, 1.0), 1)
    )
    individual_complement = [
        joint_coverage_margin(
            complement_time,
            (bomb,),
            surface_angles=360,
            surface_levels=41,
            continuous_refinement=True,
            refinement_starts=1,
            refinement_maxfev=50,
        )
        for bomb in complement_bombs
    ]
    joint_complement = joint_coverage_margin(
        complement_time,
        complement_bombs,
        surface_angles=360,
        surface_levels=41,
        continuous_refinement=True,
        refinement_starts=1,
        refinement_maxfev=50,
    )
    return {
        "feasibility": {
            "violation": _feasibility_violation(
                evaluation.strategy, float(config["gravity"]), config
            ),
            "speeds_m_s": [control.speed_m_s for control in evaluation.strategy.controls],
            "release_times_s": [control.release_time_s for control in evaluation.strategy.controls],
            "fuse_delays_s": [control.fuse_delay_s for control in evaluation.strategy.controls],
            "explosion_times_s": [
                control.explosion_time_s for control in evaluation.strategy.controls
            ],
            "explosion_heights_m": [bomb.explosion_point[2] for bomb in evaluation.bombs],
            "cloud_minimum_heights_m": [
                bomb.explosion_point[2] - CLOUD_DESCENT_SPEED * CLOUD_LIFETIME
                for bomb in evaluation.bombs
            ],
            "all_speeds_within_bounds": all(
                70.0 <= control.speed_m_s <= 140.0 for control in evaluation.strategy.controls
            ),
            "all_release_and_fuse_times_nonnegative": all(
                control.release_time_s >= 0.0 and control.fuse_delay_s >= 0.0
                for control in evaluation.strategy.controls
            ),
            "all_explosions_before_missile_hit": all(
                control.explosion_time_s <= hit_time for control in evaluation.strategy.controls
            ),
            "all_explosions_before_ground": all(
                bomb.explosion_point[2] >= 0.0 for bomb in evaluation.bombs
            ),
            "cloud_center_below_ground_triggered": any(
                bomb.explosion_point[2] - CLOUD_DESCENT_SPEED * CLOUD_LIFETIME < 0.0
                for bomb in evaluation.bombs
            ),
        },
        "surface_time_convergence": convergence,
        "boundary_checks": boundary,
        "independent_unit_direction_ray_checks": direct,
        "label_permutation_durations_s": permutation_durations,
        "q3_geometry_regression": {
            "computed_duration_s": _duration(q3_intervals),
            "expected_duration_s": float(validation["q3_regression_expected_s"]),
            "absolute_difference_s": abs(
                _duration(q3_intervals) - float(validation["q3_regression_expected_s"])
            ),
            "tolerance_s": float(validation["q3_regression_tolerance_s"]),
        },
        "complementarity_construct": {
            "individual_margins_m": individual_complement,
            "joint_margin_m": joint_complement,
            "each_individual_incomplete": all(value > 0.0 for value in individual_complement),
            "joint_complete": joint_complement <= 0.0,
        },
        "joint_vs_conservative": {
            "joint_duration_s": evaluation.joint_duration_s,
            "conservative_duration_s": evaluation.conservative_duration_s,
            "joint_not_less": evaluation.joint_duration_s + 1e-9
            >= evaluation.conservative_duration_s,
            "joint_not_less_than_each_individual": evaluation.joint_duration_s + 1e-9
            >= max(_duration(intervals) for intervals in evaluation.individual_intervals),
        },
        "deletion": [
            {
                "uav": name,
                "remaining_duration_s": duration,
                "marginal_loss_s": evaluation.joint_duration_s - duration,
            }
            for name, duration in zip(UAV_NAMES, evaluation.deletion_durations_s, strict=True)
        ],
    }


def run_question_4(project_root: Path, config: dict[str, Any]) -> list[Path]:
    """Run Question 4 and write strategy, workbook, solver, and validation artifacts."""
    paths = prepare_output_paths(project_root / "outputs")
    stage_path = paths.logs / "q4_search_stage.json"
    evaluations, solver = optimize_question_4(config, stage_path)
    if not evaluations:
        raise RuntimeError("Question 4 optimization produced no formal candidate")
    best = evaluations[0]
    near_count = min(int(config["near_optimal_count"]), len(evaluations))
    summaries = [_summary_row(item, rank) for rank, item in enumerate(evaluations[:near_count], 1)]
    summary_path = save_table(
        pd.DataFrame(summaries[:1]), paths.tables / "q4_summary.csv", overwrite=True
    )
    candidates_path = save_table(
        pd.DataFrame(summaries), paths.tables / "q4_candidates.csv", overwrite=True
    )
    rows = pd.DataFrame(_uav_rows(best))
    uavs_path = save_table(rows, paths.tables / "q4_uavs.csv", overwrite=True)
    interval_rows = []
    groups = [
        ("joint-union-cover", best.joint_intervals),
        ("conservative-single-cloud-union", best.conservative_intervals),
        *(
            (f"{name}-strict", intervals)
            for name, intervals in zip(UAV_NAMES, best.individual_intervals, strict=True)
        ),
    ]
    for criterion, intervals in groups:
        for index, interval in enumerate(intervals, 1):
            interval_rows.append(
                {
                    "criterion": criterion,
                    "interval": index,
                    "start_s": interval.start,
                    "end_s": interval.end,
                    "duration_s": interval.duration,
                }
            )
    intervals_path = save_table(
        pd.DataFrame(interval_rows), paths.tables / "q4_intervals.csv", overwrite=True
    )
    workbook_path = paths.tables / "result2.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        rows.to_excel(writer, sheet_name="provisional_result2", index=False)
        pd.DataFrame(
            [
                {
                    "schema_status": "provisional",
                    "reason": "official result2.xlsx template is missing",
                    "mapping_action": "map explicit fields when the official template is supplied",
                }
            ]
        ).to_excel(writer, sheet_name="schema_note", index=False)
    optimization_path = paths.logs / "q4_optimization.json"
    optimization_path.write_text(
        json.dumps(
            {
                "model": "q4-three-uav-joint-ray-cover-v1",
                "best": {**_summary_row(best, 1), "uavs": _uav_rows(best)},
                "near_optimal_candidates": summaries,
                "solver": solver,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    validation_path = paths.logs / "q4_validation.json"
    validation_path.write_text(
        json.dumps(_validation_record(best, config), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return [
        summary_path,
        candidates_path,
        uavs_path,
        intervals_path,
        workbook_path,
        stage_path,
        optimization_path,
        validation_path,
    ]

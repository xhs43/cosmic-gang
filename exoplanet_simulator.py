"""Configurable synthetic light-curve generator and production-pipeline adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from transit_detection_pipeline import OUTPUT_COLUMNS, detect_transit
from xgb_validation_layer import FEATURE_COLUMNS, extract_physics_features, validate_candidate


@dataclass
class SimulationConfig:
    scenario: str = "planet"
    target: str = "SIMULATED"
    session_id: str = "sim_01"
    seed: int = 1103
    sample_count: int = 1200
    baseline_days: float = 12.0
    period_days: float = 1.5
    duration_days: float = 0.10
    depth: float = 0.015
    secondary_depth: float = 0.0
    cadence_jitter: float = 0.20
    noise_sigma: float = 0.0025
    noise_variation: float = 0.35
    red_noise_sigma: float = 0.001
    red_noise_tau_days: float = 0.05
    gap_fraction: float = 0.0
    outlier_fraction: float = 0.0
    outlier_sigma: float = 8.0

    def __post_init__(self):
        if self.scenario == "eclipsing_binary":
            self.depth = 0.15
            self.secondary_depth = 0.05
            self.duration_days = 0.20
        elif self.scenario == "grazing_binary":
            self.depth = 0.10
            self.duration_days = 0.15


def _time_grid(config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    intervals = config.baseline_days / config.sample_count
    steps = intervals * rng.lognormal(0.0, config.cadence_jitter, config.sample_count)
    times = np.cumsum(steps)
    return config.baseline_days * (times - times.min()) / (times.max() - times.min())


def _phase(times: np.ndarray, period: float) -> np.ndarray:
    return ((times + 0.5 * period) % period) - 0.5 * period


def _red_noise(times: np.ndarray, config: SimulationConfig, rng: np.random.Generator) -> np.ndarray:
    red = np.empty(len(times), dtype=float)
    red[0] = rng.normal(0.0, config.red_noise_sigma)
    for index in range(1, len(times)):
        phi = np.exp(-(times[index] - times[index - 1]) / config.red_noise_tau_days)
        red[index] = phi * red[index - 1] + rng.normal(0.0, config.red_noise_sigma * np.sqrt(1.0 - phi * phi))
    return red


def generate_light_curve(config: SimulationConfig = SimulationConfig()) -> pd.DataFrame:
    if config.scenario not in {"planet", "eclipsing_binary", "grazing_binary", "stellar_flare", "instrumental_noise"}:
        raise ValueError(f"Unsupported scenario: {config.scenario}")
    rng = np.random.default_rng(config.seed)
    times = _time_grid(config, rng)
    phase = _phase(times, config.period_days)
    flux = np.ones(config.sample_count, dtype=float)

    if config.scenario in {"planet", "eclipsing_binary", "grazing_binary"}:
        width = config.duration_days / (2.0 if config.scenario != "grazing_binary" else 1.4)
        primary = np.abs(phase) <= width
        flux[primary] -= config.depth
        if config.scenario == "eclipsing_binary" and config.secondary_depth > 0:
            secondary_phase = np.abs(np.abs(phase) - config.period_days / 2.0) <= width
            flux[secondary_phase] -= config.secondary_depth
    elif config.scenario == "stellar_flare":
        flare_center = config.baseline_days * 0.55
        flare_width = max(config.duration_days, config.baseline_days * 0.01)
        flux += config.depth * np.exp(-0.5 * ((times - flare_center) / flare_width) ** 2)

    heteroscedastic_sigma = config.noise_sigma * rng.lognormal(0.0, config.noise_variation, config.sample_count)
    flux += _red_noise(times, config, rng)
    flux += rng.normal(0.0, heteroscedastic_sigma)

    if config.outlier_fraction > 0:
        outlier_mask = rng.random(config.sample_count) < config.outlier_fraction
        flux[outlier_mask] += rng.normal(0.0, config.outlier_sigma * heteroscedastic_sigma[outlier_mask])
    if config.gap_fraction > 0:
        keep = rng.random(config.sample_count) >= config.gap_fraction
        times, flux, heteroscedastic_sigma = times[keep], flux[keep], heteroscedastic_sigma[keep]

    return pd.DataFrame({
        "target": config.target,
        "time_bjd": times,
        "raw_flux": flux,
        "flux_err": heteroscedastic_sigma,
        "session_id": config.session_id,
        "is_real_data": False,
    })[OUTPUT_COLUMNS]


def run_simulation(config: SimulationConfig = SimulationConfig(), model_path: str | Path = "xgb_transit_vetter.json") -> dict[str, Any]:
    curve = generate_light_curve(config)
    time = curve["time_bjd"].to_numpy(float)
    flux = curve["raw_flux"].to_numpy(float)
    flux_err = curve["flux_err"].to_numpy(float)
    detection = detect_transit(time, flux, flux_err, expected_period=config.period_days, expected_duration_hours=config.duration_days * 24.0)
    result = {"target": config.target, "detection": detection, "validation": {}}
    features = extract_physics_features(curve, result)
    vetter = validate_candidate(features, model_path=model_path)
    return {
        "config": asdict(config),
        "curve": curve,
        "detection": detection,
        "features": {column: features[column] for column in FEATURE_COLUMNS},
        "vetter": vetter,
    }


if __name__ == "__main__":
    smoke = run_simulation()
    print({"rows": len(smoke["curve"]), "detected": smoke["detection"].get("transit_detected"), "vetter_flags": smoke["vetter"]["flags"]})

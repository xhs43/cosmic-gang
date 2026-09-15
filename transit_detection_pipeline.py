"""Transit detection and astrophysical validation for cleaned light curves.

The module is importable by the Streamlit dashboard and executable from the
repository root.  Measurements are kept separate from NASA benchmark values:
benchmarks are priors for the search window and validation, never substitutes
for a failed detection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

from astropy.timeseries import BoxLeastSquares
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


LOGGER = logging.getLogger("transit_detection_pipeline")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

NASA_BENCHMARKS: dict[str, dict[str, float]] = {
    "CoRoT-2": {"period": 1.74299, "depth": 0.0315, "rp_rs": 0.1667, "duration_hours": 2.25},
    "Qatar-1": {"period": 1.42003, "depth": 0.0210, "rp_rs": 0.1449, "duration_hours": 1.60},
    "TrES-3": {"period": 1.30619, "depth": 0.0290, "rp_rs": 0.1655, "duration_hours": 1.55},
    "TrES-5": {"period": 1.48225, "depth": 0.0215, "rp_rs": 0.1432, "duration_hours": 1.80},
    "HAT-P-10": {"period": 3.07634, "depth": 0.0145, "rp_rs": 0.1190, "duration_hours": 2.70},
    "WASP-2": {"period": 2.15222, "depth": 0.0160, "rp_rs": 0.1280, "duration_hours": 1.80},
    "WASP-10": {"period": 3.09276, "depth": 0.0240, "rp_rs": 0.1550, "duration_hours": 2.30},
    "TRES-1": {"period": 3.03007, "depth": 0.0210, "rp_rs": 0.1370, "duration_hours": 2.50},
}
OUTPUT_COLUMNS = ["target", "time_bjd", "raw_flux", "flux_err", "session_id", "is_real_data"]


def load_photometry(path: str | Path = "CLEANED_PHOTOMETRY.csv") -> pd.DataFrame:
    """Load and validate the Part 1 light-curve contract."""
    frame = pd.read_csv(path)
    missing = set(OUTPUT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required photometry columns: {sorted(missing)}")
    for column in ("time_bjd", "raw_flux", "flux_err"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["target"] = frame["target"].astype(str).str.strip()
    frame["is_real_data"] = frame["is_real_data"].map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes"}
    )
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_bjd", "raw_flux"])
    return frame.sort_values(["target", "time_bjd"]).reset_index(drop=True)


def _valid_savgol_window(window_len: int, sample_count: int, poly_order: int) -> int | None:
    window = int(window_len)
    if window % 2 == 0:
        window += 1
    window = min(window, sample_count if sample_count % 2 else sample_count - 1)
    minimum = max(5, poly_order + 2 + ((poly_order + 2) % 2 == 0))
    return window if window >= minimum and window > poly_order else None


def detrend_airmass(time: Iterable[float], flux: Iterable[float], window_len: int, poly_order: int = 2) -> np.ndarray:
    """Remove smooth terrestrial trends with a Savitzky-Golay baseline.

    ``window_len`` is measured in samples and callers should choose it longer
    than three transit durations.  The function gracefully returns a median-
    normalized series when a session is too short for the requested window.
    """
    del time  # The filter operates on cadence-ordered samples; time is retained in the public API.
    values = np.asarray(flux, dtype=float)
    normalized = values / np.nanmedian(values) if np.isfinite(values).any() else np.ones_like(values)
    valid = np.isfinite(normalized)
    if valid.sum() < 5:
        return np.where(valid, normalized, 1.0)
    filled = np.interp(np.arange(len(normalized)), np.flatnonzero(valid), normalized[valid])
    window = _valid_savgol_window(window_len, len(filled), poly_order)
    if window is None:
        baseline = np.nanmedian(filled)
    else:
        baseline = savgol_filter(filled, window_length=window, polyorder=min(poly_order, window - 1), mode="interp")
    baseline = np.where(np.isfinite(baseline) & (np.abs(baseline) > 1e-12), baseline, 1.0)
    corrected = filled / baseline
    return np.where(valid, corrected, np.nan)


def _clip_out_of_transit(time: np.ndarray, flux: np.ndarray, expected_period: float, expected_duration_days: float, sigma: float = 3.5) -> np.ndarray:
    phase = ((time - time[0] + 0.5 * expected_period) % expected_period) - 0.5 * expected_period
    out_of_transit = np.abs(phase) > expected_duration_days / 2.0
    if out_of_transit.sum() < 2:
        return np.isfinite(flux)
    residuals = flux - np.nanmedian(flux[out_of_transit])
    mad = 1.4826 * np.nanmedian(np.abs(residuals[out_of_transit] - np.nanmedian(residuals[out_of_transit])))
    scale = mad if np.isfinite(mad) and mad > 0 else np.nanstd(residuals[out_of_transit])
    if not np.isfinite(scale) or scale <= 0:
        return np.isfinite(flux)
    return np.isfinite(flux) & (~out_of_transit | (np.abs(residuals) <= sigma * scale))


def _period_grid(expected_period: float, count: int = 240) -> np.ndarray:
    return np.linspace(expected_period * 0.9, expected_period * 1.1, count)


def search_bls(time: Iterable[float], flux: Iterable[float], flux_err: Iterable[float] | None, expected_period: float, expected_duration_hours: float) -> dict[str, Any]:
    """Search +/-10% around a benchmark period and return the best BLS box."""
    time_array = np.asarray(time, dtype=float)
    flux_array = np.asarray(flux, dtype=float)
    error_array = None if flux_err is None else np.asarray(flux_err, dtype=float)
    valid = np.isfinite(time_array) & np.isfinite(flux_array)
    if error_array is not None:
        valid &= ~np.isfinite(error_array) | (error_array > 0)
    time_array, flux_array = time_array[valid], flux_array[valid]
    if error_array is not None:
        error_array = error_array[valid]
    if len(time_array) < 8 or np.ptp(time_array) <= 0:
        return {"detected": False, "reason": "insufficient_samples_or_baseline"}
    flux_array = flux_array / np.nanmedian(flux_array)
    duration_days = expected_duration_hours / 24.0
    durations = np.array([0.75, 1.0, 1.25, 1.5]) * duration_days
    durations = durations[durations < 0.25 * expected_period]
    if len(durations) == 0:
        durations = np.array([min(duration_days, 0.25 * expected_period)])
    try:
        model = BoxLeastSquares(time_array, flux_array, dy=error_array)
        result = model.power(_period_grid(expected_period), durations, objective="snr", oversample=10)
        index = int(np.nanargmax(result.power))
    except Exception as exc:
        LOGGER.warning("BLS search failed: %s", exc)
        return {"detected": False, "reason": f"bls_error: {exc}"}
    depth = float(result.depth[index])
    return {
        "detected": bool(np.isfinite(result.power[index]) and depth > 0),
        "period": float(result.period[index]),
        "t0": float(result.transit_time[index]),
        "depth": depth,
        "duration_days": float(result.duration[index]),
        "duration_hours": float(result.duration[index] * 24.0),
        "rp_rs": float(np.sqrt(max(depth, 0.0))),
        "power": float(result.power[index]),
        "period_grid_min": float(expected_period * 0.9),
        "period_grid_max": float(expected_period * 1.1),
    }


def phase_fold_and_bin(time: Iterable[float], flux: Iterable[float], t0: float, period: float, bins: int = 60) -> dict[str, list[float]]:
    """Phase-fold using the documented convention and return finite bins."""
    time_array = np.asarray(time, dtype=float)
    flux_array = np.asarray(flux, dtype=float)
    phase = ((time_array - t0 + 0.5 * period) % period) - 0.5 * period
    edges = np.linspace(-0.5 * period, 0.5 * period, int(np.clip(bins, 50, 80)) + 1)
    centers: list[float] = []
    means: list[float] = []
    errors: list[float] = []
    counts: list[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = np.isfinite(phase) & np.isfinite(flux_array) & (phase >= left) & (phase < right)
        if not selected.any():
            continue
        values = flux_array[selected]
        centers.append(float((left + right) / 2.0))
        means.append(float(np.mean(values)))
        errors.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)
        counts.append(int(len(values)))
    return {"phase": centers, "flux": means, "flux_err": errors, "count": counts}


def _relative_error(measured: float | None, truth: float) -> float | None:
    return None if measured is None or not np.isfinite(measured) else float(abs(measured - truth) / truth * 100.0)


def _transit_mask(time: np.ndarray, t0: float, period: float, duration_days: float) -> np.ndarray:
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    return np.abs(phase) <= duration_days / 2.0


def validate_detection(time: Iterable[float], flux: Iterable[float], detection: dict[str, Any], truth: dict[str, float]) -> dict[str, Any]:
    """Compare BLS geometry with benchmarks and run EB parity/secondary checks."""
    if not detection.get("detected"):
        return {"relative_errors_percent": {}, "false_positive_checks": {"status": "not_evaluated"}}
    time_array = np.asarray(time, dtype=float)
    flux_array = np.asarray(flux, dtype=float)
    in_transit = _transit_mask(time_array, detection["t0"], detection["period"], detection["duration_days"])
    baseline = np.nanmedian(flux_array[~in_transit]) if (~in_transit).any() else 1.0
    cycles = np.floor((time_array - detection["t0"]) / detection["period"]).astype(int)
    odd_depth = 1.0 - np.nanmedian(flux_array[in_transit & (cycles % 2 != 0)]) / baseline if (in_transit & (cycles % 2 != 0)).any() else np.nan
    even_depth = 1.0 - np.nanmedian(flux_array[in_transit & (cycles % 2 == 0)]) / baseline if (in_transit & (cycles % 2 == 0)).any() else np.nan
    phase = ((time_array - detection["t0"] + 0.5 * detection["period"]) % detection["period"]) - 0.5 * detection["period"]
    secondary = np.abs(np.abs(phase) - 0.5 * detection["period"]) <= detection["duration_days"] / 2.0
    secondary_depth = 1.0 - np.nanmedian(flux_array[secondary]) / baseline if secondary.any() else np.nan
    out_of_transit = flux_array[~in_transit]
    noise = np.nanstd(out_of_transit) if len(out_of_transit) else np.nan
    parity_delta = abs(odd_depth - even_depth) if np.isfinite(odd_depth) and np.isfinite(even_depth) else np.nan
    return {
        "relative_errors_percent": {
            "period": _relative_error(detection.get("period"), truth["period"]),
            "depth": _relative_error(detection.get("depth"), truth["depth"]),
            "rp_rs": _relative_error(detection.get("rp_rs"), truth["rp_rs"]),
            "duration_hours": _relative_error(detection.get("duration_hours"), truth["duration_hours"]),
        },
        "false_positive_checks": {
            "odd_depth": float(odd_depth) if np.isfinite(odd_depth) else None,
            "even_depth": float(even_depth) if np.isfinite(even_depth) else None,
            "odd_even_depth_difference": float(parity_delta) if np.isfinite(parity_delta) else None,
            "odd_even_consistent": bool(parity_delta <= max(0.25 * truth["depth"], 3.0 * noise)) if np.isfinite(parity_delta) and np.isfinite(noise) else None,
            "secondary_depth": float(secondary_depth) if np.isfinite(secondary_depth) else None,
            "secondary_eclipse_detected": bool(secondary_depth > max(3.0 * noise, 0.005)) if np.isfinite(secondary_depth) and np.isfinite(noise) else None,
            "status": "evaluated",
        },
    }


def analyze_target(target: str, light_curve: pd.DataFrame, bins: int = 60) -> dict[str, Any]:
    """Run detrending, BLS, phase binning, and validation for one target."""
    if target not in NASA_BENCHMARKS:
        raise KeyError(f"Unknown benchmark target: {target}")
    truth = NASA_BENCHMARKS[target]
    curve = light_curve[light_curve["target"] == target].copy()
    time = curve["time_bjd"].to_numpy(dtype=float)
    raw_flux = curve["raw_flux"].to_numpy(dtype=float)
    flux_err = curve["flux_err"].to_numpy(dtype=float) if "flux_err" in curve else None
    cadence = float(np.nanmedian(np.diff(np.sort(time)))) if len(time) > 2 else np.nan
    required_window = int(np.ceil((3.0 * truth["duration_hours"] / 24.0) / cadence)) if np.isfinite(cadence) and cadence > 0 else 9
    detrend_window = max(9, required_window | 1)
    detrended = detrend_airmass(time, raw_flux, detrend_window, poly_order=2)
    clip_mask = _clip_out_of_transit(time, detrended, truth["period"], truth["duration_hours"] / 24.0)
    detection = search_bls(time[clip_mask], detrended[clip_mask], flux_err[clip_mask] if flux_err is not None else None, truth["period"], truth["duration_hours"])
    validation = validate_detection(time[clip_mask], detrended[clip_mask], detection, truth)
    phase_bins = phase_fold_and_bin(time[clip_mask], detrended[clip_mask], detection["t0"], detection["period"], bins=bins) if detection.get("detected") else {}
    real_flags = curve["is_real_data"].astype(bool) if "is_real_data" in curve else pd.Series(False, index=curve.index)
    return {
        "target": target,
        "sample_count": int(len(time)),
        "usable_sample_count": int(clip_mask.sum()),
        "real_data_count": int(real_flags.sum()),
        "detrending": {"method": "savgol_filter", "window_samples": int(detrend_window), "poly_order": 2, "out_of_transit_clip_sigma": 3.5},
        "nasa_truth": truth,
        "detection": detection,
        "validation": validation,
        "phase_binned": phase_bins,
    }


def run_pipeline(input_path: str | Path = "CLEANED_PHOTOMETRY.csv", output_path: str | Path = "TRANSIT_RESULTS.json", targets: Iterable[str] | None = None) -> dict[str, Any]:
    """Run all requested targets and export JSON results."""
    data = load_photometry(input_path)
    selected = list(targets or NASA_BENCHMARKS)
    results: dict[str, Any] = {}
    for target in selected:
        try:
            results[target] = analyze_target(target, data)
        except Exception as exc:
            LOGGER.exception("Target %s failed safely", target)
            results[target] = {"target": target, "error": str(exc), "nasa_truth": NASA_BENCHMARKS.get(target)}
    payload = {"pipeline": "transit_detection_pipeline", "targets": results}
    Path(output_path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print("\n=== Transit Detection Executive Summary ===")
    for target, result in results.items():
        detection = result.get("detection", {})
        status = "detected" if detection.get("detected") else detection.get("reason", result.get("error", "not detected"))
        print(f"{target}: {status}; samples={result.get('sample_count', 0)}; real={result.get('real_data_count', 0)}")
    print(f"Results written to: {Path(output_path).resolve()}")
    return payload


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:
        LOGGER.exception("Transit detection stopped safely: %s", exc)
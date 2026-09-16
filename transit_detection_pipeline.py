"""Robust transit discovery for BJD/HJD photometric time series."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter

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


def _first_column(columns: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    normalized = {str(column).strip().lower().replace("-", "_"): str(column) for column in columns}
    return next((normalized[name] for name in candidates if name in normalized), None)


def load_light_curve(path: str | Path, target: str | None = None) -> pd.DataFrame:
    """Load generic CSV columns into ``time_bjd``, ``raw_flux``, and ``flux_err``."""
    source = pd.read_csv(path)
    time_column = _first_column(source.columns, ("time", "time_bjd", "bjd", "hjd", "jd"))
    flux_column = _first_column(source.columns, ("flux", "raw_flux", "relative_flux", "normalized_flux"))
    error_column = _first_column(source.columns, ("flux_err", "flux_error", "error", "flux_uncertainty"))
    target_column = _first_column(source.columns, ("target", "target_name", "object"))
    if time_column is None or flux_column is None:
        raise ValueError("Input CSV must contain a time column (BJD/HJD) and a flux column")
    result = pd.DataFrame({
        "time_bjd": pd.to_numeric(source[time_column], errors="coerce"),
        "raw_flux": pd.to_numeric(source[flux_column], errors="coerce"),
        "flux_err": pd.to_numeric(source[error_column], errors="coerce") if error_column else np.nan,
    })
    if target_column:
        result["target"] = source[target_column].astype(str).str.strip()
        if target is not None:
            result = result[result["target"] == target]
    result = result.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_bjd", "raw_flux"])
    result = result.sort_values("time_bjd").drop_duplicates("time_bjd").reset_index(drop=True)
    if result.empty:
        raise ValueError("Input light curve contains no finite time/flux rows")
    return result


def load_photometry(path: str | Path = "CLEANED_PHOTOMETRY.csv") -> pd.DataFrame:
    """Load and validate the project's photometry contract."""
    source = pd.read_csv(path)
    missing = set(OUTPUT_COLUMNS) - set(source.columns)
    if missing:
        raise ValueError(f"Missing required photometry columns: {sorted(missing)}")
    for column in ("time_bjd", "raw_flux", "flux_err"):
        source[column] = pd.to_numeric(source[column], errors="coerce")
    source["target"] = source["target"].astype(str).str.strip()
    source["is_real_data"] = source["is_real_data"].map(lambda value: str(value).lower() in {"true", "1", "yes"})
    return source.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_bjd", "raw_flux"]).sort_values(["target", "time_bjd"]).reset_index(drop=True)


def _valid_savgol_window(window_len: int, sample_count: int, poly_order: int) -> int | None:
    if sample_count < 5:
        return None
    window = min(max(int(window_len), 5), sample_count if sample_count % 2 else sample_count - 1)
    if window % 2 == 0:
        window -= 1
    return window if window > poly_order and window >= poly_order + 2 else None


def _robust_scale(values: np.ndarray) -> float:
    median = float(np.nanmedian(values))
    scale = 1.4826 * float(np.nanmedian(np.abs(values - median)))
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.nanstd(values))
    return scale if np.isfinite(scale) and scale > 0 else 1.0


def iterative_sigma_clip(time: Iterable[float], flux: Iterable[float], sigma: float = 4.0, max_iterations: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively reject isolated robust-sigma outliers while preserving coherent temporal structures."""
    time_array = np.asarray(time, dtype=float)
    flux_array = np.asarray(flux, dtype=float)
    mask = np.isfinite(time_array) & np.isfinite(flux_array)
    
    for _ in range(max_iterations):
        if mask.sum() < 3:
            break
            
        # Interpolate masked values to avoid gaps affecting the median filter
        filled = np.copy(flux_array)
        if (~mask).any():
            filled[~mask] = np.interp(np.flatnonzero(~mask), np.flatnonzero(mask), flux_array[mask])
            
        # Use a rolling median filter of size 5 to establish a local baseline.
        # This protects coherent astrophysical structures spanning >= 3 points (like transits) 
        # from being clipped, while exposing isolated 1-2 point instrumental spikes.
        local_baseline = median_filter(filled, size=5)
        
        residuals = flux_array - local_baseline
        valid_res = residuals[mask]
        
        scale = _robust_scale(valid_res)
        if scale <= 0:
            break
            
        # Symmetrical sigma clip on the residuals
        upper_bound = sigma * scale
        lower_bound = -sigma * scale
        
        updated = mask & (residuals <= upper_bound) & (residuals >= lower_bound)
        if updated.sum() == mask.sum() or updated.sum() < 3:
            break
        mask = updated
        
    return time_array[mask], flux_array[mask]


def detrend_airmass(time: Iterable[float], flux: Iterable[float], window_len: int = 31, poly_order: int = 2) -> np.ndarray:
    """Median-normalize and remove smooth baseline variability while preserving transit signals."""
    del time
    values = np.asarray(flux, dtype=float)
    valid = np.isfinite(values)
    if not valid.any():
        return np.ones_like(values)
    
    normalized = values / np.nanmedian(values[valid])
    filled = np.interp(np.arange(len(normalized)), np.flatnonzero(valid), normalized[valid])
    window = _valid_savgol_window(window_len, len(filled), poly_order)
    
    if window is None:
        baseline = np.full_like(filled, np.nanmedian(filled))
    else:
        # 1. Initial robust baseline estimate using median filter
        # Median filter is much more robust against deep transit dips than savgol_filter
        initial_baseline = median_filter(filled, size=window)
        
        # 2. Detect downward transit candidates
        residuals = filled - initial_baseline
        robust_std = _robust_scale(residuals)
        
        # Mask points that are significantly below the baseline (e.g., > 2.5 sigma)
        non_transit_mask = residuals > -2.5 * robust_std
        
        # Fallback if too much is masked (e.g., > 30% of data)
        if non_transit_mask.sum() < 0.7 * len(filled):
            non_transit_mask = np.ones(len(filled), dtype=bool)
            
        # 3. Fit baseline using non-transit points
        safe_filled = np.copy(filled)
        if (~non_transit_mask).any():
            safe_filled[~non_transit_mask] = np.interp(
                np.flatnonzero(~non_transit_mask), 
                np.flatnonzero(non_transit_mask), 
                filled[non_transit_mask]
            )
            
        baseline = savgol_filter(safe_filled, window, min(poly_order, window - 1), mode="interp")
        
    baseline = np.where(np.isfinite(baseline) & (np.abs(baseline) > 1e-12), baseline, 1.0)
    return np.where(valid, filled / baseline, np.nan)


def _dynamic_period_grid(time: np.ndarray, min_period: float | None, max_period: float | None, expected_period: float | None) -> np.ndarray:
    baseline = float(np.ptp(time))
    cadence = float(np.nanmedian(np.diff(np.sort(time)))) if len(time) > 1 else 0.0
    lower = float(min_period if min_period is not None else max(0.1, 2.0 * cadence))
    upper = float(max_period if max_period is not None else min(baseline, 100.0))
    upper = min(upper, baseline)
    if lower >= upper or baseline <= 0:
        return np.array([])
    count = int(np.clip(5 * np.sqrt(max(len(time), 1)) * (upper - lower) / max(lower, 1e-6), 120, 5000))
    return np.linspace(lower, upper, count)


def _duration_grid(periods: np.ndarray, expected_duration_hours: float | None) -> np.ndarray:
    if len(periods) == 0:
        return np.array([])
    values = float(np.nanmedian(periods)) * np.array([0.005, 0.01, 0.02, 0.04, 0.08])
    return np.unique(np.clip(values, 0.01, max(float(np.nanmin(periods)) * 0.25, 0.02)))


def _transit_snr(time: np.ndarray, flux: np.ndarray, t0: float, period: float, duration: float, depth: float) -> tuple[float, float, float]:
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    in_transit = np.abs(phase) <= duration / 2.0
    
    n_in = int(np.sum(in_transit))
    n_out = int(np.sum(~in_transit))
    
    if n_in < 3 or n_out < 3:
        return 0.0, 0.0, float(np.nanmedian(flux)) if len(flux) > 0 else 1.0
        
    out_flux = flux[~in_transit]
    in_flux = flux[in_transit]
    
    baseline = float(np.nanmedian(out_flux))
    
    # Robust noise estimation from out-of-transit points using MAD
    noise = 1.4826 * float(np.nanmedian(np.abs(out_flux - baseline)))
    
    if noise <= 0:
        noise = float(np.nanstd(out_flux))
        
    if noise <= 0:
        return 0.0, 0.0, baseline
        
    # Standard error of the difference between means (in-transit vs out-of-transit)
    standard_error = noise * np.sqrt(1.0 / n_in + 1.0 / n_out)
    
    snr = float(depth / standard_error) if standard_error > 0 else 0.0
    
    return snr, float(standard_error), baseline


def search_bls(time: Iterable[float], flux: Iterable[float], flux_err: Iterable[float] | None = None, expected_period: float | None = None, expected_duration_hours: float | None = None, min_period: float | None = None, max_period: float | None = None, sigma_clip: float = 4.0) -> dict[str, Any]:
    """Run dynamic BLS and return the best period, epoch, duration, depth, and SNR."""
    time_array, flux_array = iterative_sigma_clip(time, flux, sigma=sigma_clip)
    baseline = float(np.ptp(time_array)) if len(time_array) > 0 else 0.0
    
    if len(time_array) < 8 or baseline < 0.1:
        return {"transit_detected": False, "detected": False, "reason": "INSUFFICIENT_DATA", "clean_time": time_array.tolist(), "clean_flux": flux_array.tolist()}
        
    flux_array = flux_array / float(np.nanmedian(flux_array))
    periods = _dynamic_period_grid(time_array, min_period, max_period, expected_period)
    
    if len(periods) == 0:
        return {"transit_detected": False, "detected": False, "reason": "INSUFFICIENT_DATA", "clean_time": time_array.tolist(), "clean_flux": flux_array.tolist()}
        
    durations = _duration_grid(periods, expected_duration_hours)
    
    if len(durations) == 0:
        return {"transit_detected": False, "detected": False, "reason": "INSUFFICIENT_DATA", "clean_time": time_array.tolist(), "clean_flux": flux_array.tolist()}
        
    try:
        result = BoxLeastSquares(time_array, flux_array).power(periods, durations, objective="snr", oversample=10)
        index = int(np.nanargmax(result.power))
    except Exception as exc:
        return {"transit_detected": False, "detected": False, "reason": f"bls_error: {exc}", "clean_time": time_array.tolist(), "clean_flux": flux_array.tolist()}
        
    period = float(result.period[index])
    t0 = float(result.transit_time[index])
    duration = float(result.duration[index])
    depth = max(float(result.depth[index]), 0.0)
    
    # Check for minimum number of transit events
    in_transit = np.abs(((time_array - t0 + 0.5 * period) % period) - 0.5 * period) <= duration / 2.0
    transit_epochs = np.round((time_array[in_transit] - t0) / period)
    num_events = len(np.unique(transit_epochs))
    
    if num_events < 2:
        return {"transit_detected": False, "detected": False, "reason": "INSUFFICIENT_DATA", "clean_time": time_array.tolist(), "clean_flux": flux_array.tolist()}
        
    snr, standard_error, baseline_median_flux = _transit_snr(time_array, flux_array, t0, period, duration, depth)
    detected = bool(np.isfinite(result.power[index]) and depth > 0 and snr >= 3.0)
    
    return {"period": period, "t0": t0, "duration_days": duration, "duration_hours": duration * 24.0, "depth": depth, "rp_rs": float(np.sqrt(depth)), "snr": snr, "standard_error_transit": standard_error, "baseline_median_flux": baseline_median_flux, "power": float(result.power[index]), "period_grid_min": float(periods.min()), "period_grid_max": float(periods.max()), "transit_detected": detected, "detected": detected, "num_events": num_events, "clean_time": time_array.tolist(), "clean_flux": flux_array.tolist()}


def _transit_mask(time: np.ndarray, t0: float, period: float, duration_days: float) -> np.ndarray:
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    return np.abs(phase) <= duration_days / 2.0


def assess_preprocessing_mask(time: Iterable[float], clean_time: Iterable[float], detection: dict[str, Any]) -> dict[str, Any]:
    """Report whether clipping removed samples from the detected candidate region."""
    time_array = np.asarray(time, dtype=float)
    clean_time_array = np.asarray(clean_time, dtype=float)
    period = detection.get("period")
    t0 = detection.get("t0")
    duration = detection.get("duration_days")
    if period is None or t0 is None or duration is None or not np.isfinite([period, t0, duration]).all():
        return {"status": "UNAVAILABLE", "candidate_points_before_clipping": None, "candidate_points_removed": None, "candidate_removal_fraction": None}

    candidate_mask = _transit_mask(time_array, float(t0), float(period), float(duration))
    kept_mask = np.isin(time_array, clean_time_array)
    candidate_count = int(candidate_mask.sum())
    removed_count = int((candidate_mask & ~kept_mask).sum())
    removal_fraction = removed_count / candidate_count if candidate_count else None
    return {"status": "SUSPECT" if removed_count > 0 else "CLEAR", "candidate_points_before_clipping": candidate_count, "candidate_points_removed": removed_count, "candidate_removal_fraction": removal_fraction}


def phase_fold_and_bin(time: Iterable[float], flux: Iterable[float], t0: float, period: float, bins: int = 60) -> dict[str, list[float]]:
    time_array, flux_array = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    phase = ((time_array - t0 + 0.5 * period) % period) - 0.5 * period
    edges = np.linspace(-0.5 * period, 0.5 * period, int(np.clip(bins, 50, 80)) + 1)
    centers: list[float] = []
    means: list[float] = []
    errors: list[float] = []
    counts: list[int] = []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = np.isfinite(phase) & np.isfinite(flux_array) & (phase >= left) & (phase < right)
        if selected.any():
            values = flux_array[selected]
            centers.append(float((left + right) / 2.0))
            means.append(float(np.mean(values)))
            errors.append(float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0)
            counts.append(int(len(values)))
    return {"phase": centers, "flux": means, "flux_err": errors, "count": counts}


def validate_detection(time: Iterable[float], flux: Iterable[float], detection: dict[str, Any], truth: dict[str, float]) -> dict[str, Any]:
    """Compute benchmark errors and odd/even or secondary-eclipse checks."""
    if not detection.get("transit_detected", detection.get("detected", False)):
        return {"relative_errors_percent": {}, "false_positive_checks": {"status": "not_evaluated"}}
    time_array, flux_array = np.asarray(time, dtype=float), np.asarray(flux, dtype=float)
    in_transit = _transit_mask(time_array, detection["t0"], detection["period"], detection["duration_days"])
    baseline = np.nanmedian(flux_array[~in_transit]) if (~in_transit).any() else 1.0
    cycles = np.floor((time_array - detection["t0"]) / detection["period"]).astype(int)
    odd, even = in_transit & (cycles % 2 != 0), in_transit & (cycles % 2 == 0)
    odd_depth = 1.0 - np.nanmedian(flux_array[odd]) / baseline if odd.any() else np.nan
    even_depth = 1.0 - np.nanmedian(flux_array[even]) / baseline if even.any() else np.nan
    phase = ((time_array - detection["t0"] + 0.5 * detection["period"]) % detection["period"]) - 0.5 * detection["period"]
    secondary = np.abs(np.abs(phase) - 0.5 * detection["period"]) <= detection["duration_days"] / 2.0
    secondary_depth = 1.0 - np.nanmedian(flux_array[secondary]) / baseline if secondary.any() else np.nan
    noise = np.nanstd(flux_array[~in_transit]) if (~in_transit).any() else np.nan
    parity = abs(odd_depth - even_depth) if np.isfinite(odd_depth) and np.isfinite(even_depth) else np.nan
    return {"relative_errors_percent": {"period": abs(detection["period"] - truth["period"]) / truth["period"] * 100.0, "depth": abs(detection["depth"] - truth["depth"]) / truth["depth"] * 100.0, "rp_rs": abs(detection["rp_rs"] - truth["rp_rs"]) / truth["rp_rs"] * 100.0, "duration_hours": abs(detection["duration_hours"] - truth["duration_hours"]) / truth["duration_hours"] * 100.0}, "false_positive_checks": {"odd_depth": float(odd_depth) if np.isfinite(odd_depth) else None, "even_depth": float(even_depth) if np.isfinite(even_depth) else None, "odd_even_depth_difference": float(parity) if np.isfinite(parity) else None, "odd_even_consistent": bool(parity <= max(0.25 * truth["depth"], 3.0 * noise)) if np.isfinite(parity) and np.isfinite(noise) else None, "secondary_depth": float(secondary_depth) if np.isfinite(secondary_depth) else None, "secondary_eclipse_detected": bool(secondary_depth > max(3.0 * noise, 0.005)) if np.isfinite(secondary_depth) and np.isfinite(noise) else None, "status": "evaluated"}}


def detect_transit(time: Iterable[float], flux: Iterable[float], flux_err: Iterable[float] | None = None, *, expected_period: float | None = None, expected_duration_hours: float | None = None, min_period: float | None = None, max_period: float | None = None, detrend_window: int = 31, poly_order: int = 2, sigma_clip: float = 4.0) -> dict[str, Any]:
    """Return the stable discovery contract requested by the pipeline."""
    clean_time, clean_flux = iterative_sigma_clip(time, flux, sigma=sigma_clip)
    detrended = detrend_airmass(clean_time, clean_flux, detrend_window, poly_order)
    detection = search_bls(clean_time, detrended, flux_err, expected_period, expected_duration_hours, min_period, max_period, sigma_clip)
    result = {"period": detection.get("period"), "t0": detection.get("t0"), "duration": detection.get("duration_days"), "duration_days": detection.get("duration_days"), "depth": detection.get("depth"), "snr": detection.get("snr", 0.0), "transit_detected": bool(detection.get("transit_detected", False)), "clean_time": detection.get("clean_time", clean_time.tolist()), "clean_flux": detection.get("clean_flux", np.nan_to_num(detrended, nan=1.0).tolist()), "standard_error_transit": detection.get("standard_error_transit"), "baseline_median_flux": detection.get("baseline_median_flux"), "rp_rs": detection.get("rp_rs"), "power": detection.get("power"), "reason": detection.get("reason")}
    result["preprocessing_mask"] = assess_preprocessing_mask(time, result["clean_time"], result)
    return result


def analyze_target(target: str, light_curve: pd.DataFrame, bins: int = 60) -> dict[str, Any]:
    """Run the rigorous pipeline for one benchmark target."""
    if target not in NASA_BENCHMARKS:
        raise KeyError(f"Unknown benchmark target: {target}")
    truth = NASA_BENCHMARKS[target]
    curve = light_curve[light_curve["target"] == target].copy()
    time = curve["time_bjd"].to_numpy(dtype=float)
    raw_flux = curve["raw_flux"].to_numpy(dtype=float)
    flux_err = curve["flux_err"].to_numpy(dtype=float) if "flux_err" in curve else None
    cadence = float(np.nanmedian(np.diff(np.sort(time)))) if len(time) > 2 else np.nan
    required_window = int(np.ceil((3.0 * truth["duration_hours"] / 24.0) / cadence)) if np.isfinite(cadence) and cadence > 0 else 31
    detrend_window = max(9, required_window | 1)
    detection = detect_transit(time, raw_flux, flux_err, expected_period=truth["period"], expected_duration_hours=truth["duration_hours"], detrend_window=detrend_window)
    validation = validate_detection(detection["clean_time"], detection["clean_flux"], {**detection, "duration_days": detection["duration"]}, truth)
    real_flags = curve["is_real_data"].astype(bool) if "is_real_data" in curve else pd.Series(False, index=curve.index)
    return {"target": target, "sample_count": int(len(time)), "usable_sample_count": len(detection["clean_time"]), "real_data_count": int(real_flags.sum()), "detrending": {"method": "savgol_filter", "window_samples": int(detrend_window), "poly_order": 2, "sigma_clip": 4.0}, "nasa_truth": truth, "detection": {**detection, "detected": detection["transit_detected"], "duration_days": detection["duration"]}, "validation": validation, "phase_binned": phase_fold_and_bin(detection["clean_time"], detection["clean_flux"], detection["t0"], detection["period"], bins=bins) if detection["transit_detected"] else {}}


def run_pipeline(input_path: str | Path = "CLEANED_PHOTOMETRY.csv", output_path: str | Path = "TRANSIT_RESULTS.json", targets: Iterable[str] | None = None) -> dict[str, Any]:
    """Run all benchmark targets and export JSON results."""
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
    for target, result in results.items():
        detection = result.get("detection", {})
        status = "detected" if detection.get("transit_detected") else detection.get("reason", result.get("error", "not detected"))
        print(f"{target}: {status}; samples={result.get('sample_count', 0)}; real={result.get('real_data_count', 0)}")
    print(f"Results written to: {Path(output_path).resolve()}")
    return payload


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Robust BLS transit discovery for photometric CSV data")
    parser.add_argument("--input", type=Path, default=Path("CLEANED_PHOTOMETRY.csv"))
    parser.add_argument("--target")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-period", type=float)
    parser.add_argument("--max-period", type=float)
    parser.add_argument("--expected-period", type=float)
    args = parser.parse_args()
    curve = load_light_curve(args.input, args.target)
    result = detect_transit(curve["time_bjd"], curve["raw_flux"], curve["flux_err"], expected_period=args.expected_period, min_period=args.min_period, max_period=args.max_period)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()

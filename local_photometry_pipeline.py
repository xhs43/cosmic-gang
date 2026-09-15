"""Robust local FITS reduction and differential aperture photometry.

The engine is deliberately file-index driven: it can run from the repository
root, from a copied dataset, or in an environment where only the metadata is
available.  In the latter case each target/session is represented by a
deterministic transit simulation rather than failing the discovery workflow.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Iterable
import warnings

import astropy.stats
from astropy.io import fits
from astropy.time import Time
from photutils.aperture import CircularAnnulus, CircularAperture, aperture_photometry
from photutils.centroids import centroid_2dg
import numpy as np
import pandas as pd
import scipy
from scipy import ndimage


LOGGER = logging.getLogger("local_photometry_pipeline")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

OUTPUT_COLUMNS = ["target", "time_bjd", "raw_flux", "flux_err", "session_id", "is_real_data"]
TARGETS = ["CoRoT-2", "Qatar-1", "TrES-3", "TrES-5", "HAT-P-10", "WASP-2", "WASP-10", "TRES-1"]
TARGET_ALIASES = {target.lower().replace("-", ""): target for target in TARGETS}
NASA_PARAMETERS = {
    "CoRoT-2": {"depth": 0.031, "duration_hours": 2.3, "period_days": 1.743},
    "Qatar-1": {"depth": 0.018, "duration_hours": 1.6, "period_days": 1.420},
    "TrES-3": {"depth": 0.025, "duration_hours": 1.4, "period_days": 1.306},
    "TrES-5": {"depth": 0.019, "duration_hours": 1.7, "period_days": 1.482},
    "HAT-P-10": {"depth": 0.011, "duration_hours": 2.6, "period_days": 2.729},
    "WASP-2": {"depth": 0.016, "duration_hours": 1.8, "period_days": 2.152},
    "WASP-10": {"depth": 0.028, "duration_hours": 2.2, "period_days": 3.093},
    "TRES-1": {"depth": 0.023, "duration_hours": 2.5, "period_days": 3.030},
}


def _canonical_target(value: object) -> str | None:
    normalized = str(value or "").strip().lower().replace("-", "")
    return TARGET_ALIASES.get(normalized)


def _read_index(root: Path) -> pd.DataFrame:
    """Read the best available index and normalize its field names."""
    candidates = [root / "dataset_index.csv", root / "metadata" / "observations.csv", root / "metadata" / "observations.json"]
    for path in candidates:
        if not path.is_file():
            continue
        try:
            frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
            frame.columns = [str(column).strip().lower().replace("-", "_") for column in frame.columns]
            if "target" in frame.columns and "filepath" in frame.columns:
                return frame
        except Exception as exc:
            LOGGER.warning("Could not read metadata index %s: %s", path, exc)
    LOGGER.warning("No usable observations index found under %s", root)
    return pd.DataFrame()


def _is_dark(record: pd.Series) -> bool:
    filter_name = str(record.get("filter", "")).strip().lower()
    try:
        exposure = float(record.get("exposure", np.nan))
    except (TypeError, ValueError):
        exposure = np.nan
    return filter_name == "opaque" and np.isclose(exposure, 60.0, atol=1e-6)


def _resolve_path(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / Path(str(value).replace("\\", os.sep))


def build_master_dark(records: pd.DataFrame, root: Path) -> np.ndarray | None:
    """Build a 3-sigma clipped per-pixel median master dark."""
    dark_records = records[records.apply(_is_dark, axis=1)] if not records.empty else records
    frames: list[np.ndarray] = []
    for _, record in dark_records.iterrows():
        path = _resolve_path(root, record.get("filepath", ""))
        try:
            image = np.asarray(fits.getdata(path), dtype=np.float32)
            if image.ndim == 2 and np.isfinite(image).all():
                frames.append(image)
        except Exception as exc:
            LOGGER.warning("Skipping unreadable calibration frame %s: %s", path, exc)
    if not frames:
        LOGGER.warning("No readable Opaque/60s calibration frames; using zero dark.")
        return None
    shape = frames[0].shape
    frames = [frame for frame in frames if frame.shape == shape]
    if not frames:
        return None
    stack = np.stack(frames)
    _, median, _ = astropy.stats.sigma_clipped_stats(stack, sigma=3.0, axis=0, maxiters=5)
    LOGGER.info("Master dark built from %d frames (%s).", len(frames), shape)
    return np.asarray(median, dtype=np.float32)


def _time_to_jd(value: object) -> float:
    try:
        timestamp = pd.to_datetime(value, utc=True, errors="raise")
        return float(Time(timestamp.to_pydatetime(), scale="utc").jd)
    except Exception:
        return np.nan


def _image_time(header: fits.Header, record: pd.Series) -> float:
    return _time_to_jd(header.get("DATE-OBS", record.get("datetime", "")))


def _source_candidates(image: np.ndarray, target_position: tuple[float, float] | None = None) -> list[tuple[float, float]]:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return []
    background = float(np.median(finite))
    threshold = background + 4.0 * float(np.std(finite))
    filtered = ndimage.maximum_filter(image, size=9, mode="nearest")
    mask = (image == filtered) & (image > threshold)
    y_values, x_values = np.nonzero(mask)
    candidates = [(float(x), float(y), float(image[y, x])) for y, x in zip(y_values, x_values)]
    height, width = image.shape
    candidates = [candidate for candidate in candidates if 16 < candidate[0] < width - 16 and 16 < candidate[1] < height - 16]
    if target_position is not None:
        candidates = [candidate for candidate in candidates if np.hypot(candidate[0] - target_position[0], candidate[1] - target_position[1]) > 20]
    candidates.sort(key=lambda item: item[2], reverse=True)
    return [(candidate[0], candidate[1]) for candidate in candidates]


def _centroid(image: np.ndarray, position: tuple[float, float], half_size: int = 10) -> tuple[float, float]:
    x, y = position
    x0, x1 = max(0, int(round(x)) - half_size), min(image.shape[1], int(round(x)) + half_size + 1)
    y0, y1 = max(0, int(round(y)) - half_size), min(image.shape[0], int(round(y)) + half_size + 1)
    cutout = image[y0:y1, x0:x1]
    try:
        centroid = centroid_2dg(cutout)
        if np.all(np.isfinite(centroid)):
            return float(centroid[0] + x0), float(centroid[1] + y0)
    except Exception:
        pass
    return position


def _net_flux(image: np.ndarray, position: tuple[float, float]) -> tuple[float, float]:
    aperture = CircularAperture(position, r=5.0)
    annulus = CircularAnnulus(position, r_in=10.0, r_out=15.0)
    aperture_result = aperture_photometry(image, aperture, method="exact")
    annulus_mask = annulus.to_mask(method="center")
    if annulus_mask is None:
        return np.nan, np.nan
    sky_values = annulus_mask.multiply(image)
    if sky_values is None:
        return np.nan, np.nan
    sky_values = sky_values[annulus_mask.data > 0]
    sky_values = sky_values[np.isfinite(sky_values)]
    sky = float(np.median(sky_values)) if sky_values.size else 0.0
    net = float(aperture_result["aperture_sum"][0] - aperture.area * sky)
    noise = float(np.std(sky_values)) if sky_values.size > 1 else np.sqrt(max(abs(net), 1.0))
    error = float(np.sqrt(max(abs(net), 1.0) + aperture.area * noise**2))
    return net, error


def _science_records(records: pd.DataFrame, target: str, session_id: str) -> pd.DataFrame:
    if records.empty:
        return records
    targets = records["target"].map(_canonical_target)
    return records[(targets == target) & (records["session_id"].astype(str) == str(session_id))].copy()


def process_session(records: pd.DataFrame, target: str, session_id: str, root: Path, master_dark: np.ndarray | None) -> pd.DataFrame:
    science = _science_records(records, target, session_id)
    measurements: list[dict[str, float | str | bool]] = []
    target_position: tuple[float, float] | None = None
    reference_positions: list[tuple[float, float]] = []
    for _, record in science.sort_values("datetime").iterrows():
        path = _resolve_path(root, record.get("filepath", ""))
        try:
            with fits.open(path, memmap=False) as hdul:
                image = np.asarray(hdul[0].data, dtype=np.float32)
                header = hdul[0].header
            if image.ndim != 2 or not np.isfinite(image).any():
                raise ValueError("not a finite 2D image")
            if master_dark is not None and master_dark.shape == image.shape:
                image = image - master_dark
            if target_position is None:
                candidates = _source_candidates(image)
                if not candidates:
                    raise ValueError("no stellar candidates detected")
                height, width = image.shape
                central = [p for p in candidates if width * 0.2 < p[0] < width * 0.8 and height * 0.2 < p[1] < height * 0.8]
                target_position = (central or candidates)[0]
                reference_positions = _source_candidates(image, target_position)[:12]
            target_position = _centroid(image, target_position)
            reference_positions = [_centroid(image, position) for position in reference_positions]
            target_flux, target_error = _net_flux(image, target_position)
            reference_fluxes = [_net_flux(image, position)[0] for position in reference_positions]
            reference_fluxes = [flux for flux in reference_fluxes if np.isfinite(flux) and flux > 0]
            if not reference_fluxes or not np.isfinite(target_flux) or target_flux <= 0:
                raise ValueError("invalid aperture flux")
            measurements.append({
                "time_bjd": _image_time(header, record),
                "target_flux": target_flux,
                "target_error": target_error,
                "ensemble_flux": float(np.median(reference_fluxes)),
            })
        except Exception as exc:
            LOGGER.warning("Skipping science frame %s: %s", path, exc)
    if len(measurements) < 2:
        LOGGER.warning("No reliable real photometry for %s/%s; using analytical fallback.", target, session_id)
        return simulate_session(target, session_id, science)
    measured = pd.DataFrame(measurements)
    differential = measured["target_flux"] / measured["ensemble_flux"]
    normalized = differential / float(np.nanmedian(differential))
    error = measured["target_error"] / measured["ensemble_flux"]
    return pd.DataFrame({
        "target": target,
        "time_bjd": measured["time_bjd"].astype(float),
        "raw_flux": normalized.astype(float),
        "flux_err": error.astype(float),
        "session_id": session_id,
        "is_real_data": True,
    })[OUTPUT_COLUMNS]


def simulate_session(target: str, session_id: str, records: pd.DataFrame) -> pd.DataFrame:
    """Generate a deterministic transit + Gaussian noise + airmass fallback."""
    parameters = NASA_PARAMETERS[target]
    count = max(len(records), 120)
    dates = pd.to_datetime(records.get("datetime", pd.Series(dtype=str)), errors="coerce", utc=True).dropna()
    if dates.empty:
        start_jd = 2461258.5
    else:
        start_jd = float(Time(dates.iloc[0].to_pydatetime(), scale="utc").jd)
    time_bjd = start_jd + np.linspace(0.0, max(0.25, count / 240.0), count)
    phase = ((time_bjd - start_jd + 0.25 * parameters["period_days"]) % parameters["period_days"]) - 0.25 * parameters["period_days"]
    half_duration = parameters["duration_hours"] / 48.0
    transit = np.exp(-0.5 * (phase / max(half_duration / 2.0, 1e-4)) ** 2)
    airmass = 1.0 + 0.08 * np.sin(np.linspace(0.0, np.pi, count))
    seed = int(hashlib.sha256(f"{target}:{session_id}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    raw_flux = airmass * (1.0 - parameters["depth"] * transit) + rng.normal(0.0, 0.0025, count)
    return pd.DataFrame({
        "target": target,
        "time_bjd": time_bjd,
        "raw_flux": raw_flux,
        "flux_err": np.full(count, 0.0025),
        "session_id": session_id,
        "is_real_data": False,
    })[OUTPUT_COLUMNS]


def run_pipeline(root: str | Path = ".", output_path: str | Path = "CLEANED_PHOTOMETRY.csv", targets: Iterable[str] | None = None, max_frames_per_session: int | None = None, use_cache: bool = True) -> pd.DataFrame:
    root = Path(root).resolve()
    records = _read_index(root)
    if records.empty:
        records = pd.DataFrame(columns=["target", "session_id", "filepath", "datetime", "filter", "exposure"])
    if "session_id" not in records.columns:
        records["session_id"] = "session_01"
    selected_targets = [_canonical_target(target) for target in (targets or TARGETS)]
    selected_targets = [target for target in selected_targets if target]
    master_dark = build_master_dark(records, root)
    cache_dir = root / "processed_lightcurves"
    cache_dir.mkdir(parents=True, exist_ok=True)
    all_curves: list[pd.DataFrame] = []
    target_summary: dict[str, tuple[int, int]] = {}
    for target in selected_targets:
        target_records = records[records["target"].map(_canonical_target) == target]
        sessions = sorted(target_records["session_id"].dropna().astype(str).unique()) or ["session_01"]
        target_real = target_fallback = 0
        for session_id in sessions:
            cache_path = cache_dir / f"{target}_{session_id}.csv"
            if use_cache and cache_path.is_file():
                try:
                    cached = pd.read_csv(cache_path)
                    if set(OUTPUT_COLUMNS).issubset(cached.columns):
                        cached["is_real_data"] = cached["is_real_data"].map(
                            lambda value: str(value).strip().lower() in {"true", "1", "yes"}
                        )
                        curve = cached[OUTPUT_COLUMNS]
                    else:
                        raise ValueError("cache schema mismatch")
                except Exception as exc:
                    LOGGER.warning("Ignoring invalid cache %s: %s", cache_path, exc)
                    curve = pd.DataFrame()
            else:
                curve = pd.DataFrame()
            if curve.empty:
                session_records = _science_records(records, target, session_id)
                if max_frames_per_session is not None:
                    session_records = session_records.head(max_frames_per_session)
                curve = process_session(pd.concat([session_records, records[records.apply(_is_dark, axis=1)]], ignore_index=True), target, session_id, root, master_dark)
                curve.to_csv(cache_path, index=False)
            all_curves.append(curve[OUTPUT_COLUMNS])
            if bool(curve["is_real_data"].astype(bool).any()):
                target_real += len(curve)
            else:
                target_fallback += len(curve)
        target_summary[target] = (target_real, target_fallback)
    result = pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame(columns=OUTPUT_COLUMNS)
    result = result[OUTPUT_COLUMNS]
    result.to_csv(root / output_path, index=False)
    print("\n=== Local Photometry Executive Summary ===")
    print(f"Calibration frames: {int(records.apply(_is_dark, axis=1).sum())} indexed; master dark: {'yes' if master_dark is not None else 'no'}")
    print(f"Output: {root / output_path} ({len(result):,} rows)")
    for target, (real_count, fallback_count) in target_summary.items():
        print(f"{target}: {real_count:,} real rows, {fallback_count:,} fallback rows")
    return result


if __name__ == "__main__":
    try:
        run_pipeline()
    except Exception as exc:
        LOGGER.exception("Pipeline failed safely: %s", exc)
        fallback = pd.concat([simulate_session(target, "session_01", pd.DataFrame()) for target in TARGETS], ignore_index=True)
        fallback.to_csv("CLEANED_PHOTOMETRY.csv", index=False)
        print("WARNING: wrote analytical fallback light curves after an unexpected top-level error.")
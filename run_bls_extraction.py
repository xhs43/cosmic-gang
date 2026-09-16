"""
run_bls_extraction.py
--------------------
Standalone script that:
1. Loads qatar1_real_lightcurve.csv (or falls back to synthetic CoRoT-2 data).
2. Sigma-clips outliers at 3.5-sigma using astropy.
3. Runs Box Least Squares (BLS) over a [1.0, 5.0] day period grid.
4. Extracts 7 astrophysical features.
5. Saves them to extracted_features.json and prints a single JSON summary.

Does NOT touch transit_detection_pipeline.py, app.py, or any other existing file.
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from astropy.timeseries import BoxLeastSquares
from scipy.stats import kurtosis as scipy_kurtosis, skew as scipy_skew

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "qatar1_real_lightcurve.csv"

# ──────────────────────────────────────────────────────────────────────────────
# 1.  Load or synthesise data
# ──────────────────────────────────────────────────────────────────────────────

def load_or_synthesise() -> tuple[np.ndarray, np.ndarray]:
    """Return (time_array_days, flux_array).  Falls back to synthetic if CSV absent."""
    if CSV_PATH.is_file():
        df = pd.read_csv(CSV_PATH)
        # Accept either column name set
        if "time" in df.columns and "flux" in df.columns:
            t = pd.to_numeric(df["time"], errors="coerce")
            # If timestamps are ISO strings, convert to MJD offset
            if df["time"].dtype == object:
                from astropy.time import Time
                try:
                    t_astropy = Time(df["time"].values, format="isot", scale="utc")
                    t = t_astropy.mjd
                    t = t - t.min()
                except Exception:
                    t = np.arange(len(df), dtype=float) / 48.0  # assume 30-min cadence
            else:
                t = t.values
                t = t - np.nanmin(t)
            f = df["flux"].values.astype(float)
            valid = np.isfinite(t) & np.isfinite(f)
            return t[valid], f[valid]
        else:
            print("Warning: CSV does not have expected columns; falling back to synthetic data.", file=sys.stderr)

    # ── Synthetic fallback (uses exoplanet_simulator if present, else pure numpy) ──
    try:
        sys.path.insert(0, str(ROOT))
        from exoplanet_simulator import SimulationConfig, run_simulation
        cfg = SimulationConfig(
            scenario="planet",
            seed=42,
            sample_count=1200,
            period_days=1.5,
            duration_days=0.10,
            depth=0.015,
            noise_sigma=0.002,
        )
        res = run_simulation(cfg)
        det = res["detection"]
        t = np.array(det.get("clean_time", []))
        f = np.array(det.get("clean_flux", []))
        if len(t) == 0:
            raise ValueError("Empty synthetic light curve")
        return t, f
    except Exception as ex:
        print(f"Simulator fallback failed ({ex}); generating pure numpy transit.", file=sys.stderr)

    # Last resort: pure numpy transit model
    rng = np.random.default_rng(42)
    t = np.linspace(0, 12, 1200)
    f = np.ones_like(t)
    period = 1.5
    depth  = 0.015
    dur    = 0.10
    t0     = 0.2
    phase  = ((t - t0) % period)
    in_transit = phase < dur
    f[in_transit] -= depth
    f += rng.normal(0, 0.002, len(t))
    return t, f


# ──────────────────────────────────────────────────────────────────────────────
# 2.  Sigma clipping
# ──────────────────────────────────────────────────────────────────────────────

def sigma_filter(t: np.ndarray, f: np.ndarray, sigma: float = 3.5) -> tuple[np.ndarray, np.ndarray]:
    clipped = sigma_clip(f, sigma=sigma, maxiters=5)
    mask = ~clipped.mask if hasattr(clipped, "mask") else np.ones(len(f), dtype=bool)
    return t[mask], f[mask]


# ──────────────────────────────────────────────────────────────────────────────
# 3.  BLS search
# ──────────────────────────────────────────────────────────────────────────────

def run_bls(t: np.ndarray, f: np.ndarray, p_min: float = 1.0, p_max: float = 5.0) -> dict:
    baseline = t.max() - t.min()
    if baseline <= 0:
        raise ValueError("Time baseline is zero – cannot run BLS.")

    bls = BoxLeastSquares(t, f)
    # Build a log-spaced period grid; at least 1000 steps
    periods = np.logspace(np.log10(p_min), np.log10(p_max), max(1000, 3 * len(t)))
    result  = bls.power(periods, duration=np.linspace(0.02, 0.3, 50))

    best_idx = np.argmax(result.power)
    best_period   = float(result.period[best_idx])
    best_power    = float(result.power[best_idx])
    best_duration = float(result.duration[best_idx])
    best_t0       = float(result.transit_time[best_idx])

    # Transit depth from BLS stats at the best period
    stats = bls.compute_stats(best_period, best_duration, best_t0)
    depth    = float(stats["depth"][0])
    depth_err= float(stats["depth"][1]) if stats["depth"][1] > 0 else 1e-9

    return {
        "period":    best_period,
        "depth":     depth,
        "depth_err": depth_err,
        "duration":  best_duration,
        "t0":        best_t0,
        "power":     best_power,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Seven features
# ──────────────────────────────────────────────────────────────────────────────

def extract_features(bls_result: dict, f: np.ndarray) -> dict:
    period    = bls_result["period"]
    depth     = bls_result["depth"]
    duration  = bls_result["duration"]
    power     = bls_result["power"]
    depth_err = bls_result["depth_err"]

    duration_ratio = duration / period if period > 0 else 0.0
    depth_snr      = depth / depth_err if depth_err > 0 else 0.0
    power_snr      = float(power)

    # Out-of-transit flux statistics
    oot = f[f > (np.median(f) - depth / 2)]
    skewness = float(scipy_skew(oot)) if len(oot) > 3 else 0.0
    kurt     = float(scipy_kurtosis(oot, fisher=True)) if len(oot) > 3 else 0.0

    return {
        "period":         round(period, 6),
        "transit_depth":  round(depth, 6),
        "duration_ratio": round(duration_ratio, 6),
        "depth_snr":      round(depth_snr, 4),
        "power_snr":      round(power_snr, 4),
        "skewness":       round(skewness, 6),
        "kurtosis":       round(kurt, 6),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    # Load
    t, f = load_or_synthesise()
    print(f"Loaded {len(t)} photometric points.", file=sys.stderr)

    # Sigma clip
    t, f = sigma_filter(t, f, sigma=3.5)
    print(f"After 3.5-sigma clip: {len(t)} points remain.", file=sys.stderr)

    # BLS
    bls_result = run_bls(t, f, p_min=1.0, p_max=5.0)

    # 7 features
    features = extract_features(bls_result, f)

    # Output payload
    output = {
        "features": features,
        "bls_max_power": round(bls_result["power"], 6),
        "bls_period_days": round(bls_result["period"], 6),
        "bls_depth": round(bls_result["depth"], 6),
        "bls_duration_days": round(bls_result["duration"], 6),
    }

    # Save JSON
    out_path = ROOT / "extracted_features.json"
    with open(out_path, "w") as fh:
        json.dump(output, fh, indent=2)

    # Single-line print
    print(json.dumps(output))


if __name__ == "__main__":
    main()

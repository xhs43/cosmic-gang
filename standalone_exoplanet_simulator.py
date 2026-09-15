"""Standalone browser simulator for the local XGBoost transit vetter."""

from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from transit_detection_pipeline import (
    NASA_BENCHMARKS,
    detrend_airmass,
    load_photometry,
    search_bls,
    validate_detection,
)
from xgb_validation_layer import extract_physics_features, verify_transit
from xgb_validation_layer import FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = 8765
PHOTOMETRY_PATH = ROOT / "CLEANED_PHOTOMETRY.csv"
FRAME_INDEX_PATH = ROOT / "CLEANED_DATA.csv"
MODEL_PATH = ROOT / "xgb_transit_vetter.json"
PAGE_PATH = ROOT / "standalone_exoplanet_simulator.html"


def _load_data() -> pd.DataFrame:
    if not PHOTOMETRY_PATH.is_file():
        raise FileNotFoundError(f"Missing existing photometry: {PHOTOMETRY_PATH}")
    data = load_photometry(PHOTOMETRY_PATH)
    return data[data["is_real_data"]].copy()


def _load_frame_index() -> pd.DataFrame:
    if not FRAME_INDEX_PATH.is_file():
        return pd.DataFrame(columns=["TARGET_NAME", "FRAME_TYPE"])
    index = pd.read_csv(FRAME_INDEX_PATH)
    index["TARGET_NAME"] = index["TARGET_NAME"].astype(str).str.strip()
    return index[index["FRAME_TYPE"].astype(str).str.upper().eq("SCIENCE")].copy()


def _canonical_target(value: object) -> str | None:
    normalized = str(value).strip().lower().replace("-", "")
    aliases = {name.lower().replace("-", ""): name for name in NASA_BENCHMARKS}
    return aliases.get(normalized)


def available_targets(data: pd.DataFrame, frame_index: pd.DataFrame) -> list[str]:
    measured = set(data["target"].dropna().astype(str))
    indexed = {_canonical_target(value) for value in frame_index["TARGET_NAME"]}
    return [target for target in NASA_BENCHMARKS if target in measured or target in indexed]


def analyze_target(data: pd.DataFrame, target: str) -> dict[str, object]:
    if target not in NASA_BENCHMARKS:
        raise ValueError(f"Unknown target: {target}")
    curve = data[data["target"] == target].sort_values("time_bjd").reset_index(drop=True)
    if curve.empty:
        return {
            "target": target,
            "analysis_available": False,
            "message": "إطارات FITS موجودة، لكن لا يوجد قياس ضوئي aperture صالح لهذا الهدف.",
            "sources": {"frames": str(FRAME_INDEX_PATH.name), "photometry": str(PHOTOMETRY_PATH.name), "model": str(MODEL_PATH.name)},
            "sample_count": 0,
            "real_data_count": 0,
        }

    truth = NASA_BENCHMARKS[target]
    time = curve["time_bjd"].to_numpy(float)
    raw_flux = curve["raw_flux"].to_numpy(float)
    detrended = detrend_airmass(time, raw_flux, window_len=31, poly_order=2)
    detection = search_bls(time, detrended, curve["flux_err"].to_numpy(float), truth["period"], truth["duration_hours"])
    validation = validate_detection(time, detrended, detection, truth)
    result = {"target": target, "detection": detection, "validation": validation}
    features = extract_physics_features(curve.assign(raw_flux=detrended), result)
    ml = verify_transit(features, model_path=MODEL_PATH)
    return {
        "target": target,
        "analysis_available": True,
        "sample_count": int(len(curve)),
        "real_data_count": int(curve["is_real_data"].astype(bool).sum()),
        "features": features,
        "detection": detection,
        "validation": validation,
        "ml": ml,
        "sources": {
            "photometry": str(PHOTOMETRY_PATH.name),
            "model": str(MODEL_PATH.name),
            "benchmarks": "transit_detection_pipeline.NASA_BENCHMARKS",
            "method": "detrend_airmass -> BoxLeastSquares -> verify_transit",
        },
        "series": {
            "time": time.tolist(),
            "raw_flux": raw_flux.tolist(),
            "detrended_flux": np.nan_to_num(detrended, nan=1.0).tolist(),
        },
    }


def catalog(data: pd.DataFrame, frame_index: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    frame_counts = {}
    for value in frame_index["TARGET_NAME"]:
        canonical = _canonical_target(value)
        if canonical:
            frame_counts[canonical] = frame_counts.get(canonical, 0) + 1
    for target in available_targets(data, frame_index):
        try:
            result = analyze_target(data, target)
            if not result.get("analysis_available"):
                rows.append({"target": target, "frames": frame_counts.get(target, 0), "samples": 0, "real_data_count": 0, "verdict": "NO PHOTOMETRY", "confidence_score": None, "detected": False, "snr": None, "transit_depth": None, "source": str(FRAME_INDEX_PATH.name)})
                continue
            rows.append({
                "target": target,
                "frames": frame_counts.get(target, 0),
                "samples": result["sample_count"],
                "real_data_count": result["real_data_count"],
                "verdict": result["ml"]["verdict"],
                "confidence_score": result["ml"]["confidence_score"],
                "detected": bool(result["detection"].get("detected")),
                "snr": result["features"]["snr"],
                "transit_depth": result["features"]["transit_depth"],
                "source": result["sources"]["photometry"],
            })
        except Exception as exc:
            rows.append({"target": target, "error": str(exc), "source": str(PHOTOMETRY_PATH.name)})
    return rows


class SimulatorHandler(BaseHTTPRequestHandler):
    data = _load_data()
    frame_index = _load_frame_index()

    def _send_json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_page(self) -> None:
        body = PAGE_PATH.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 10000:
            raise ValueError("JSON body is missing or too large")
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("JSON body must be an object")
        return payload

    def do_GET(self) -> None:  # noqa: N802
        request = urlparse(self.path)
        try:
            if request.path in {"/", "/index.html"}:
                self._send_page()
                return
            if request.path == "/api/targets":
                self._send_json({"targets": available_targets(self.data, self.frame_index)})
                return
            if request.path == "/api/catalog":
                self._send_json({
                    "rows": catalog(self.data, self.frame_index),
                    "sources": {
                        "frames": str(FRAME_INDEX_PATH.name),
                        "photometry": str(PHOTOMETRY_PATH.name),
                        "model": str(MODEL_PATH.name),
                        "benchmarks": "transit_detection_pipeline.NASA_BENCHMARKS",
                    },
                })
                return
            if request.path == "/api/report":
                targets = parse_qs(request.query).get("target", [])
                selected = targets or available_targets(self.data, self.frame_index)
                report = [analyze_target(self.data, target) for target in selected]
                self._send_json({"generated_from": str(PHOTOMETRY_PATH.name), "results": report})
                return
            if request.path == "/api/analyze":
                target = parse_qs(request.query).get("target", [""])[0]
                self._send_json(analyze_target(self.data, target))
                return
            self._send_json({"error": "Not found"}, 404)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def do_POST(self) -> None:  # noqa: N802
        request = urlparse(self.path)
        try:
            if request.path != "/api/predict":
                self._send_json({"error": "Not found"}, 404)
                return
            payload = self._read_json()
            missing = [column for column in FEATURE_COLUMNS if column not in payload]
            if missing:
                raise ValueError(f"Missing features: {', '.join(missing)}")
            features = {}
            for column in FEATURE_COLUMNS:
                value = float(payload[column])
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{column} must be a finite non-negative number")
                features[column] = value
            result = verify_transit(features, model_path=MODEL_PATH)
            self._send_json({
                "input": features,
                "ml": result,
                "sources": {
                    "model": str(MODEL_PATH.name),
                    "feature_contract": "xgb_validation_layer.FEATURE_COLUMNS",
                    "training_definition": "xgb_validation_layer.build_astrophysical_corpus",
                },
            })
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send_json({"error": str(exc)}, 400)
        except Exception as exc:
            self._send_json({"error": str(exc)}, 500)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), SimulatorHandler)
    print(f"Standalone exoplanet simulator: http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSimulator stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
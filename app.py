import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
from pathlib import Path

from exoplanet_simulator import SimulationConfig, run_simulation
from transit_detection_pipeline import detect_transit
from xgb_validation_layer import FEATURE_COLUMNS, extract_physics_features, validate_candidate

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Exoplanet Transit Vetting Pipeline",
    page_icon=None,
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark card */
.card {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 16px;
    padding: 24px 28px;
    margin: 12px 0;
}

/* Phase badge */
.phase-badge {
    display: inline-block;
    background: linear-gradient(90deg, #e94560, #0f3460);
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 4px 14px;
    border-radius: 20px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* Verdict box */
.verdict-correct {
    background: linear-gradient(135deg, #0d4f3c, #1a7a5e);
    border: 2px solid #2ecc71;
    border-radius: 12px;
    padding: 18px 24px;
    color: #2ecc71;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
}
.verdict-wrong {
    background: linear-gradient(135deg, #4f0d1a, #7a1a2e);
    border: 2px solid #e74c3c;
    border-radius: 12px;
    padding: 18px 24px;
    color: #e74c3c;
    font-size: 1.4rem;
    font-weight: 700;
    text-align: center;
}

/* Feature bar */
.feat-bar-wrap { margin: 6px 0; }
.feat-label { font-size: 0.82rem; color: #aaa; font-family: 'JetBrains Mono', monospace; }
.feat-val   { font-size: 0.82rem; color: #eee; font-weight: 600; float: right; }

/* Score pill */
.score-pill {
    background: linear-gradient(90deg, #e94560 0%, #533483 100%);
    color: white;
    border-radius: 30px;
    padding: 6px 22px;
    font-size: 1.1rem;
    font-weight: 700;
    display: inline-block;
}

/* Divider */
hr.glow { border: none; border-top: 1px solid #0f3460; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "challenge_res":    None,   # run_simulation output
        "challenge_scen":   None,   # hidden true scenario
        "challenge_params": None,   # SimulationConfig kwargs
        "user_answer":      None,   # "planet" | "false_positive"
        "revealed":         False,
        "correct_count":    0,
        "total_count":      0,
        "challenge_seed":   random.randint(0, 99999),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

SCENARIOS = ["planet", "eclipsing_binary", "grazing_binary", "stellar_flare", "instrumental_noise"]
PLANET_SCENARIOS = {"planet"}

SCENARIO_LABELS = {
    "planet":             "Exoplanet Transit",
    "eclipsing_binary":   "Eclipsing Binary",
    "grazing_binary":     "Grazing Eclipsing Binary",
    "stellar_flare":      "Stellar Flare",
    "instrumental_noise": "Instrumental Noise Artifact",
}

FEATURE_DESCRIPTIONS = {
    "transit_depth":            "Fractional dimming at mid-transit",
    "period_days":              "Orbital/signal period",
    "duration_ratio":           "Transit duration / period",
    "snr":                      "Signal-to-noise ratio",
    "odd_even_diff":            "Depth difference between odd/even transits",
    "secondary_depth":          "Depth of secondary eclipse (if any)",
    "background_flux_variance": "Out-of-transit flux scatter",
}

# ── Helper: random SimulationConfig ──────────────────────────────────────────
def _random_config(seed: int) -> tuple[SimulationConfig, str]:
    rng = random.Random(seed)
    scen = rng.choice(SCENARIOS)
    period = round(rng.uniform(0.8, 4.0), 3)
    duration = round(rng.uniform(0.04, 0.18), 3)
    depth = round(rng.uniform(0.005, 0.04), 4)
    sec_depth = round(rng.uniform(0.003, depth * 0.8), 4) if scen in ("eclipsing_binary", "grazing_binary") else 0.0
    noise = round(rng.uniform(0.001, 0.004), 4)
    red_noise = round(rng.uniform(0.0005, 0.002), 4)
    gap = round(rng.uniform(0.0, 0.15), 2)
    outlier = round(rng.uniform(0.0, 0.04), 2)
    cfg = SimulationConfig(
        scenario=scen,
        seed=seed,
        period_days=period,
        duration_days=duration,
        depth=depth,
        secondary_depth=sec_depth,
        noise_sigma=noise,
        red_noise_sigma=red_noise,
        gap_fraction=gap,
        outlier_fraction=outlier,
        sample_count=1200,
    )
    return cfg, scen

# ── Helper: light-curve figure ────────────────────────────────────────────────
def _lc_figure(det: dict, title: str = "", show_fold: bool = False):
    times = det.get("clean_time", [])
    fluxes = det.get("clean_flux", [])
    if not times:
        return None

    t = np.array(times)
    f = np.array(fluxes)

    if show_fold and det.get("period") and det.get("t0"):
        period = det["period"]
        t0 = det["t0"]
        phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
        t_plot, f_plot = phase, f
        xlabel = f"Orbital Phase  [period = {period:.3f} d]"
    else:
        t_plot, f_plot = t - t.min(), f
        xlabel = "Time [BJD - T0]  (days)"

    fig, ax = plt.subplots(figsize=(10, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    ax.scatter(t_plot, f_plot, s=3, color="#4fc3f7", alpha=0.7, rasterized=True)
    ax.set_xlabel(xlabel, color="#aaa", fontsize=10)
    ax.set_ylabel("Normalized Relative Flux", color="#aaa", fontsize=10)
    ax.set_title(title, color="#eee", fontsize=11, pad=10)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d40")
    if show_fold and det.get("duration"):
        dur = det["duration"] / 2
        ax.axvspan(-dur, dur, alpha=0.15, color="#e94560", label="Transit window")
        ax.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=9)
    plt.tight_layout()
    return fig

# ── Helper: feature bar chart ─────────────────────────────────────────────────
def _feature_figure(features: dict, planet_prob: float):
    keys = list(FEATURE_DESCRIPTIONS.keys())
    vals = [features.get(k, 0.0) for k in keys]

    # Normalise to [0,1] for bar width only (display actual values as text)
    max_abs = max(abs(v) for v in vals) or 1.0
    widths = [abs(v) / max_abs for v in vals]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    colors = ["#4fc3f7" if v >= 0 else "#e94560" for v in vals]
    bars = ax.barh(keys, widths, color=colors, height=0.55, alpha=0.85)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", color="#eee", fontsize=8,
                fontfamily="monospace")

    ax.set_xlim(0, 1.25)
    ax.set_xlabel("Normalised Feature Magnitude", color="#aaa", fontsize=9)
    ax.set_title(f"Astrophysical Feature Vector  |  Planet Probability: {planet_prob*100:.1f}%",
                 color="#eee", fontsize=10, pad=8)
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e2d40")
    plt.tight_layout()
    return fig

# =============================================================================
# TAB LAYOUT
# =============================================================================
tab_challenge, tab_simulator = st.tabs([
    "Transit Classification — Diagnostic Mode",
    "Astrophysical Vetting Pipeline"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — TRANSIT CLASSIFICATION / DIAGNOSTIC MODE
# ─────────────────────────────────────────────────────────────────────────────
with tab_challenge:

    # Header
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        st.markdown("## Automated Transit Candidate Classifier")
        st.markdown(
            "Inspect the photometric time series and determine whether the detected signal "
            "is consistent with a **planetary transit** or constitutes a **false positive**. "
            "The vetting pipeline will display its classification after you submit."
        )
    with col_h2:
        correct = st.session_state.correct_count
        total   = st.session_state.total_count
        st.markdown(
            f"<div style='text-align:right;padding-top:18px'>"
            f"<span class='score-pill'>Score: {correct} / {total}</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='glow'>", unsafe_allow_html=True)

    # Generate / Next case button
    if st.button("Generate New Case", type="primary", use_container_width=False):
        st.session_state.challenge_seed   = random.randint(0, 99999)
        st.session_state.challenge_res    = None
        st.session_state.challenge_scen   = None
        st.session_state.challenge_params = None
        st.session_state.user_answer      = None
        st.session_state.revealed         = False

    # Auto-generate on first visit
    if st.session_state.challenge_res is None:
        seed = st.session_state.challenge_seed
        cfg, true_scen = _random_config(seed)
        with st.spinner("Generating synthetic photometric time series..."):
            res = run_simulation(cfg, model_path="xgb_transit_vetter.json")
        st.session_state.challenge_res    = res
        st.session_state.challenge_scen   = true_scen
        st.session_state.challenge_params = cfg
        st.session_state.user_answer      = None
        st.session_state.revealed         = False

    res        = st.session_state.challenge_res
    true_scen  = st.session_state.challenge_scen
    det        = res["detection"]
    vetter     = res["vetter"]
    features   = res["features"]

    # Phase 1 — Observe
    st.markdown("<div class='phase-badge'>Phase 1 — Observe</div>", unsafe_allow_html=True)
    st.markdown("#### Photometric Time Series")

    fig_lc = _lc_figure(det, title="")
    if fig_lc:
        st.pyplot(fig_lc, use_container_width=True)
    else:
        st.info("No photometric data returned for this simulation.")

    # Show partial BLS stats (no scenario reveal)
    c1, c2, c3 = st.columns(3)
    c1.metric("BLS Transit Detected", "Yes" if det.get("transit_detected") else "No")
    c2.metric("Best-Fit Period",      f"{det.get('period', 0.0):.3f} d" if det.get("period") else "—")
    c3.metric("Transit Depth (BLS)",  f"{det.get('depth', 0.0):.4f}"    if det.get("depth") else "—")

    st.markdown("<hr class='glow'>", unsafe_allow_html=True)

    # Phase 2 — Classify
    if not st.session_state.revealed:
        st.markdown("<div class='phase-badge'>Phase 2 — Classify</div>", unsafe_allow_html=True)
        st.markdown("#### Is the detected signal consistent with a planetary transit?")

        btn_col1, btn_col2, _ = st.columns([1, 1, 3])
        with btn_col1:
            if st.button("CONFIRMED — Exoplanet Transit", use_container_width=True, type="primary"):
                st.session_state.user_answer = "planet"
                st.session_state.revealed    = True
                st.session_state.total_count += 1
                is_planet = true_scen in PLANET_SCENARIOS
                if is_planet:
                    st.session_state.correct_count += 1
                st.rerun()

        with btn_col2:
            if st.button("REJECTED — False Positive", use_container_width=True):
                st.session_state.user_answer = "false_positive"
                st.session_state.revealed    = True
                st.session_state.total_count += 1
                is_not_planet = true_scen not in PLANET_SCENARIOS
                if is_not_planet:
                    st.session_state.correct_count += 1
                st.rerun()

    # Phase 3 — Disclose
    if st.session_state.revealed:
        user_ans   = st.session_state.user_answer
        is_planet  = true_scen in PLANET_SCENARIOS
        user_right = (user_ans == "planet") == is_planet

        st.markdown("<div class='phase-badge'>Phase 3 — Pipeline Analysis</div>", unsafe_allow_html=True)

        # Verdict banner
        if user_right:
            st.markdown(
                f"<div class='verdict-correct'>Classification Matches Astrophysical Ground Truth — "
                f"{'Planetary Transit Confirmed' if user_ans == 'planet' else 'False Positive Correctly Rejected'}.</div>",
                unsafe_allow_html=True,
            )
        else:
            correct_label = "a PLANETARY TRANSIT" if is_planet else "a FALSE POSITIVE"
            st.markdown(
                f"<div class='verdict-wrong'>Classification Discrepancy — Signal is {correct_label}.</div>",
                unsafe_allow_html=True,
            )

        st.markdown("")

        # Results grid
        r1, r2 = st.columns(2)

        with r1:
            st.markdown("##### Ground Truth")
            st.markdown(
                f"<div class='card'>"
                f"<b>Scenario:</b> {SCENARIO_LABELS.get(true_scen, true_scen)}<br>"
                f"<b>User classification:</b> {'Exoplanet Transit' if user_ans == 'planet' else 'False Positive'}<br>"
                f"<b>Outcome:</b> {'Correct' if user_right else 'Incorrect'}"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.markdown("##### BLS Detection Report")
            st.markdown(
                f"<div class='card'>"
                f"<b>Transit detected:</b> {det.get('transit_detected', False)}<br>"
                f"<b>Period:</b> {det.get('period', 0.0):.4f} d<br>"
                f"<b>Depth:</b> {det.get('depth', 0.0):.4f}<br>"
                f"<b>SNR:</b> {det.get('snr', 0.0):.2f}<br>"
                f"<b>Duration:</b> {det.get('duration_days', det.get('duration', 0.0)):.4f} d"
                f"</div>",
                unsafe_allow_html=True,
            )

        with r2:
            conf     = vetter.get("candidate_confidence", 0.0)
            xverdict = "CONFIRMED PLANET CANDIDATE" if vetter.get("is_exoplanet_candidate") else "FALSE POSITIVE — REJECTED"
            xcolor   = "#2ecc71" if vetter.get("is_exoplanet_candidate") else "#e74c3c"

            st.markdown("##### XGBoost Vetter Output")
            st.markdown(
                f"<div class='card'>"
                f"<b>Planet probability:</b> {conf:.2f}%<br>"
                f"<b>Verdict:</b> <span style='color:{xcolor};font-weight:700'>{xverdict}</span><br>"
                f"<b>Safety flags:</b> {', '.join(vetter.get('flags', [])) or 'None'}"
                f"</div>",
                unsafe_allow_html=True,
            )

            flags = vetter.get("flags", [])
            if flags:
                for flag in flags:
                    st.warning(flag)

        # Folded light curve
        st.markdown("##### Phase-Folded Photometric Time Series")
        fig_fold = _lc_figure(det, title="Phase-Folded at BLS Best-Fit Period", show_fold=True)
        if fig_fold:
            st.pyplot(fig_fold, use_container_width=True)

        # Feature breakdown
        st.markdown("##### Astrophysical Feature Vector (7 XGBoost Inputs)")
        prob = conf / 100.0
        fig_feat = _feature_figure(features, prob)
        st.pyplot(fig_feat, use_container_width=True)

        # Feature table
        feat_rows = []
        for k, desc in FEATURE_DESCRIPTIONS.items():
            val = features.get(k, 0.0)
            feat_rows.append({"Feature": k, "Value": f"{val:.6f}", "Description": desc})
        st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

        # Safety flags
        if vetter.get("flags"):
            st.markdown("##### Physical Safety Flag Report")
            for flag in vetter["flags"]:
                st.warning(flag)

        # Next case
        st.markdown("<hr class='glow'>", unsafe_allow_html=True)
        if st.button("Next Case", type="primary", use_container_width=False):
            st.session_state.challenge_seed   = random.randint(0, 99999)
            st.session_state.challenge_res    = None
            st.session_state.challenge_scen   = None
            st.session_state.challenge_params = None
            st.session_state.user_answer      = None
            st.session_state.revealed         = False
            st.rerun()

# =============================================================================
# TAB 2 — ASTROPHYSICAL VETTING PIPELINE
# =============================================================================

# Helper: run BLS + XGBoost on a real CSV light curve
def _run_real_data(csv_path: Path) -> dict | None:
    """Load a real-observation CSV and push it through the full pipeline.
    Returns a result dict compatible with the simulator output, or None on failure."""
    try:
        df = pd.read_csv(csv_path)
        if not {"time", "flux"}.issubset(df.columns):
            st.error("Input CSV must contain at minimum the columns: 'time' and 'flux'.")
            return None

        # Convert time column: ISO strings to MJD float offset; numeric columns pass through
        t_raw = df["time"].values
        try:
            from astropy.time import Time
            t_astropy = Time(t_raw, format="isot", scale="utc")
            time = t_astropy.mjd.astype(float)
        except Exception:
            time = pd.to_numeric(df["time"], errors="coerce").values.astype(float)
        time = time - np.nanmin(time)  # zero-offset

        flux     = df["flux"].values.astype(float)
        flux_err = df["flux_err"].values.astype(float) if "flux_err" in df.columns \
                   else np.full_like(flux, np.nanstd(flux) * 0.1)

        valid = np.isfinite(time) & np.isfinite(flux)
        time, flux, flux_err = time[valid], flux[valid], flux_err[valid]

        with st.spinner("Running BLS period search on observed time series..."):
            detection = detect_transit(time, flux, flux_err)

        # Build a minimal DataFrame for feature extraction
        curve_df = pd.DataFrame({
            "target": "REAL_OBS",
            "time_bjd": time,
            "raw_flux": flux,
            "flux_err": flux_err,
            "session_id": "real_01",
            "is_real_data": True,
        })
        transit_result = {"target": "REAL_OBS", "detection": detection, "validation": {}}
        feats  = extract_physics_features(curve_df, transit_result)
        vetter = validate_candidate(feats, model_path="xgb_transit_vetter.json")

        return {
            "detection": {
                **detection,
                "clean_time": time.tolist(),
                "clean_flux": flux.tolist(),
            },
            "features": {k: feats[k] for k in FEATURE_COLUMNS},
            "vetter":   vetter,
        }
    except Exception as exc:
        st.error(f"Pipeline execution error: {exc}")
        return None


with tab_simulator:
    st.markdown("## Astrophysical Vetting Pipeline")
    st.markdown(
        "Configure and execute the photometric pipeline on a synthetic light curve, "
        "or load an observed photometric time series for transit candidate vetting."
    )

    with st.sidebar:
        st.header("Pipeline Controls")

        # Data-source toggle
        sim_mode = st.radio(
            "Data Source",
            ["Synthetic Transit Simulation", "Observed Photometric Time-Series (Qatar-1)"],
            index=0,
            help="Select whether to generate a synthetic light curve or load real photometric data.",
        )
        st.markdown("<hr class='glow'>", unsafe_allow_html=True)

        if sim_mode == "Synthetic Transit Simulation":
            # Synthetic controls (original, unchanged)
            scenario = st.selectbox(
                "Astrophysical Scenario",
                ["planet", "eclipsing_binary", "grazing_binary", "stellar_flare", "instrumental_noise"]
            )

            st.subheader("Transit / Signal Parameters")
            period_days     = st.number_input("Orbital Period (days)",   value=1.5,   step=0.1)
            duration_days   = st.number_input("Transit Duration (days)", value=0.1,   step=0.01)
            depth           = st.number_input("Transit Depth",           value=0.015, step=0.005, format="%.4f")
            secondary_depth = st.number_input("Secondary Eclipse Depth", value=0.0,   step=0.005, format="%.4f")

            st.subheader("Noise Model")
            noise_sigma     = st.number_input("White Noise Sigma",       value=0.0025, step=0.0005, format="%.4f")
            red_noise_sigma = st.number_input("Red Noise Sigma",         value=0.001,  step=0.0005, format="%.4f")

            st.subheader("Observational Parameters")
            sample_count    = st.number_input("Cadence (Sample Count)",  value=1200, step=100)
            gap_fraction    = st.slider("Gap Fraction",     0.0, 1.0, 0.0)
            outlier_fraction = st.slider("Outlier Fraction", 0.0, 1.0, 0.0)

            st.subheader("Reproducibility")
            seed = st.number_input("Random Seed", value=1103, step=1)

            run_btn = st.button("Run Simulation", type="primary", use_container_width=True)

        else:
            # Real-data controls
            _default_csv = str(Path("qatar1_real_lightcurve.csv").resolve())
            real_csv_path = st.text_input(
                "Photometric CSV File Path",
                value=_default_csv,
                help="CSV must contain columns: time, flux [, flux_err]",
            )
            run_btn = st.button("Run Photometric Analysis", type="primary", use_container_width=True)

    # Execution block
    if run_btn:
        if sim_mode == "Synthetic Transit Simulation":
            # Synthetic path (original, unchanged logic)
            config = SimulationConfig(
                scenario=scenario,
                period_days=period_days,
                duration_days=duration_days,
                depth=depth,
                secondary_depth=secondary_depth,
                noise_sigma=noise_sigma,
                red_noise_sigma=red_noise_sigma,
                sample_count=sample_count,
                gap_fraction=gap_fraction,
                outlier_fraction=outlier_fraction,
                seed=seed,
            )
            with st.spinner("Executing simulation pipeline..."):
                res = run_simulation(config, model_path="xgb_transit_vetter.json")

            det      = res["detection"]
            vetter   = res["vetter"]
            features = res["features"]

            st.header("Simulation Results")

            st.subheader("1. Synthetic Photometric Time Series")
            fig = _lc_figure(det, title="Synthetic Light Curve")
            if fig:
                st.pyplot(fig, use_container_width=True)
            else:
                st.write("No light curve data returned.")

            st.subheader("2. BLS Detection Parameters")
            st.write(f"**Transit Detected:** {det.get('transit_detected', False)}")
            st.write(f"**Period:** {det.get('period', 0.0):.4f} days")
            st.write(f"**Depth:** {det.get('depth', 0.0):.4f}")
            st.write(f"**SNR:** {det.get('snr', 0.0):.2f}")

            st.subheader("3. Astrophysical Feature Vector (XGBoost Input)")
            st.json(features)

            st.subheader("4. XGBoost Vetter Output")
            st.metric("Planet Probability", f"{vetter['candidate_confidence']:.2f}%")

            st.subheader("5. Classification Result")
            if vetter["is_exoplanet_candidate"]:
                st.success("ACCEPTED — Exoplanet Transit Candidate")
            else:
                st.error("REJECTED — False Positive / Noise Artifact")

            st.subheader("6. Physical Safety Flag Report")
            if vetter["flags"]:
                for flag in vetter["flags"]:
                    st.warning(flag)
            else:
                st.info("No physical safety flags triggered.")

        else:
            # Real-data path
            csv_p = Path(real_csv_path.strip())
            if not csv_p.is_file():
                st.info(
                    f"Notice: Time-series file not found — `{csv_p}`\n\n"
                    "Execute `python run_fits_photometry.py` to generate "
                    "`qatar1_real_lightcurve.csv` from the FITS science frames, then reload."
                )
            else:
                res = _run_real_data(csv_p)
                if res is not None:
                    det      = res["detection"]
                    vetter   = res["vetter"]
                    features = res["features"]

                    st.header("Observed Photometric Time-Series — Transit Analysis")
                    n_pts = len(det["clean_time"]) if det.get("clean_time") else 0
                    st.caption(f"Source file: `{csv_p.name}`  |  Photometric points: {n_pts}")

                    st.subheader("1. Observed Photometric Time Series")
                    fig = _lc_figure(det, title="Differential Photometry — Observed Data")
                    if fig:
                        st.pyplot(fig, use_container_width=True)
                    else:
                        st.write("No light curve data returned.")

                    st.subheader("2. BLS Detection Results")
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Transit Detected",
                              "Positive" if det.get("transit_detected") else "Negative")
                    c2.metric("Best-Fit Period",
                              f"{det.get('period', 0.0):.4f} d" if det.get("period") else "—")
                    c3.metric("Transit Depth",
                              f"{det.get('depth', 0.0):.4f}" if det.get("depth") else "—")
                    c4.metric("Detection SNR",
                              f"{det.get('snr', 0.0):.2f}" if det.get("snr") else "—")

                    if det.get("transit_detected"):
                        st.subheader("2b. Phase-Folded Time Series")
                        fig_fold = _lc_figure(
                            det, title="Phase-Folded at BLS Best-Fit Period", show_fold=True
                        )
                        if fig_fold:
                            st.pyplot(fig_fold, use_container_width=True)

                    st.subheader("3. Astrophysical Feature Vector (XGBoost Input)")
                    feat_rows = [
                        {"Feature": k,
                         "Value": f"{v:.6f}" if isinstance(v, float) else str(v),
                         "Description": FEATURE_DESCRIPTIONS.get(k, "")}
                        for k, v in features.items()
                    ]
                    st.dataframe(pd.DataFrame(feat_rows), use_container_width=True, hide_index=True)

                    st.subheader("4. XGBoost Vetter Output")
                    conf     = vetter.get("candidate_confidence", 0.0)
                    xverdict = "CONFIRMED PLANET CANDIDATE" if vetter.get("is_exoplanet_candidate") \
                               else "FALSE POSITIVE — REJECTED"
                    xcolor   = "#2ecc71" if vetter.get("is_exoplanet_candidate") else "#e74c3c"
                    st.markdown(
                        f"<div class='card'><b>Planet probability:</b> {conf:.2f}%<br>"
                        f"<b>Classification verdict:</b> <span style='color:{xcolor};font-weight:700'>"
                        f"{xverdict}</span></div>",
                        unsafe_allow_html=True,
                    )

                    st.subheader("5. Classification Result")
                    if vetter.get("is_exoplanet_candidate"):
                        st.success("ACCEPTED — Exoplanet Transit Candidate")
                    else:
                        st.error("REJECTED — False Positive / Noise Artifact")

                    st.subheader("6. Physical Safety Flag Report")
                    flags = vetter.get("flags", [])
                    if flags:
                        for flag in flags:
                            st.warning(flag)
                    else:
                        st.info("No physical safety flags triggered.")

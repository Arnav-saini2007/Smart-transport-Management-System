"""
utils.py — Shared helpers, data layer, AI layer, export utilities
"""
import io
import random
import time
from datetime import datetime

import pandas as pd
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ATTACK_TYPES = ["Sybil Attack", "GPS Spoofing", "DDoS", "Replay Attack", "Man in Middle"]
RISK_MAP = {"Sybil Attack": "High", "GPS Spoofing": "High", "DDoS": "High",
            "Replay Attack": "Medium", "Man in Middle": "Medium"}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94A3B8"),
    margin=dict(t=40, b=20, l=20, r=20),
)
PLOTLY_LEGEND = dict(
    bgcolor="rgba(15,22,41,.80)",
    bordercolor="#1E2D4A",
    borderwidth=1,
    font=dict(color="#94A3B8"),
)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LAYER
# ─────────────────────────────────────────────────────────────────────────────
def load_predictions() -> pd.DataFrame:
    try:
        df = pd.read_csv("data/predictions.csv")
        df["vehicle_id"] = df["vehicle_id"].astype(str)   # fix .str.contains crash
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
        df["speed"] = pd.to_numeric(df["speed"], errors="coerce")
        return df
    except FileNotFoundError:
        st.error("data/predictions.csv not found.")
        return pd.DataFrame()


def simulate_live_data(base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate live monitoring: apply small random deltas every ~10 s.
    Also occasionally promotes a Normal vehicle to Attack.
    """
    if base_df.empty:
        return base_df

    seed = int(time.time() / 10)   # seed changes every 10 s → smooth simulation
    rng = random.Random(seed)

    df = base_df.copy()
    df["speed"]      = df["speed"].apply(lambda s: max(0, int(s) + rng.randint(-12, 12)))
    df["confidence"] = df["confidence"].apply(
        lambda c: round(min(99.9, max(50.0, float(c) + rng.uniform(-1.5, 1.5))), 1)
    )

    # ~8 % chance each Normal vehicle becomes a transient attack
    for i in df.index:
        if df.at[i, "status"] == "Normal" and rng.random() < 0.08:
            atype = rng.choice(ATTACK_TYPES)
            df.at[i, "status"]      = "Attack"
            df.at[i, "attack_type"] = atype
            df.at[i, "risk_level"]  = RISK_MAP.get(atype, "Medium")
            df.at[i, "confidence"]  = round(rng.uniform(82, 96), 1)

    df["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return df


def get_display_data(base_df: pd.DataFrame) -> pd.DataFrame:
    return base_df


# ─────────────────────────────────────────────────────────────────────────────
# GEMMA AI PLACEHOLDER
# ─────────────────────────────────────────────────────────────────────────────
def generate_gemma_explanation(vehicle_data: dict) -> dict:
    # If real explanation already exists in the row (from your pipeline), use it directly
    real_explanation = vehicle_data.get("explanation", "")
    status = vehicle_data.get("status", "Normal")

    if real_explanation and status == "Attack":
        return {
            "anomaly":        "Anomalous V2X message pattern detected",
            "explanation":    real_explanation,   # ← your actual Gemma output
            "risk_score":     int(float(vehicle_data.get("confidence", 80))),
            "recommendation": "Flag vehicle for RSU-level monitoring and verification.",
        }

    if status == "Normal":
        vid   = vehicle_data.get("vehicle_id", "Unknown")
        speed = vehicle_data.get("speed", 0)
        conf  = float(vehicle_data.get("confidence", 0))
        return {
            "anomaly":        "No anomalies detected",
            "explanation":    f"Vehicle {vid} operating within expected parameters. Confidence: {conf:.1f}%.",
            "risk_score":     max(100 - int(conf), 2),
            "recommendation": "Continue normal monitoring.",
        }

    # fallback only if explanation missing for some reason
    return {
        "anomaly": "Unclassified anomaly",
        "explanation": "Explanation unavailable.",
        "risk_score": int(float(vehicle_data.get("confidence", 50))),
        "recommendation": "Manual review recommended.",
    }

# ─────────────────────────────────────────────────────────────────────────────
# EXPORT UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def generate_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def generate_html_report(df: pd.DataFrame) -> str:
    total   = len(df)
    attacks = int((df["status"] == "Attack").sum())
    normals = int((df["status"] == "Normal").sum())
    accuracy = f"{(normals / total * 100):.1f}" if total else "0"
    ts_now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    attack_rows = ""
    for _, r in df[df["status"] == "Attack"].iterrows():
        ts = r["timestamp"]
        ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        attack_rows += (
            f"<tr><td>{r['vehicle_id']}</td><td>{r['speed']} km/h</td>"
            f"<td>{r.get('attack_type','—')}</td>"
            f"<td>{float(r['confidence']):.1f}%</td>"
            f"<td>{r.get('risk_level','—')}</td><td>{ts_str}</td></tr>\n"
        )
    if not attack_rows:
        attack_rows = "<tr><td colspan='6' style='text-align:center;color:#94A3B8'>No attacks detected</td></tr>"

    all_rows = ""
    for _, r in df.iterrows():
        ts = r["timestamp"]
        ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        color = "#F87171" if r["status"] == "Attack" else "#34D399"
        all_rows += (
            f"<tr><td>{r['vehicle_id']}</td><td>{r['speed']} km/h</td>"
            f"<td style='color:{color};font-weight:600'>{r['status']}</td>"
            f"<td>{float(r['confidence']):.1f}%</td>"
            f"<td>{r.get('attack_type','None') or 'None'}</td>"
            f"<td>{r.get('risk_level','Low')}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ITMS Security Report — {ts_now}</title>
<style>
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0B1020;color:#F1F5F9;margin:0;padding:40px}}
  h1{{font-size:28px;font-weight:900;background:linear-gradient(90deg,#3B82F6,#8B5CF6);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px}}
  .sub{{color:#64748B;font-size:13px;margin-bottom:40px}}
  .cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:40px}}
  .card{{background:#0F1629;border:1px solid #1E2D4A;border-radius:16px;padding:20px}}
  .card-label{{font-size:11px;color:#64748B;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}}
  .card-val{{font-size:32px;font-weight:800}}
  .card-val.blue{{color:#60A5FA}} .card-val.red{{color:#F87171}}
  .card-val.green{{color:#34D399}} .card-val.purp{{color:#A78BFA}}
  h2{{font-size:16px;font-weight:700;color:#94A3B8;margin:32px 0 12px;
      border-bottom:1px solid #1E2D4A;padding-bottom:8px}}
  table{{width:100%;border-collapse:collapse;font-size:13px}}
  th{{text-align:left;padding:10px 12px;background:#111827;color:#64748B;
      font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.8px}}
  td{{padding:10px 12px;border-bottom:1px solid #1A2640;color:#CBD5E1}}
  tr:hover td{{background:rgba(59,130,246,.04)}}
  .footer{{margin-top:48px;font-size:12px;color:#374151;text-align:center}}
</style>
</head>
<body>
<h1>Intelligent Transport Management System</h1>
<div class="sub">VANET Security Report &nbsp;·&nbsp; Generated: {ts_now}</div>
<div class="cards">
  <div class="card"><div class="card-label">Total Vehicles</div><div class="card-val blue">{total}</div></div>
  <div class="card"><div class="card-label">Attacks Detected</div><div class="card-val red">{attacks}</div></div>
  <div class="card"><div class="card-label">Normal Vehicles</div><div class="card-val green">{normals}</div></div>
  <div class="card"><div class="card-label">ML Accuracy</div><div class="card-val purp">{accuracy}%</div></div>
</div>
<h2>Attack Events</h2>
<table>
  <thead><tr><th>Vehicle ID</th><th>Speed</th><th>Attack Type</th><th>Confidence</th><th>Risk Level</th><th>Time</th></tr></thead>
  <tbody>{attack_rows}</tbody>
</table>
<h2>Full Vehicle Log</h2>
<table>
  <thead><tr><th>Vehicle ID</th><th>Speed</th><th>Status</th><th>Confidence</th><th>Attack Type</th><th>Risk Level</th></tr></thead>
  <tbody>{all_rows}</tbody>
</table>
<div class="footer">ITMS Dashboard &nbsp;·&nbsp; AI-Powered VANET Security &nbsp;·&nbsp; v2.1.0</div>
</body>
</html>"""
    return html


def generate_pdf_bytes(df: pd.DataFrame) -> bytes:
    """Generate a PDF report using fpdf2."""
    try:
        from fpdf import FPDF
    except ImportError:
        # Fallback: return HTML as bytes if fpdf2 unavailable
        return generate_html_report(df).encode("utf-8")

    total   = len(df)
    attacks = int((df["status"] == "Attack").sum())
    normals = int((df["status"] == "Normal").sum())
    accuracy = f"{(normals / total * 100):.1f}%" if total else "0%"
    ts_now  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ── Header ──────────────────────────────────────────────────────────
    pdf.set_fill_color(9, 14, 26)
    pdf.rect(0, 0, 210, 297, "F")

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(96, 165, 250)
    pdf.cell(0, 14, "ITMS Security Report", ln=True, align="C")

    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(0, 6, f"AI-Powered VANET Security Dashboard   |   {ts_now}", ln=True, align="C")
    pdf.ln(8)

    # ── Summary boxes ───────────────────────────────────────────────────
    def box(x, y, label, val, r, g, b):
        pdf.set_xy(x, y)
        pdf.set_draw_color(30, 45, 74)
        pdf.set_fill_color(15, 22, 41)
        pdf.rect(x, y, 42, 22, "FD")
        pdf.set_xy(x + 2, y + 2)
        pdf.set_font("Helvetica", "", 7)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(38, 4, label.upper(), ln=True)
        pdf.set_xy(x + 2, y + 7)
        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(r, g, b)
        pdf.cell(38, 10, str(val))

    box(14,  46, "Total Vehicles",   total,    96, 165, 250)
    box(58,  46, "Attacks Detected", attacks,  248, 113, 113)
    box(102, 46, "Normal Vehicles",  normals,  52, 211, 153)
    box(146, 46, "ML Accuracy",      accuracy, 167, 139, 250)
    pdf.ln(32)
    pdf.set_xy(14, 74)

    # ── Attack table ────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 8, "Attack Events", ln=True)
    pdf.set_draw_color(26, 38, 64)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(2)

    headers = ["Vehicle ID", "Speed", "Attack Type", "Confidence", "Risk", "Time"]
    widths  = [30, 25, 45, 30, 25, 27]
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(100, 116, 139)
    for h, w in zip(headers, widths):
        pdf.cell(w, 7, h, border=0)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)

    atk_df = df[df["status"] == "Attack"]
    if atk_df.empty:
        pdf.set_text_color(148, 163, 184)
        pdf.cell(0, 7, "No attack events recorded.", ln=True)
    else:
        for _, r in atk_df.iterrows():
            pdf.set_text_color(248, 113, 113)
            ts = r["timestamp"]
            ts_s = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
            row_vals = [
                str(r["vehicle_id"]),
                f"{r['speed']} km/h",
                str(r.get("attack_type", "—")),
                f"{float(r['confidence']):.1f}%",
                str(r.get("risk_level", "—")),
                ts_s,
            ]
            for v, w in zip(row_vals, widths):
                pdf.cell(w, 6, v[:18], border=0)
            pdf.ln()

    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 8, "Full Vehicle Log", ln=True)
    pdf.set_draw_color(26, 38, 64)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(2)

    all_hdrs = ["Vehicle ID", "Speed", "Status", "Confidence", "Attack Type", "Risk"]
    all_wds  = [30, 25, 25, 30, 50, 22]
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(100, 116, 139)
    for h, w in zip(all_hdrs, all_wds):
        pdf.cell(w, 7, h, border=0)
    pdf.ln()
    pdf.set_font("Helvetica", "", 8)
    for _, r in df.iterrows():
        if r["status"] == "Attack":
            pdf.set_text_color(248, 113, 113)
        else:
            pdf.set_text_color(52, 211, 153)
        row_vals = [
            str(r["vehicle_id"]),
            f"{r['speed']} km/h",
            str(r["status"]),
            f"{float(r['confidence']):.1f}%",
            str(r.get("attack_type", "None") or "None"),
            str(r.get("risk_level", "Low")),
        ]
        for v, w in zip(row_vals, all_wds):
            pdf.cell(w, 6, v[:24], border=0)
        pdf.ln()

    pdf.ln(10)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(0, 5, "ITMS Dashboard  |  AI-Powered VANET Security  |  v2.1.0", align="C")

    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_SETTINGS = {
    "theme":         "Dark",
    "animations":    True,
    "grid_bg":       True,
    "refresh_interval": 30,
    "language":      "English",
    "live_mode":     False,
    "notifications": True,
}

def init_session_state():
    """Initialise all session state keys on first run."""
    for k, v in DEFAULT_SETTINGS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Dashboard"
    if "ignored_alerts" not in st.session_state:
        st.session_state["ignored_alerts"] = set()
    if "investigated_vid" not in st.session_state:
        st.session_state["investigated_vid"] = None


def navigate_to(page: str):
    """Change page via session state and rerun."""
    st.session_state["current_page"] = page
    st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPERS  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────
def speed_icon(speed):
    if speed > 120:
        return "🟠"
    elif speed > 80:
        return "🟡"
    return "🟢"


def severity_class(risk_level):
    return {"High": "high", "Medium": "medium", "Low": "low"}.get(risk_level, "low")


def confidence_bar_html(value):
    return (
        f'<div class="conf-bar">'
        f'  <div class="conf-bar-track">'
        f'    <div class="conf-bar-fill" style="width:{value}%"></div>'
        f'  </div>'
        f'  <span class="conf-bar-val">{value:.1f}%</span>'
        f'</div>'
    )


def metric_card_html(icon, label, value, color, trend="", corner=""):
    return (
        f'<div class="metric-card {color} fade-in">'
        f'  <div class="metric-corner {color}">{corner}</div>'
        f'  <span class="metric-icon">{icon}</span>'
        f'  <div class="metric-label">{label}</div>'
        f'  <div class="metric-value {color}">{value}</div>'
        f'  <div class="metric-trend">{trend}</div>'
        f'</div>'
    )


def gemma_panel_html(vid, anomaly, explanation, risk_score, recommendation):
    if risk_score > 75:
        risk_cls = "high"
    elif risk_score > 40:
        risk_cls = "medium"
    else:
        risk_cls = "low"
    return f"""
<div class="gemma-panel fade-in">
  <div class="gemma-title">
    <div class="gemma-title-icon">&#129302;</div>
    Gemma AI Analysis &mdash; {vid}
  </div>
  <div class="chat-bubble-ai" style="margin-top:24px;">
    {explanation}
  </div>
  <div class="gemma-field">
    <div class="gemma-field-label">Detected Anomaly</div>
    <div class="gemma-field-value">&#9888; {anomaly}</div>
  </div>
  <div class="gemma-field">
    <div class="gemma-field-label">Risk Score</div>
    <div class="gemma-field-value">
      {risk_score} / 100
      <div class="risk-bar">
        <div class="risk-fill {risk_cls}" style="width:{risk_score}%"></div>
      </div>
    </div>
  </div>
  <div class="gemma-field">
    <div class="gemma-field-label">Recommended Action</div>
    <div class="gemma-field-value">&#10003; {recommendation}</div>
  </div>
</div>"""


def alert_card_html(row, explanation):
    sev       = severity_class(row.get("risk_level", "Low"))
    vid       = row.get("vehicle_id", "")
    atype     = row.get("attack_type", "Unknown")
    conf      = float(row.get("confidence", 0))
    ts        = row.get("timestamp", "")
    spd       = row.get("speed", 0)
    sev_label = {"high": "&#128308; High", "medium": "&#128993; Medium",
                 "low": "&#128994; Low"}.get(sev, "")
    short_exp = explanation[:180] + ("..." if len(explanation) > 180 else "")
    return f"""
<div class="alert-card {sev} fade-in">
  <div class="alert-header">
    <div>
      <div class="alert-vid">&#128680; {vid}</div>
      <div class="alert-type">&#9889; {atype} &nbsp;&middot;&nbsp; {spd} km/h</div>
    </div>
    <div style="text-align:right;">
      <div class="badge badge-attack">{sev_label}</div>
      <div class="alert-time" style="margin-top:6px;">&#128336; {ts}</div>
    </div>
  </div>
  {confidence_bar_html(conf)}
  <div style="margin-top:14px;padding:12px;background:rgba(139,92,246,.08);
      border:1px solid rgba(139,92,246,.18);border-radius:10px;
      font-size:13px;color:#CBD5E1;line-height:1.6;">
    &#129302; <em>{short_exp}</em>
  </div>
</div>"""

"""
pages/alerts.py — Alerts center (fully wired: filter, sort, ignore, investigate)
"""
import streamlit as st
from utils import (
    generate_gemma_explanation, confidence_bar_html,
    severity_class, gemma_panel_html, navigate_to,
)


def render(df):
    st.markdown("""
<div class="hero fade-in" style="padding-bottom:16px;">
  <h1 style="font-size:clamp(28px,3vw,42px)!important;">&#128680; <span>Alerts</span> Center</h1>
  <div class="hero-desc">All detected anomalies and attack events across the VANET network.</div>
</div>""", unsafe_allow_html=True)

    # ── Ensure session state keys exist ───────────────────────────────
    if "ignored_alerts" not in st.session_state:
        st.session_state["ignored_alerts"] = set()
    if "investigated_vid" not in st.session_state:
        st.session_state["investigated_vid"] = None

    ignored  = st.session_state["ignored_alerts"]
    attack_df = df[
        (df["status"] == "Attack") &
        (~df["vehicle_id"].isin(ignored))
    ].copy()

    # ── Filters ───────────────────────────────────────────────────────
    f1, f2, f3 = st.columns([1, 1, 2])
    with f1:
        sev_filter = st.selectbox("Severity", ["All", "High", "Medium", "Low"], key="alert_sev")
    with f2:
        sort_opt = st.selectbox(
            "Sort", ["Latest First", "Oldest First", "Highest Confidence", "Highest Risk"],
            key="alert_sort",
        )
    with f3:
        search = st.text_input("🔍 Search Vehicle ID", key="alert_search")

    if sev_filter != "All":
        attack_df = attack_df[attack_df["risk_level"] == sev_filter]
    if search.strip():
        attack_df = attack_df[
            attack_df["vehicle_id"].str.contains(search.strip(), case=False, na=False)
        ]

    # Sort
    risk_order = {"High": 0, "Medium": 1, "Low": 2}
    if sort_opt == "Latest First":
        attack_df = attack_df.sort_values("timestamp", ascending=False)
    elif sort_opt == "Oldest First":
        attack_df = attack_df.sort_values("timestamp", ascending=True)
    elif sort_opt == "Highest Confidence":
        attack_df = attack_df.sort_values("confidence", ascending=False)
    elif sort_opt == "Highest Risk":
        attack_df["_risk_order"] = attack_df["risk_level"].map(risk_order).fillna(3)
        attack_df = attack_df.sort_values("_risk_order").drop(columns="_risk_order")

    # Stats row
    total_active = len(attack_df)
    high_count   = int((attack_df["risk_level"] == "High").sum())
    ign_count    = len(ignored)

    s1, s2, s3, _ = st.columns([1, 1, 1, 3])
    with s1:
        st.markdown(
            f"<div style='background:#0F1629;border:1px solid #1E2D4A;border-radius:12px;"
            f"padding:14px 18px;'><div style='font-size:11px;color:#64748B;text-transform:uppercase;"
            f"letter-spacing:1px'>Active Alerts</div>"
            f"<div style='font-size:28px;font-weight:800;color:#F87171'>{total_active}</div></div>",
            unsafe_allow_html=True,
        )
    with s2:
        st.markdown(
            f"<div style='background:#0F1629;border:1px solid #1E2D4A;border-radius:12px;"
            f"padding:14px 18px;'><div style='font-size:11px;color:#64748B;text-transform:uppercase;"
            f"letter-spacing:1px'>High Severity</div>"
            f"<div style='font-size:28px;font-weight:800;color:#EF4444'>{high_count}</div></div>",
            unsafe_allow_html=True,
        )
    with s3:
        st.markdown(
            f"<div style='background:#0F1629;border:1px solid #1E2D4A;border-radius:12px;"
            f"padding:14px 18px;'><div style='font-size:11px;color:#64748B;text-transform:uppercase;"
            f"letter-spacing:1px'>Ignored</div>"
            f"<div style='font-size:28px;font-weight:800;color:#A78BFA'>{ign_count}</div></div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    if attack_df.empty:
        st.success("✅ No active alerts match the current filters.")
        if ignored:
            if st.button("🔄 Restore All Ignored Alerts", key="restore_ignored"):
                st.session_state["ignored_alerts"] = set()
                st.success("All ignored alerts restored.")
                st.rerun()
        return

    # ── Alert Cards ───────────────────────────────────────────────────
    for _, row in attack_df.iterrows():
        ai   = generate_gemma_explanation(row.to_dict())
        sev  = severity_class(row.get("risk_level", "Low"))
        vid  = row["vehicle_id"]
        atype = row.get("attack_type", "Unknown")
        risk  = row.get("risk_level", "Low")
        ts    = row.get("timestamp", "")
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S") if hasattr(ts, "strftime") else str(ts)
        conf   = float(row.get("confidence", 0))
        spd    = row.get("speed", 0)

        sev_label = {"high": "🔴 High", "medium": "🟡 Medium", "low": "🔵 Low"}.get(sev, "")
        exp_label = f"🚨 {vid}  |  ⚡ {atype}  |  {sev_label}  |  {ts_str}"

        with st.expander(exp_label, expanded=False):
            col_l, col_r = st.columns([3, 2])

            with col_l:
                st.markdown(f"""
<div class="alert-card {sev}" style="margin-bottom:0;">
  <div class="alert-header">
    <div>
      <div class="alert-vid">&#128680; {vid}</div>
      <div class="alert-type">&#9889; {atype} &nbsp;&middot;&nbsp; {spd} km/h</div>
    </div>
    <div style="text-align:right;">
      <div class="badge badge-attack">{sev_label}</div>
      <div class="alert-time" style="margin-top:6px;">&#128336; {ts_str}</div>
    </div>
  </div>
  {confidence_bar_html(conf)}
</div>""", unsafe_allow_html=True)

                st.markdown("<div style='margin-top:14px;'></div>", unsafe_allow_html=True)

                ab1, ab2, ab3 = st.columns(3)

                with ab1:
                    # IGNORE — removes from active alert list
                    if st.button(f"🚫 Ignore", key=f"ignore_{vid}"):
                        st.session_state["ignored_alerts"].add(vid)
                        st.success(f"✅ Alert for **{vid}** has been ignored and removed from the active list.")
                        st.rerun()

                with ab2:
                    # INVESTIGATE — open detail panel, navigate to Vehicles
                    if st.button(f"🔍 Investigate", key=f"invest_{vid}"):
                        st.session_state["selected_vehicle"] = vid
                        st.session_state["investigated_vid"] = vid
                        st.info(f"🔍 Opening investigation for **{vid}** on the Vehicles page…")
                        navigate_to("Vehicles")

                with ab3:
                    if st.button(f"📋 Details", key=f"det_{vid}"):
                        st.session_state["investigated_vid"] = vid

            with col_r:
                st.markdown(
                    gemma_panel_html(vid, ai["anomaly"], ai["explanation"],
                                     ai["risk_score"], ai["recommendation"]),
                    unsafe_allow_html=True,
                )

    # Restore all ignored
    if ignored:
        st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
        if st.button("🔄 Restore All Ignored Alerts", key="restore_all"):
            st.session_state["ignored_alerts"] = set()
            st.success("All ignored alerts restored.")
            st.rerun()

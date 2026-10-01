"""
pages/dashboard.py — Dashboard page (fully wired)
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import (
    generate_gemma_explanation, confidence_bar_html,
    metric_card_html, gemma_panel_html, alert_card_html,
    PLOTLY_LAYOUT, navigate_to,
)


def render(df):
    # ── HERO ──────────────────────────────────────────────────────────
    st.markdown("""
<div class="hero fade-in">
  <div class="hero-badge">
    <div class="hero-badge-dot"></div>
    Live Monitoring Active
  </div>
  <h1>Intelligent <span>Transport</span><br>Management System</h1>
  <div class="hero-subtitle">AI-powered VANET Security Dashboard</div>
  <div class="hero-desc">
    Detect malicious vehicle behaviour in real-time using Machine Learning
    and Gemma AI &mdash; securing every node across the network.
  </div>
</div>
""", unsafe_allow_html=True)

    c1, c2, _, _ = st.columns([1, 1, 1, 3])
    with c1:
        if st.button("📊 View Reports", key="hero_reports"):
            navigate_to("Reports")          # ← wired: go to Reports page
    with c2:
        if st.button("🚗 Live Monitoring", key="hero_monitor"):
            navigate_to("Vehicles")         # ← wired: go to Vehicles page

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # ── METRIC CARDS (live from df) ────────────────────────────────────
    total   = len(df)
    attacks = int((df["status"] == "Attack").sum())
    normals = int((df["status"] == "Normal").sum())
    accuracy = f"{(normals / total * 100):.1f}%" if total else "0%"

    c1, c2, c3, c4 = st.columns(4)
    for col, icon, label, val, color, trend, corner in [
        (c1, "🚗", "Total Vehicles",   total,    "blue",  f"&#x25CF; {total} active",        "🚗"),
        (c2, "🚨", "Attacks Detected", attacks,  "red",   f"&#x25B2; {attacks} this cycle",  "🔴"),
        (c3, "✅", "Normal Vehicles",  normals,  "green", f"&#x25CF; {normals} monitored",   "✅"),
        (c4, "🎯", "ML Accuracy",      accuracy, "purp",  "&#x25CF; High precision",          "🎯"),
    ]:
        with col:
            st.markdown(metric_card_html(icon, label, val, color, trend, corner),
                        unsafe_allow_html=True)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # ── LIVE TABLE + CHARTS ───────────────────────────────────────────
    left_col, right_col = st.columns([3, 2])

    with left_col:
        st.markdown("""
<div class="section-header">
  <div class="section-title"><div class="dot"></div>Live Vehicle Status</div>
</div>""", unsafe_allow_html=True)

        search = st.text_input("🔍 Search vehicle", placeholder="e.g. V101", key="dash_search")
        filter_opt = st.selectbox("Filter by status", ["All", "Normal", "Attack"], key="dash_filter")

        display_df = df.copy()
        if search:
            display_df = display_df[
                display_df["vehicle_id"].str.contains(search, case=False, na=False)
            ]
        if filter_opt != "All":
            display_df = display_df[display_df["status"] == filter_opt]

        show_cols = ["vehicle_id", "speed", "status", "confidence", "timestamp", "attack_type"]
        available = [c for c in show_cols if c in display_df.columns]
        st.dataframe(
            display_df[available].rename(columns={
                "vehicle_id": "Vehicle ID", "speed": "Speed (km/h)",
                "status": "Status", "confidence": "Confidence (%)",
                "timestamp": "Timestamp", "attack_type": "Attack Type",
            }),
            width="stretch", hide_index=True, height=340,
        )

    with right_col:
        st.markdown("""
<div class="section-header">
  <div class="section-title"><div class="dot"></div>Attack Distribution</div>
</div>""", unsafe_allow_html=True)

        sc = df["status"].value_counts()
        fig_pie = px.pie(
            values=sc.values, names=sc.index, color=sc.index,
            color_discrete_map={"Normal": "#10B981", "Attack": "#EF4444"},
            hole=0.55,
        )
        fig_pie.update_traces(
            textposition="outside", textinfo="percent+label",
            marker=dict(line=dict(color="#0B1020", width=3)), pull=[0.04, 0.04],
        )
        pie_layout = dict(PLOTLY_LAYOUT)
        pie_layout.update(dict(
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                        xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#94A3B8")),
            annotations=[dict(text=f"<b>{total}</b><br>Vehicles", x=0.5, y=0.5,
                              font_size=18, showarrow=False, font=dict(color="#F1F5F9"))],
        ))
        fig_pie.update_layout(**pie_layout)
        st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

        fig_spd = px.histogram(
            df, x="speed", nbins=12,
            color_discrete_sequence=["#3B82F6"],
            labels={"speed": "Speed (km/h)"}, title="Speed Distribution",
        )
        fig_spd.update_traces(marker_line_width=0)
        fig_spd.update_layout(
            **PLOTLY_LAYOUT, bargap=0.1,
            xaxis=dict(showgrid=False, color="#64748B"),
            yaxis=dict(showgrid=True, gridcolor="#1A2640", color="#64748B"),
            title_font=dict(size=14, color="#94A3B8"),
        )
        st.plotly_chart(fig_spd, width="stretch", config={"displayModeBar": False})

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # ── RECENT ALERTS ─────────────────────────────────────────────────
    st.markdown("""
<div class="section-title fade-in" style="margin-bottom:16px;">
  <div class="dot"></div>Recent Alerts
</div>""", unsafe_allow_html=True)

    ignored = st.session_state.get("ignored_alerts", set())
    attack_df = (
        df[(df["status"] == "Attack") & (~df["vehicle_id"].isin(ignored))]
        .sort_values("timestamp", ascending=False)
        .head(4)
    )
    if attack_df.empty:
        st.info("✅ No active alerts.")
    else:
        for _, row in attack_df.iterrows():
            ai = generate_gemma_explanation(row.to_dict())
            st.markdown(alert_card_html(row.to_dict(), ai["explanation"]),
                        unsafe_allow_html=True)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # ── GEMMA AI PANEL ────────────────────────────────────────────────
    st.markdown("""
<div class="section-title fade-in" style="margin-bottom:16px;">
  <div class="dot" style="background:#8B5CF6;box-shadow:0 0 8px #8B5CF6;"></div>
  Gemma AI Analysis
</div>""", unsafe_allow_html=True)

    selected_vid = st.selectbox(
        "Select vehicle for Gemma AI analysis",
        df["vehicle_id"].tolist(), key="gemma_select_dash",
    )
    row_data = df[df["vehicle_id"] == selected_vid].iloc[0].to_dict()
    ai = generate_gemma_explanation(row_data)
    st.markdown(
        gemma_panel_html(selected_vid, ai["anomaly"], ai["explanation"],
                         ai["risk_score"], ai["recommendation"]),
        unsafe_allow_html=True,
    )

"""
pages/reports.py — Analytics & Reports (fully wired)
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from utils import PLOTLY_LAYOUT, generate_csv_bytes, generate_pdf_bytes, generate_html_report


def render(df):
    st.markdown("""
<div class="hero fade-in" style="padding-bottom:16px;">
  <h1 style="font-size:clamp(28px,3vw,42px)!important;">&#128202; Analytics <span>Reports</span></h1>
  <div class="hero-desc">Comprehensive analytics and exportable reports for the VANET security system.</div>
</div>""", unsafe_allow_html=True)

    # ensure timestamp is datetime
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    ts_now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ── Export Buttons ─────────────────────────────────────────────────
    ec1, ec2, ec3, _ = st.columns([1, 1, 1, 3])
    with ec1:
        if st.download_button("⬇ Download CSV", data=generate_csv_bytes(df),
                              file_name=f"itms_{ts_now}.csv", mime="text/csv", key="dl_csv"):
            st.toast("✅ CSV downloaded!", icon="📥")
    with ec2:
        if st.download_button("⬇ Export PDF", data=generate_pdf_bytes(df),
                              file_name=f"itms_{ts_now}.pdf", mime="application/pdf", key="dl_pdf"):
            st.toast("✅ PDF downloaded!", icon="📄")
    with ec3:
        if st.download_button("📤 Export HTML", data=generate_html_report(df).encode(),
                              file_name=f"itms_{ts_now}.html", mime="text/html", key="dl_html"):
            st.toast("✅ HTML exported!", icon="🌐")

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # ── Row 1 ───────────────────────────────────────────────────────────
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        # Attack trend — group by hour
        valid_ts = df.dropna(subset=["timestamp"])
        if len(valid_ts) > 1:
            t_min, t_max = valid_ts["timestamp"].min(), valid_ts["timestamp"].max()
            time_bins = pd.date_range(t_min, t_max, periods=8)
            labels = [t.strftime("%H:%M") for t in time_bins]
            attack_counts = []
            for i in range(len(time_bins)):
                lo = time_bins[i]
                hi = time_bins[min(i+1, len(time_bins)-1)]
                n = int(((valid_ts["status"] == "Attack") &
                         (valid_ts["timestamp"] >= lo) &
                         (valid_ts["timestamp"] <= hi)).sum())
                attack_counts.append(n)
        else:
            labels = ["00:00"]
            attack_counts = [0]

        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=labels, y=attack_counts, mode="lines+markers",
            line=dict(color="#EF4444", width=2.5),
            marker=dict(size=7, color="#F87171"),
            fill="tozeroy", fillcolor="rgba(239,68,68,.10)", name="Attacks",
        ))
        fig_trend.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Attack Trend Over Time", font=dict(size=14, color="#94A3B8")),
            xaxis=dict(showgrid=False, color="#64748B"),
            yaxis=dict(showgrid=True, gridcolor="#1A2640", color="#64748B", title="Count"),
            height=300,
        )
        st.plotly_chart(fig_trend, width="stretch", config={"displayModeBar": False})

    with r1c2:
        fig_spd = px.histogram(
            df, x="speed", color="status", nbins=15,
            color_discrete_map={"Normal": "#10B981", "Attack": "#EF4444"},
            barmode="overlay", opacity=0.75, labels={"speed": "Speed (km/h)"},
        )
        fig_spd.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Vehicle Speed Distribution", font=dict(size=14, color="#94A3B8")),
            xaxis=dict(showgrid=False, color="#64748B"),
            yaxis=dict(showgrid=True, gridcolor="#1A2640", color="#64748B"),
            bargap=0.05, height=300,
        )
        st.plotly_chart(fig_spd, width="stretch", config={"displayModeBar": False})

    # ── Row 2 ───────────────────────────────────────────────────────────
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        atk = df[df["status"] == "Attack"]["attack_type"].value_counts().reset_index()
        atk.columns = ["Attack Type", "Count"]
        if atk.empty:
            atk = pd.DataFrame({"Attack Type": ["None"], "Count": [0]})
        fig_bar = px.bar(atk, x="Count", y="Attack Type", orientation="h",
                         color="Count",
                         color_continuous_scale=[[0,"#3B82F6"],[0.5,"#8B5CF6"],[1.0,"#EF4444"]])
        fig_bar.update_traces(marker_line_width=0)
        fig_bar.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Attack Type Distribution", font=dict(size=14, color="#94A3B8")),
            xaxis=dict(showgrid=True, gridcolor="#1A2640", color="#64748B"),
            yaxis=dict(showgrid=False, color="#64748B"),
            coloraxis_showscale=False, height=300,
        )
        st.plotly_chart(fig_bar, width="stretch", config={"displayModeBar": False})

    with r2c2:
        fig_conf = px.histogram(
            df, x="confidence", color="status", nbins=15,
            color_discrete_map={"Normal": "#10B981", "Attack": "#EF4444"},
            barmode="overlay", opacity=0.8, labels={"confidence": "Confidence (%)"},
        )
        fig_conf.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Confidence Score Distribution", font=dict(size=14, color="#94A3B8")),
            xaxis=dict(showgrid=False, color="#64748B"),
            yaxis=dict(showgrid=True, gridcolor="#1A2640", color="#64748B"),
            bargap=0.05, height=300,
        )
        st.plotly_chart(fig_conf, width="stretch", config={"displayModeBar": False})

    # ── Row 3 ───────────────────────────────────────────────────────────
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        sc = df["status"].value_counts()
        total = len(df)
        fig_pie = px.pie(
            values=sc.values, names=sc.index, color=sc.index,
            color_discrete_map={"Normal": "#10B981", "Attack": "#EF4444"}, hole=0.55,
        )
        fig_pie.update_traces(
            textposition="outside", textinfo="percent+label",
            marker=dict(line=dict(color="#0B1020", width=3)), pull=[0.05, 0.05],
        )
        pie_l = dict(PLOTLY_LAYOUT)
        pie_l.update(dict(
            title=dict(text="Normal vs Attack", font=dict(size=14, color="#94A3B8")),
            showlegend=False,
            annotations=[dict(text=f"<b>{total}</b><br>Total", x=0.5, y=0.5,
                              font_size=16, showarrow=False, font=dict(color="#F1F5F9"))],
            height=320,
        ))
        fig_pie.update_layout(**pie_l)
        st.plotly_chart(fig_pie, width="stretch", config={"displayModeBar": False})

    with r3c2:
        rc = df["risk_level"].value_counts().reset_index()
        rc.columns = ["Risk Level", "Count"]
        fig_risk = px.bar(rc, x="Risk Level", y="Count", color="Risk Level",
                          color_discrete_map={"High":"#EF4444","Medium":"#F59E0B","Low":"#10B981"})
        fig_risk.update_traces(marker_line_width=0)
        fig_risk.update_layout(
            **PLOTLY_LAYOUT,
            title=dict(text="Risk Level Distribution", font=dict(size=14, color="#94A3B8")),
            xaxis=dict(showgrid=False, color="#64748B"),
            yaxis=dict(showgrid=True, gridcolor="#1A2640", color="#64748B"),
            showlegend=False, height=320,
        )
        st.plotly_chart(fig_risk, width="stretch", config={"displayModeBar": False})

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)
    st.markdown("""<div class="section-title" style="margin-bottom:16px;">
  <div class="dot"></div>Full Data Table</div>""", unsafe_allow_html=True)
    st.dataframe(df, width="stretch", hide_index=True)

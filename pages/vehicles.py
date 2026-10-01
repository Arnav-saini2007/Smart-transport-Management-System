"""
pages/vehicles.py — Vehicle monitor
"""
import streamlit as st
from utils import (
    speed_icon, confidence_bar_html,
    metric_card_html,
)



def render(df):
    st.markdown("""
<div class="hero fade-in" style="padding-bottom:16px;">
  <h1 style="font-size:clamp(28px,3vw,42px)!important;">&#x1F699; Vehicle <span>Monitor</span></h1>
  <div class="hero-desc">Detailed real-time monitoring for every node in the VANET network.</div>
</div>""", unsafe_allow_html=True)

    total   = len(df)
    normals = int((df["status"] == "Normal").sum())
    attacks = int((df["status"] == "Attack").sum())

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(metric_card_html("🚗","Total Vehicles",total,"blue","● Active","🚗"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html("✅","Normal",normals,"green","● Safe","✅"), unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html("🚨","Threats",attacks,"red","▲ Detected","🔴"), unsafe_allow_html=True)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # ── Search + Filter ────────────────────────────────────────────────
    col_s, col_f = st.columns([2, 1])
    with col_s:
        search = st.text_input("🔍 Search Vehicle ID", key="veh_search", placeholder="Type a vehicle ID…")
    with col_f:
        status_filter = st.selectbox("Status", ["All", "Normal", "Attack"], key="veh_filter")

    filtered = df.copy()
    if search.strip():
        filtered = filtered[filtered["vehicle_id"].str.contains(search.strip(), case=False, na=False)]
    if status_filter != "All":
        filtered = filtered[filtered["status"] == status_filter]

    st.markdown(
        f"<p style='color:#64748B;font-size:13px;margin-bottom:16px;'>"
        f"Showing <b style='color:#60A5FA'>{len(filtered)}</b> / {len(df)} vehicles</p>",
        unsafe_allow_html=True,
    )

    # ── VEHICLE CARDS — flat loop, no nested rerun ────────────────────
    cols_per_row = 3
    rows = [filtered.iloc[i:i+cols_per_row] for i in range(0, len(filtered), cols_per_row)]

    for row_group in rows:
        card_cols = st.columns(cols_per_row)
        for col_idx, (_, row) in enumerate(row_group.iterrows()):
            vid        = row["vehicle_id"]
            is_attack  = row["status"] == "Attack"
            card_cls   = "attack-card" if is_attack else ""
            status_badge = (
                '<span class="badge badge-attack">&#128308; Attack</span>'
                if is_attack else
                '<span class="badge badge-normal">&#128994; Normal</span>'
            )
            spd_ic = speed_icon(float(row.get("speed", 0)))
            ts     = row.get("timestamp", "")
            ts_str = ts.strftime("%H:%M:%S") if hasattr(ts, "strftime") else str(ts)
            conf   = float(row.get("confidence", 0))

            with card_cols[col_idx]:
                st.markdown(f"""
<div class="vehicle-card {card_cls}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;">
    <div class="vehicle-id">{vid}</div>
    {status_badge}
  </div>
  <div class="vehicle-meta">
    <div>
      <div class="vehicle-meta-item">Speed</div>
      <div class="vehicle-meta-val">{spd_ic} {row.get('speed','?')} km/h</div>
    </div>
    <div>
      <div class="vehicle-meta-item">Confidence</div>
      <div class="vehicle-meta-val">{conf:.1f}%</div>
    </div>
    <div>
      <div class="vehicle-meta-item">Risk</div>
      <div class="vehicle-meta-val">{row.get('risk_level','Low')}</div>
    </div>
  </div>
  <div style="margin-top:12px;">{confidence_bar_html(conf)}</div>
  <div style="margin-top:10px;font-size:11px;color:#64748B;">Last seen: {ts_str}</div>
</div>""", unsafe_allow_html=True)

"""
app.py — ITMS Main Entry Point
"""
import streamlit as st

st.set_page_config(
    page_title="ITMS · AI Transport Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

from styles import inject_css
from utils import load_predictions, init_session_state, get_display_data

import pages.dashboard as pg_dashboard
import pages.vehicles  as pg_vehicles
import pages.alerts    as pg_alerts
import pages.reports   as pg_reports
import pages.settings  as pg_settings

PAGES = ["Dashboard", "Vehicles", "Alerts", "Reports", "Settings"]
PAGE_ICONS = {"Dashboard":"📊","Vehicles":"🚗","Alerts":"🚨","Reports":"📈","Settings":"⚙️"}


def render_sidebar(df):
    attack_count = int((df["status"] == "Attack").sum()) if not df.empty else 0
    ignored_count = len(st.session_state.get("ignored_alerts", set()))

    with st.sidebar:
        st.markdown("""
<div class="sidebar-logo">
  <h2>&#x1F697; ITMS</h2>
  <p>VANET Security &middot; AI-Powered</p>
</div>""", unsafe_allow_html=True)

        # ── Navigation buttons (session_state controlled) ────────────
        current = st.session_state.get("current_page", "Dashboard")
        for p in PAGES:
            active_style = (
                "background:rgba(59,130,246,.18);color:#60A5FA;font-weight:700;"
                "border:1px solid rgba(59,130,246,.40);"
                if p == current else
                "background:transparent;color:#94A3B8;border:1px solid transparent;"
            )
            # Use markdown button trick: real st.button for click, styled with CSS
            if st.button(
                f"{PAGE_ICONS[p]}  {p}",
                key=f"nav_{p}",
                use_container_width=True,
            ):
                st.session_state["current_page"] = p
                st.rerun()

        # ── Quick stats ───────────────────────────────────────────────
        st.markdown(f"""
<div style="margin:20px 4px 0;padding-top:16px;border-top:1px solid #1A2640;">
  <div style="font-size:11px;color:#64748B;font-weight:600;text-transform:uppercase;
              letter-spacing:1px;margin-bottom:10px;">Quick Stats</div>
  <div style="display:flex;flex-direction:column;gap:7px;">
    <div style="display:flex;justify-content:space-between;font-size:13px;">
      <span style="color:#94A3B8;">Active Vehicles</span>
      <span style="color:#60A5FA;font-weight:700;">{len(df)}</span>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:13px;">
      <span style="color:#94A3B8;">Threats</span>
      <span style="color:#F87171;font-weight:700;">{attack_count}</span>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:13px;">
      <span style="color:#94A3B8;">Ignored Alerts</span>
      <span style="color:#A78BFA;font-weight:700;">{ignored_count}</span>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

        live = st.toggle("Live Monitoring", value=st.session_state.get("live_mode", True), key="sidebar_live")
        st.session_state["live_mode"] = live

        st.markdown("""
<div style="margin-top:20px;padding-top:12px;border-top:1px solid #1A2640;">
  <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#64748B;">
    <div style="width:8px;height:8px;border-radius:50%;background:#10B981;
                box-shadow:0 0 6px #10B981;display:inline-block;"></div>
    System Online &middot; v2.1.0
  </div>
</div>""", unsafe_allow_html=True)


def main():
    inject_css()
    init_session_state()

    base_df = load_predictions()
    df      = get_display_data(base_df)

    render_sidebar(df)

    page = st.session_state.get("current_page", "Dashboard")

    if df.empty and page != "Settings":
        st.error("No data found. Please check data/predictions.csv")
        return

    if   page == "Dashboard": pg_dashboard.render(df)
    elif page == "Vehicles":  pg_vehicles.render(df)
    elif page == "Alerts":    pg_alerts.render(df)
    elif page == "Reports":   pg_reports.render(df)
    elif page == "Settings":  pg_settings.render()

    # Auto-refresh on dashboard
    if page == "Dashboard" and st.session_state.get("live_mode", True):
        interval = st.session_state.get("refresh_interval", 30)
        st.markdown(
            f"<script>setTimeout(()=>window.location.reload(),{interval*1000});</script>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()

"""
pages/settings.py — Settings page (fully wired with session_state persistence)
"""
import streamlit as st
from utils import DEFAULT_SETTINGS, navigate_to


# Inject theme-specific overrides on top of the base dark theme
THEME_CSS = {
    "Dark": "",   # default — no override needed
    "Cyber": """
<style>
:root{
  --blue:#00FFD1;--purple:#FF00A8;--cyan:#00CFFF;
  --card-bg:#050A10;--card-bdr:#002030;
  --bg1:#030812;
}
.hero h1 span{background:linear-gradient(90deg,#00FFD1,#FF00A8)!important;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent!important}
.stApp{background-color:#030812!important;}
</style>""",
    "Purple": """
<style>
:root{
  --blue:#A855F7;--purple:#C084FC;--cyan:#E879F9;
  --card-bg:#12082A;--card-bdr:#2D1554;
  --bg1:#0C0518;
}
.stApp{background-color:#0C0518!important;
  background-image:
    linear-gradient(rgba(255,255,255,.02) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,255,255,.02) 1px,transparent 1px),
    radial-gradient(ellipse 80% 50% at 50% -10%,rgba(168,85,247,.25),transparent),
    radial-gradient(ellipse 60% 40% at 90% 100%,rgba(192,132,252,.15),transparent)!important;
}
</style>""",
}


def render():
    st.markdown("""
<div class="hero fade-in" style="padding-bottom:16px;">
  <h1 style="font-size:clamp(28px,3vw,42px)!important;">&#9881;&#65039; <span>Settings</span></h1>
  <div class="hero-desc">Customize ITMS dashboard behaviour, appearance, and data sources.</div>
</div>""", unsafe_allow_html=True)

    # Inject the active theme CSS override
    active_theme = st.session_state.get("theme", "Dark")
    if THEME_CSS.get(active_theme):
        st.markdown(THEME_CSS[active_theme], unsafe_allow_html=True)

    # ── Appearance ──────────────────────────────────────────────────
    st.markdown('<div class="settings-section fade-in">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">🎨 Appearance</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        theme = st.selectbox(
            "Theme",
            ["Dark", "Cyber", "Purple"],
            index=["Dark","Cyber","Purple"].index(st.session_state.get("theme","Dark")),
            key="s_theme_sel",
        )
        animations = st.toggle(
            "Enable Animations",
            value=st.session_state.get("animations", True),
            key="s_anim",
        )
    with c2:
        grid_bg = st.toggle(
            "Background Grid",
            value=st.session_state.get("grid_bg", True),
            key="s_grid",
        )
        language = st.selectbox(
            "Language",
            ["English", "Hindi", "Spanish", "French", "German"],
            index=["English","Hindi","Spanish","French","German"].index(
                st.session_state.get("language","English")
            ),
            key="s_lang",
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Live Monitoring ─────────────────────────────────────────────
    st.markdown('<div class="settings-section fade-in-1">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">🔄 Live Monitoring</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        live_mode = st.toggle(
            "Live Data Simulation",
            value=st.session_state.get("live_mode", True),
            key="s_live",
        )
        refresh_interval = st.slider(
            "Auto-Refresh Interval (seconds)",
            min_value=5, max_value=120, step=5,
            value=st.session_state.get("refresh_interval", 30),
            key="s_interval",
        )
        st.caption(f"Dashboard auto-refreshes every **{refresh_interval}s** on the Dashboard page.")
    with c4:
        data_src = st.selectbox(
            "Data Source",
            ["CSV File (data/predictions.csv)", "REST API (placeholder)", "ML Inference (placeholder)"],
            key="s_datasource",
        )
        st.caption("Switch sources without changing any UI code — only `load_predictions()` changes.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Notifications ───────────────────────────────────────────────
    st.markdown('<div class="settings-section fade-in-2">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">🔔 Notifications</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        notifications = st.toggle(
            "Enable Notifications",
            value=st.session_state.get("notifications", True),
            key="s_notif",
        )
        st.toggle("Sound Alerts", value=False, key="s_sound")
    with c6:
        threshold = st.slider(
            "Alert Confidence Threshold (%)",
            min_value=50, max_value=99, value=85, step=1,
            key="s_threshold",
        )
        st.caption(f"Only show alerts when ML confidence ≥ **{threshold}%**.")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Gemma AI ────────────────────────────────────────────────────
    st.markdown('<div class="settings-section fade-in-3">', unsafe_allow_html=True)
    st.markdown('<div class="settings-title">🤖 Gemma AI Configuration</div>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)
    with c7:
        st.selectbox(
            "Gemma Mode",
            ["Local Placeholder (Demo)", "Gemma 2B API", "Gemma 7B API"],
            key="s_gemma_mode",
        )
        st.text_input("Gemma API Key", type="password",
                      placeholder="sk-...", key="s_api_key")
    with c8:
        temp = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05, key="s_temp")
        max_tok = st.slider("Max Tokens", 128, 2048, 512, 64, key="s_tokens")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # ── Save / Reset ─────────────────────────────────────────────────
    save_col, reset_col, _ = st.columns([1, 1, 4])

    with save_col:
        if st.button("💾 Save Settings", key="save_settings"):
            # Persist all settings into session_state
            st.session_state["theme"]            = theme
            st.session_state["animations"]       = animations
            st.session_state["grid_bg"]          = grid_bg
            st.session_state["language"]         = language
            st.session_state["live_mode"]        = live_mode
            st.session_state["refresh_interval"] = refresh_interval
            st.session_state["notifications"]    = notifications

            # Apply animation/grid CSS immediately
            if not animations:
                st.markdown("""
<style>
.fade-in,.fade-in-1,.fade-in-2,.fade-in-3,.fade-in-4{animation:none!important}
.metric-card:hover{transform:none!important}
.vehicle-card:hover{transform:none!important}
.alert-card:hover{transform:none!important}
</style>""", unsafe_allow_html=True)

            if not grid_bg:
                st.markdown("""
<style>
.stApp{background-image:
  radial-gradient(ellipse 80% 50% at 50% -10%,rgba(139,92,246,.20),transparent),
  radial-gradient(ellipse 60% 40% at 90% 100%,rgba(59,130,246,.15),transparent)!important;
}
.stApp::before,.stApp::after{display:none!important}
</style>""", unsafe_allow_html=True)

            st.success("✅ Settings saved successfully! Changes applied.")
            st.toast("Settings saved!", icon="💾")

    with reset_col:
        if st.button("🔄 Reset to Defaults", key="reset_settings"):
            for k, v in DEFAULT_SETTINGS.items():
                st.session_state[k] = v
            st.warning("⚠️ All settings have been reset to defaults.")
            st.toast("Settings reset to defaults.", icon="🔄")
            st.rerun()

    # ── Current Values Preview ───────────────────────────────────────
    with st.expander("👁 Current Session State (Debug)", expanded=False):
        preview = {k: st.session_state.get(k) for k in DEFAULT_SETTINGS}
        preview["current_page"]   = st.session_state.get("current_page")
        preview["ignored_alerts"] = list(st.session_state.get("ignored_alerts", set()))
        st.json(preview)

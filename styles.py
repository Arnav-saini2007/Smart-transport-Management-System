"""
styles.py — All custom CSS for ITMS Dashboard
"""
import streamlit as st


def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg0:#090E1A; --bg1:#0B1020; --bg2:#111827;
    --card-bg:#0F1629; --card-bdr:#1E2D4A;
    --blue:#3B82F6; --purple:#8B5CF6; --cyan:#06B6D4;
    --green:#10B981; --red:#EF4444; --orange:#F59E0B;
    --text-1:#F1F5F9; --text-2:#94A3B8; --text-3:#64748B;
    --radius:18px; --radius-sm:10px;
    --glow-blue:0 0 24px rgba(59,130,246,.35);
    --glow-purp:0 0 24px rgba(139,92,246,.35);
    --glow-red:0 0 24px rgba(239,68,68,.35);
    --glow-green:0 0 24px rgba(16,185,129,.25);
    --transition:all .25s cubic-bezier(.4,0,.2,1);
}

*,*::before,*::after{box-sizing:border-box}

html,body,.stApp{
    font-family:'Inter',system-ui,-apple-system,sans-serif!important;
    color:var(--text-1)!important;
}

/* BACKGROUND */
.stApp{
    background-color:var(--bg1)!important;
    background-image:
        linear-gradient(rgba(255,255,255,.025) 1px,transparent 1px),
        linear-gradient(90deg,rgba(255,255,255,.025) 1px,transparent 1px),
        radial-gradient(ellipse 80% 50% at 50% -10%,rgba(139,92,246,.20),transparent),
        radial-gradient(ellipse 60% 40% at 90% 100%,rgba(59,130,246,.15),transparent),
        radial-gradient(ellipse 40% 30% at 10% 80%,rgba(6,182,212,.10),transparent);
    background-size:50px 50px,50px 50px,auto,auto,auto;
    min-height:100vh;
}
.stApp::before{
    content:'';position:fixed;top:-200px;left:-200px;
    width:600px;height:600px;
    background:radial-gradient(circle,rgba(139,92,246,.12) 0%,transparent 70%);
    border-radius:50%;pointer-events:none;z-index:0;
    animation:orb1 12s ease-in-out infinite alternate;
}
.stApp::after{
    content:'';position:fixed;bottom:-200px;right:-200px;
    width:700px;height:700px;
    background:radial-gradient(circle,rgba(59,130,246,.10) 0%,transparent 70%);
    border-radius:50%;pointer-events:none;z-index:0;
    animation:orb2 15s ease-in-out infinite alternate;
}
@keyframes orb1{from{transform:translate(0,0) scale(1)}to{transform:translate(80px,60px) scale(1.15)}}
@keyframes orb2{from{transform:translate(0,0) scale(1)}to{transform:translate(-60px,-80px) scale(1.1)}}

/* SIDEBAR */
section[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0A1020 0%,#0D1428 100%)!important;
    border-right:1px solid #1A2640!important;
}
section[data-testid="stSidebar"]>div{padding-top:0!important}

.sidebar-logo{padding:28px 20px 24px;border-bottom:1px solid #1A2640;margin-bottom:12px}
.sidebar-logo h2{
    font-size:22px!important;font-weight:800!important;
    background:linear-gradient(90deg,var(--blue),var(--purple));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0!important;
}
.sidebar-logo p{font-size:11px;color:var(--text-3);margin:4px 0 0;text-transform:uppercase;letter-spacing:1.2px}

section[data-testid="stSidebar"] [data-testid="stRadio"] label{
    padding:10px 14px!important;border-radius:10px!important;margin:2px 0!important;
    display:block!important;font-size:14px!important;font-weight:500!important;
    color:var(--text-2)!important;transition:var(--transition)!important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover{
    background:rgba(59,130,246,.08)!important;color:var(--text-1)!important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked){
    background:rgba(59,130,246,.15)!important;color:#60A5FA!important;font-weight:600!important;
}

/* HERO */
.hero{padding:48px 0 32px;position:relative}
.hero-badge{
    display:inline-flex;align-items:center;gap:8px;
    background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.30);
    border-radius:999px;padding:6px 16px;font-size:12px;font-weight:600;
    color:#60A5FA;letter-spacing:.5px;margin-bottom:20px;text-transform:uppercase;
}
.hero-badge-dot{
    width:6px;height:6px;border-radius:50%;background:var(--blue);
    box-shadow:0 0 6px var(--blue);animation:pulse 2s infinite;
}
.hero h1{
    font-size:clamp(32px,4vw,56px)!important;font-weight:900!important;
    line-height:1.1!important;letter-spacing:-1.5px;
    background:linear-gradient(135deg,#ffffff 0%,#94A3B8 80%);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0 0 12px!important;
}
.hero h1 span{
    background:linear-gradient(90deg,var(--blue),var(--purple),var(--cyan));
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}
.hero-subtitle{font-size:18px;color:#60A5FA;font-weight:600;margin-bottom:10px}
.hero-desc{font-size:15px;color:var(--text-2);max-width:560px;line-height:1.7;margin-bottom:32px}

/* METRIC CARDS */
.metric-card{
    background:var(--card-bg);border:1px solid var(--card-bdr);border-radius:var(--radius);
    padding:24px;position:relative;overflow:hidden;transition:var(--transition);cursor:default;
}
.metric-card::before{
    content:'';position:absolute;inset:0;border-radius:var(--radius);opacity:0;transition:opacity .3s;
}
.metric-card.blue::before{background:radial-gradient(circle at top right,rgba(59,130,246,.10),transparent 60%)}
.metric-card.red::before{background:radial-gradient(circle at top right,rgba(239,68,68,.10),transparent 60%)}
.metric-card.green::before{background:radial-gradient(circle at top right,rgba(16,185,129,.10),transparent 60%)}
.metric-card.purp::before{background:radial-gradient(circle at top right,rgba(139,92,246,.10),transparent 60%)}
.metric-card:hover{transform:translateY(-4px)}
.metric-card.blue:hover{border-color:rgba(59,130,246,.50);box-shadow:var(--glow-blue)}
.metric-card.red:hover{border-color:rgba(239,68,68,.50);box-shadow:var(--glow-red)}
.metric-card.green:hover{border-color:rgba(16,185,129,.40);box-shadow:var(--glow-green)}
.metric-card.purp:hover{border-color:rgba(139,92,246,.50);box-shadow:var(--glow-purp)}
.metric-card:hover::before{opacity:1}

.metric-corner{
    position:absolute;top:16px;right:16px;width:36px;height:36px;border-radius:10px;
    display:flex;align-items:center;justify-content:center;font-size:16px;
}
.metric-corner.blue{background:rgba(59,130,246,.15)}
.metric-corner.red{background:rgba(239,68,68,.15)}
.metric-corner.green{background:rgba(16,185,129,.15)}
.metric-corner.purp{background:rgba(139,92,246,.15)}
.metric-icon{font-size:28px;margin-bottom:12px;display:block}
.metric-label{font-size:12px;font-weight:600;color:var(--text-3);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.metric-value{font-size:36px;font-weight:800;line-height:1;margin-bottom:8px;font-variant-numeric:tabular-nums}
.metric-value.blue{color:#60A5FA}
.metric-value.red{color:#F87171}
.metric-value.green{color:#34D399}
.metric-value.purp{color:#A78BFA}
.metric-trend{font-size:12px;color:var(--text-3);display:flex;align-items:center;gap:4px}

/* SECTION HEADERS */
.section-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.section-title{font-size:18px;font-weight:700;color:var(--text-1);display:flex;align-items:center;gap:10px}
.section-title .dot{
    width:8px;height:8px;border-radius:50%;background:var(--blue);
    box-shadow:0 0 8px var(--blue);animation:pulse 2s infinite;
}

/* GLASS BOX */
.glass-box{
    background:rgba(15,22,41,.70);backdrop-filter:blur(20px);
    -webkit-backdrop-filter:blur(20px);border:1px solid var(--card-bdr);
    border-radius:var(--radius);padding:24px;margin-bottom:20px;
}

/* BADGES */
.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:999px;font-size:12px;font-weight:600;white-space:nowrap}
.badge-normal{background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.35);color:#34D399}
.badge-attack{background:rgba(239,68,68,.15);border:1px solid rgba(239,68,68,.40);color:#F87171;box-shadow:0 0 8px rgba(239,68,68,.20)}
.badge-medium{background:rgba(245,158,11,.15);border:1px solid rgba(245,158,11,.35);color:#FCD34D}

/* INPUT */
[data-testid="stTextInput"] input,[data-baseweb="select"]{
    background:#0D1528!important;border:1px solid var(--card-bdr)!important;
    border-radius:var(--radius-sm)!important;color:var(--text-1)!important;
    font-family:'Inter',sans-serif!important;
}

/* DATAFRAME */
[data-testid="stDataFrame"],[data-testid="stDataFrame"] iframe{border-radius:var(--radius)!important}

/* BUTTONS */
.stButton>button{
    background:linear-gradient(135deg,var(--blue),var(--purple))!important;
    color:white!important;border:none!important;border-radius:12px!important;
    padding:10px 22px!important;font-family:'Inter',sans-serif!important;
    font-weight:600!important;font-size:14px!important;
    transition:var(--transition)!important;
    box-shadow:0 4px 16px rgba(59,130,246,.30)!important;
}
.stButton>button:hover{transform:translateY(-2px)!important;box-shadow:0 8px 28px rgba(59,130,246,.45)!important}
.stButton>button:active{transform:translateY(0)!important}

/* ALERT CARDS */
.alert-card{
    background:var(--card-bg);border:1px solid var(--card-bdr);border-radius:var(--radius);
    padding:20px 24px;margin-bottom:12px;position:relative;overflow:hidden;transition:var(--transition);
}
.alert-card.high{border-left:3px solid var(--red);box-shadow:-4px 0 20px rgba(239,68,68,.20),0 0 0 1px rgba(239,68,68,.15)}
.alert-card.medium{border-left:3px solid var(--orange);box-shadow:-4px 0 16px rgba(245,158,11,.18)}
.alert-card.low{border-left:3px solid var(--blue)}
.alert-card:hover{transform:translateX(4px)}
.alert-card.high:hover{box-shadow:-6px 0 30px rgba(239,68,68,.35),0 0 0 1px rgba(239,68,68,.25)}
.alert-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px}
.alert-vid{font-size:16px;font-weight:700;color:var(--text-1)}
.alert-time{font-size:12px;color:var(--text-3)}
.alert-type{font-size:13px;color:var(--text-2);margin-bottom:10px}

/* GEMMA PANEL */
.gemma-panel{
    background:linear-gradient(135deg,rgba(15,22,41,.90) 0%,rgba(11,16,40,.95) 100%);
    border:1px solid rgba(139,92,246,.35);border-radius:var(--radius);
    padding:28px;position:relative;overflow:hidden;
}
.gemma-panel::before{
    content:'';position:absolute;top:-40px;right:-40px;width:200px;height:200px;
    background:radial-gradient(circle,rgba(139,92,246,.12),transparent 70%);pointer-events:none;
}
.gemma-title{display:flex;align-items:center;gap:10px;font-size:16px;font-weight:700;color:#A78BFA;margin-bottom:20px}
.gemma-title-icon{
    width:32px;height:32px;border-radius:9px;background:rgba(139,92,246,.20);
    border:1px solid rgba(139,92,246,.40);display:flex;align-items:center;justify-content:center;font-size:16px;
}
.chat-bubble-ai{
    background:rgba(139,92,246,.10);border:1px solid rgba(139,92,246,.25);
    border-radius:16px 16px 16px 4px;padding:16px 18px;margin-bottom:12px;
    font-size:14px;color:var(--text-2);line-height:1.7;position:relative;
}
.chat-bubble-ai::before{content:'🤖';position:absolute;top:-12px;left:12px;font-size:18px}
.gemma-field{display:flex;flex-direction:column;gap:4px;padding:12px 0;border-bottom:1px solid rgba(255,255,255,.06)}
.gemma-field:last-child{border-bottom:none}
.gemma-field-label{font-size:11px;font-weight:700;color:var(--text-3);text-transform:uppercase;letter-spacing:.8px}
.gemma-field-value{font-size:14px;color:var(--text-1);line-height:1.6}
.risk-bar{height:8px;border-radius:999px;background:rgba(255,255,255,.08);margin-top:6px;overflow:hidden}
.risk-fill{height:100%;border-radius:999px;transition:width .6s ease}
.risk-fill.high{background:linear-gradient(90deg,var(--orange),var(--red));box-shadow:0 0 8px rgba(239,68,68,.5)}
.risk-fill.medium{background:linear-gradient(90deg,#D97706,var(--orange))}
.risk-fill.low{background:linear-gradient(90deg,var(--green),#059669)}

/* PROGRESS BARS */
.conf-bar{display:flex;align-items:center;gap:8px;font-size:12px}
.conf-bar-track{flex:1;height:6px;background:rgba(255,255,255,.08);border-radius:999px;overflow:hidden}
.conf-bar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--blue),var(--purple))}
.conf-bar-val{color:var(--text-2);width:40px;text-align:right}

/* VEHICLE CARDS */
.vehicle-card{
    background:var(--card-bg);border:1px solid var(--card-bdr);border-radius:var(--radius);
    padding:18px 20px;transition:var(--transition);cursor:pointer;position:relative;overflow:hidden;margin-bottom:16px;
}
.vehicle-card:hover{border-color:rgba(59,130,246,.40);transform:translateY(-3px);box-shadow:var(--glow-blue)}
.vehicle-card.attack-card{border-color:rgba(239,68,68,.25)}
.vehicle-card.attack-card:hover{border-color:rgba(239,68,68,.50);box-shadow:var(--glow-red)}
.vehicle-id{font-size:16px;font-weight:700;color:var(--text-1);margin-bottom:10px}
.vehicle-meta{display:flex;flex-wrap:wrap;gap:12px;margin-top:10px}
.vehicle-meta-item{font-size:12px;color:var(--text-3)}
.vehicle-meta-val{font-size:13px;font-weight:600;color:var(--text-1)}

/* SETTINGS */
.settings-section{
    background:var(--card-bg);border:1px solid var(--card-bdr);border-radius:var(--radius);
    padding:24px;margin-bottom:20px;
}
.settings-title{
    font-size:15px;font-weight:700;color:var(--text-1);margin-bottom:16px;
    padding-bottom:12px;border-bottom:1px solid rgba(255,255,255,.07);
}

/* DIVIDER */
.styled-divider{
    height:1px;background:linear-gradient(90deg,transparent,rgba(59,130,246,.35),transparent);
    margin:28px 0;border:none;
}

/* EXPANDER */
[data-testid="stExpander"]{
    background:var(--card-bg)!important;border:1px solid var(--card-bdr)!important;
    border-radius:var(--radius)!important;margin-bottom:10px!important;
}
[data-testid="stExpander"] summary{color:var(--text-1)!important;font-weight:600!important}

/* SCROLLBAR */
::-webkit-scrollbar{width:6px;height:6px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:#2A3655;border-radius:999px}
::-webkit-scrollbar-thumb:hover{background:#3B82F6}

/* ANIMATIONS */
@keyframes fadeInUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.fade-in{animation:fadeInUp .5s ease forwards}
.fade-in-1{animation:fadeInUp .5s .1s ease both}
.fade-in-2{animation:fadeInUp .5s .2s ease both}
.fade-in-3{animation:fadeInUp .5s .3s ease both}
.fade-in-4{animation:fadeInUp .5s .4s ease both}

/* HIDE STREAMLIT CHROME */
#MainMenu,footer,header{visibility:hidden!important}
.stDeployButton{display:none!important}
.block-container{padding-top:2rem!important;padding-bottom:2rem!important}

div[data-testid="stSidebarNav"]{display:none!important}
</style>
""", unsafe_allow_html=True)

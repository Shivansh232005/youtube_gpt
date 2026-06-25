import streamlit as st
import re
import os
import json
import warnings

warnings.filterwarnings("ignore")

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="VidSearch AI", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #000000 !important;
    color: #F8FAFC !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top, rgba(99,102,241,0.03), transparent 30%),
        #000000 !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stHeader"], header {
    display: none !important; height: 0 !important; min-height: 0 !important;
}
.block-container { padding: 0 !important; margin-top: 0 !important; max-width: 100% !important; }
[data-testid="stMain"], [data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"] > section { padding-top: 0 !important; margin-top: 0 !important; }
[data-testid="stColumn"] { padding: 0 !important; margin: 0 !important; }
[data-testid="stVerticalBlock"], [data-testid="stVerticalBlockSizeWrapper"] { padding: 0 !important; margin: 0 !important; gap: 0 !important; }
[data-testid="stHorizontalBlock"] { gap: 0 !important; padding: 0 !important; margin: 0 !important; align-items: stretch !important; }
[data-testid="stColumn"] > div > div { gap: 0 !important; }
div[data-testid="element-container"], div.stMarkdown { margin: 0 !important; padding: 0 !important; }
div[data-testid="element-container"]:has(input),
div[data-testid="element-container"]:has(button),
div[data-testid="element-container"]:has(select),
div[data-testid="element-container"]:has([data-testid="stSelectbox"]),
div[data-testid="element-container"]:has([data-testid="stSlider"]) { padding: 0.25rem 1rem !important; }

/* ── Keyframe animations ── */
@keyframes fadeInUp { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes float { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-8px); } }
@keyframes gradientShift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes pulse-glow { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
@keyframes blob-drift { 0%, 100% { transform: translate(0,0) scale(1); } 33% { transform: translate(20px,-15px) scale(1.05); } 66% { transform: translate(-10px,10px) scale(0.97); } }

/* ── Navbar ── */
.vs-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 2rem; height: 56px;
    background: rgba(5,8,22,0.85);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    position: sticky; top: 0; z-index: 200;
}
.vs-nav-logo { display: flex; align-items: center; gap: 0.5rem; font-weight: 700; font-size: 0.95rem; letter-spacing: -0.02em; color: #F8FAFC; }
.vs-nav-logo-icon { width: 28px; height: 28px; background: linear-gradient(135deg,#6366F1,#06B6D4); border-radius: 7px; display: flex; align-items: center; justify-content: center; font-size: 0.75rem; box-shadow: 0 0 16px rgba(99,102,241,0.4); }
.vs-nav-pills { display: flex; align-items: center; gap: 0.4rem; }
.vs-pill { font-size: 0.68rem; font-weight: 500; letter-spacing: 0.02em; padding: 0.22rem 0.7rem; border-radius: 100px; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); color: #94A3B8; }
.vs-pill.indigo { background: rgba(99,102,241,0.1); border-color: rgba(99,102,241,0.25); color: #A5B4FC; }
.vs-pill.cyan { background: rgba(6,182,212,0.08); border-color: rgba(6,182,212,0.2); color: #67E8F9; }
.vs-pill.green { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.2); color: #34D399; }

/* ── Panel layout ──
   NOTE: these rules now target the REAL Streamlit column containers,
   not the .vs-sidebar/.vs-main/.vs-chat marker divs. A
   st.markdown('<div class="...">') opened in one call and closed in a
   later call does NOT actually wrap the widgets rendered in between —
   each st.markdown/st.button/etc call creates its own sibling element in
   the DOM. That left an EMPTY div carrying `min-height: calc(100vh - 56px)`
   sitting above the real content — that empty box was the big blank gap. */
[data-testid="stColumn"]:nth-of-type(1) { padding: 1rem 0.75rem !important; border-right: 1px solid rgba(255,255,255,0.05) !important; min-height: calc(100vh - 56px) !important; overflow-y: auto !important; }
[data-testid="stColumn"]:nth-of-type(2) { padding: 1rem 1rem 1rem 1.1rem !important; min-height: calc(100vh - 56px) !important; }
[data-testid="stColumn"]:nth-of-type(3) { display: flex !important; flex-direction: column !important; min-height: calc(100vh - 56px) !important; }
.vs-sidebar, .vs-main, .vs-chat { display: contents; }

/* ── Sidebar history ── */
.vs-sidebar-brand { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #1E293B; padding: 0.25rem 0.5rem 0.75rem; display: block; }
.vs-hist-section { font-size: 0.58rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #1E293B; padding: 0.6rem 0.5rem 0.2rem; display: block; }
.vs-hist-item { display: flex; align-items: center; gap: 0.4rem; padding: 0.4rem 0.5rem; border-radius: 8px; transition: all 0.15s; margin-bottom: 1px; }
.vs-hist-item:hover { background: rgba(255,255,255,0.04); }
.vs-hist-item.active { background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.15); }
.vs-hist-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; background: rgba(99,102,241,0.35); }
.vs-hist-item.active .vs-hist-dot { background: #6366F1; }
.vs-hist-title { font-size: 0.76rem; color: #475569; font-weight: 400; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.4; }
.vs-hist-item.active .vs-hist-title { color: #C7D2FE; }
.vs-hist-empty { color: #1E293B; font-size: 0.78rem; text-align: center; padding: 2.5rem 0.75rem; line-height: 2; }

/* ── Hero / empty state ── */
.vs-hero { padding: 2.5rem 1.5rem 2rem; text-align: center; animation: fadeInUp 0.6s ease both; }
.vs-hero-badge { display: inline-flex; align-items: center; gap: 0.4rem; background: rgba(99,102,241,0.08); border: 1px solid rgba(99,102,241,0.2); color: #A5B4FC; font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; padding: 0.3rem 0.8rem; border-radius: 100px; margin-bottom: 1.5rem; }
.vs-hero-badge-dot { width: 5px; height: 5px; border-radius: 50%; background: #6366F1; animation: pulse-glow 2s ease-in-out infinite; }
.vs-hero-headline { font-size: clamp(1.5rem, 3vw, 2.2rem); font-weight: 700; letter-spacing: -0.04em; line-height: 1.15; margin-bottom: 1rem; color: #F8FAFC; }
.vs-hero-gradient-text { background: linear-gradient(135deg,#6366F1 0%,#06B6D4 50%,#A5B4FC 100%); background-size: 200% auto; -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; animation: gradientShift 4s ease infinite; }
.vs-hero-sub { font-size: 0.88rem; color: #64748B; line-height: 1.7; max-width: 360px; margin: 0 auto 2rem; }
.vs-feature-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.6rem; margin-bottom: 1.5rem; }
.vs-feature-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 0.75rem; text-align: left; transition: all 0.25s; }
.vs-feature-card:hover { background: rgba(99,102,241,0.05); border-color: rgba(99,102,241,0.2); transform: translateY(-2px); }
.vs-feature-icon { font-size: 1rem; margin-bottom: 0.35rem; display: block; }
.vs-feature-title { font-size: 0.73rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.15rem; }
.vs-feature-desc { font-size: 0.67rem; color: #475569; line-height: 1.5; }
.vs-example-urls { display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: center; margin-top: 0.75rem; }
.vs-example-url { background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); color: #475569; font-size: 0.67rem; padding: 0.25rem 0.65rem; border-radius: 6px; }

/* ── URL & settings sections ── */
.vs-url-section { background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1.1rem; margin-bottom: 0.75rem; }
.vs-section-label { font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #6366F1; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem; }
.vs-section-label-dot { width: 4px; height: 4px; border-radius: 50%; background: #6366F1; }
.vs-glass-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 0.9rem 1rem; margin-bottom: 0.6rem; }
.vs-card-label { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.13em; text-transform: uppercase; color: #6366F1; margin-bottom: 0.65rem; display: flex; align-items: center; gap: 0.35rem; }

/* ── Video card ── */
.vs-video-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; overflow: hidden; margin-bottom: 0.75rem; }
.vs-video-info { padding: 0.85rem 1rem; }
.vs-video-title { font-size: 0.84rem; font-weight: 600; color: #E2E8F0; margin-bottom: 0.3rem; line-height: 1.4; }
.vs-video-meta { display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }
.vs-video-channel { font-size: 0.72rem; color: #6366F1; font-weight: 500; }
.vs-video-sep { color: #334155; font-size: 0.65rem; }
.vs-video-duration { font-size: 0.7rem; color: #64748B; }
.vs-ready-badge { display: inline-flex; align-items: center; gap: 0.3rem; background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.18); color: #10B981; font-size: 0.68rem; font-weight: 600; padding: 0.2rem 0.65rem; border-radius: 100px; margin-top: 0.4rem; }
.vs-ready-dot { width: 5px; height: 5px; border-radius: 50%; background: #10B981; animation: pulse-glow 2s ease-in-out infinite; }
.vs-cache-badge { display: inline-flex; align-items: center; gap: 0.3rem; background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2); color: #F59E0B; font-size: 0.68rem; font-weight: 600; padding: 0.18rem 0.6rem; border-radius: 100px; }
[data-testid="stVideo"] { border-radius: 0 !important; overflow: hidden !important; }

/* ── Analytics cards ── */
.vs-analytics-row { display: grid; grid-template-columns: repeat(2,1fr); gap: 0.5rem; margin: 0.75rem 0; }
.vs-stat-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); border-radius: 10px; padding: 0.75rem; transition: all 0.2s; }
.vs-stat-card:hover { border-color: rgba(99,102,241,0.2); background: rgba(99,102,241,0.04); }
.vs-stat-num { font-size: 1.3rem; font-weight: 700; color: #F8FAFC; letter-spacing: -0.03em; }
.vs-stat-num.indigo { color: #818CF8; }
.vs-stat-num.cyan { color: #22D3EE; }
.vs-stat-num.green { color: #34D399; }
.vs-stat-num.violet { color: #C084FC; }
.vs-stat-label { font-size: 0.65rem; color: #475569; margin-top: 0.15rem; font-weight: 500; }

/* ── Inputs & buttons ── */
[data-testid="stTextInput"] > div > div > input { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 9px !important; color: #F8FAFC !important; font-size: 0.85rem !important; padding: 0.6rem 1rem !important; transition: all 0.2s !important; }
[data-testid="stTextInput"] > div > div > input:focus { border-color: rgba(99,102,241,0.4) !important; box-shadow: 0 0 0 3px rgba(99,102,241,0.08) !important; }
[data-testid="stTextInput"] label { display: none !important; }
[data-testid="stButton"] > button { width: 100% !important; background: linear-gradient(135deg,#6366F1 0%,#4F46E5 100%) !important; color: white !important; border: none !important; border-radius: 9px !important; padding: 0.65rem 1rem !important; font-family: 'Inter', sans-serif !important; font-size: 0.83rem !important; font-weight: 600 !important; letter-spacing: 0.01em !important; margin-top: 0.4rem !important; transition: all 0.2s !important; box-shadow: 0 4px 16px rgba(99,102,241,0.3) !important; }
[data-testid="stButton"] > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 24px rgba(99,102,241,0.45) !important; filter: brightness(1.05) !important; }
[data-testid="stButton"] > button:active { transform: translateY(0) !important; }
[data-testid="stSelectbox"] label { display: none !important; }
[data-testid="stSelectbox"] > div > div { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 9px !important; color: #E2E8F0 !important; font-size: 0.83rem !important; }
[data-testid="stSlider"] label { color: #475569 !important; font-size: 0.75rem !important; }

/* ── Tabs ── */
[data-testid="stTabs"] > div:first-child { border-bottom: 1px solid rgba(255,255,255,0.06) !important; gap: 0 !important; }
[data-testid="stTabs"] button { font-family: 'Inter', sans-serif !important; font-size: 0.75rem !important; font-weight: 600 !important; color: #475569 !important; padding: 0.5rem 0.9rem !important; letter-spacing: 0.01em !important; border: none !important; border-bottom: 2px solid transparent !important; transition: all 0.2s !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #A5B4FC !important; border-bottom-color: #6366F1 !important; background: transparent !important; }

/* ── Chat ── */
.vs-chat-header { padding: 0.85rem 1.25rem; border-bottom: 1px solid rgba(255,255,255,0.05); display: flex; align-items: center; justify-content: space-between; }
.vs-chat-title { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #334155; }
.vs-chat-model-badge { font-size: 0.65rem; font-weight: 500; color: #6366F1; display: flex; align-items: center; gap: 0.3rem; }
.vs-chat-model-dot { width: 5px; height: 5px; border-radius: 50%; background: #6366F1; }
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0 !important; margin-bottom: 0.75rem !important; animation: fadeInUp 0.3s ease both !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: rgba(99,102,241,0.06) !important; border: 1px solid rgba(99,102,241,0.12) !important; border-radius: 12px !important; padding: 0.75rem 1rem !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.05) !important; border-radius: 12px !important; padding: 0.75rem 1rem !important; }
[data-testid="stChatMessage"] p { color: #E2E8F0 !important; font-size: 0.87rem !important; line-height: 1.75 !important; }
[data-testid="stChatInput"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 12px !important; margin: 0 1.25rem 1rem !important; }
[data-testid="stChatInput"] textarea { color: #E2E8F0 !important; font-size: 0.87rem !important; }
.vs-chat-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; min-height: 280px; padding: 2.5rem 2rem; text-align: center; animation: fadeIn 0.5s ease both; }
.vs-chat-empty-ring { width: 54px; height: 54px; border-radius: 50%; border: 1px solid rgba(99,102,241,0.2); display: flex; align-items: center; justify-content: center; font-size: 1.3rem; margin-bottom: 1rem; background: rgba(99,102,241,0.05); animation: float 4s ease-in-out infinite; }
.vs-chat-empty-title { font-size: 0.88rem; font-weight: 600; color: #334155; margin-bottom: 0.35rem; }
.vs-chat-empty-sub { font-size: 0.76rem; color: #1E293B; line-height: 1.6; }

/* ── AI answer & segments ── */
.vs-ai-box { background: linear-gradient(135deg,rgba(16,185,129,0.05) 0%,rgba(6,182,212,0.03) 100%); border: 1px solid rgba(16,185,129,0.15); border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 0.8rem; }
.vs-ai-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.6rem; }
.vs-ai-label { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #10B981; display: flex; align-items: center; gap: 0.4rem; }
.vs-ai-source { font-size: 0.62rem; font-weight: 500; padding: 0.1rem 0.5rem; border-radius: 5px; background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.2); color: #34D399; }
.vs-ai-text { color: #CBD5E1; font-size: 0.87rem; line-height: 1.8; }
.vs-segs-header { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #334155; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.04); margin-bottom: 0.65rem; }
.vs-seg-card { background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.05); border-left: 2px solid rgba(6,182,212,0.4); border-radius: 0 10px 10px 0; padding: 0.75rem 0.9rem; margin-bottom: 0.5rem; transition: all 0.2s; }
.vs-seg-card:hover { border-color: rgba(6,182,212,0.3); background: rgba(6,182,212,0.03); }
.vs-seg-meta { display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.35rem; flex-wrap: wrap; }
.vs-seg-num { font-size: 0.58rem; font-weight: 700; color: #1E293B; text-transform: uppercase; }
.vs-ts-pill { background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.18); color: #22D3EE; font-size: 0.65rem; font-weight: 500; padding: 0.1rem 0.55rem; border-radius: 5px; text-decoration: none; }
.vs-ts-pill:hover { background: rgba(6,182,212,0.18); }
.vs-match-high { background: rgba(16,185,129,0.08); border: 1px solid rgba(16,185,129,0.18); color: #34D399; font-size: 0.62rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 5px; }
.vs-match-mid { background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.18); color: #FBBF24; font-size: 0.62rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 5px; }
.vs-match-low { background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.15); color: #F87171; font-size: 0.62rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 5px; }
.vs-seg-text { color: #475569; font-size: 0.82rem; line-height: 1.7; }
.vs-video-placeholder { background: rgba(255,255,255,0.015); border: 1px solid rgba(255,255,255,0.06); border-radius: 12px; aspect-ratio: 16/9; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 0.4rem; }
.vs-placeholder-icon { font-size: 1.8rem; opacity: 0.35; animation: float 4s ease-in-out infinite; }
.vs-placeholder-text { font-size: 0.75rem; color: #1E293B; }

/* ── Cite tags ── */
.cite-tag { display: inline-flex; align-items: center; gap: 3px; background: rgba(6,182,212,0.08); border: 1px solid rgba(6,182,212,0.18); color: #22D3EE; font-size: 0.65rem; font-weight: 600; padding: 0.05rem 0.45rem; border-radius: 4px; cursor: pointer; text-decoration: none; vertical-align: middle; margin: 0 2px; }
.cite-tag:hover { background: rgba(6,182,212,0.18); }

/* ── Misc ── */
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg,#6366F1,#06B6D4) !important; border-radius: 100px !important; }
[data-testid="stAlert"] { border-radius: 10px !important; font-size: 0.83rem !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.06); border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.3); }

@media (max-width: 768px) {
    .vs-nav { padding: 0 1rem; }
    .vs-hero-headline { font-size: 1.4rem; }
    .vs-feature-grid { grid-template-columns: 1fr; }
    .vs-analytics-row { grid-template-columns: repeat(2,1fr); }
    [data-testid="stChatInput"] { margin: 0 0.75rem 0.75rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ── History helpers ────────────────────────────────────────────────────────────
HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".search_history.json")

def load_history():
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return []

def save_history(entry):
    history = load_history()
    # Remove duplicate video_id entries, keep latest
    history = [h for h in history if h.get("video_id") != entry["video_id"]]
    history.insert(0, entry)
    history = history[:20]  # keep last 20
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False)
    except Exception:
        pass

# ── Cache dir ─────────────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".transcript_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(video_id):
    return os.path.join(CACHE_DIR, f"{video_id}.json")

def load_cached(video_id):
    cp = cache_path(video_id)
    if os.path.exists(cp):
        try:
            with open(cp, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def save_cache(video_id, transcript):
    try:
        with open(cache_path(video_id), "w", encoding="utf-8") as f:
            json.dump(transcript, f, ensure_ascii=False)
    except Exception:
        pass

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_video_id(youtube_url):
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/shorts\/)([a-zA-Z0-9_-]{11})',
        r'([a-zA-Z0-9_-]{11})'
    ]
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            vid = match.group(1).strip()
            if len(vid) == 11:
                return vid
    raise ValueError("Could not extract YouTube video ID from URL")

def format_timestamp(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"

# ── Transcript: YouTube captions ──────────────────────────────────────────────
def get_transcript(video_id, lang="auto"):
    # 1. Check cache
    cached = load_cached(video_id)
    if cached:
        st.session_state.from_cache = True
        return cached
    st.session_state.from_cache = False

    # 2. Try YouTube captions — uses new v1.x API (YouTubeTranscriptApi instance)
    try:
        ytt = YouTubeTranscriptApi()

        # Build preferred language list
        if lang != "auto" and lang != "en":
            preferred_langs = [lang, "en"]
        else:
            preferred_langs = ["en", "hi", "gu", "ur", "ta", "te", "bn", "mr"]

        try:
            # fetch() directly tries languages in order
            data = ytt.fetch(video_id, languages=preferred_langs)
        except NoTranscriptFound:
            # Fallback: list available transcripts, take first, translate to English
            transcript_list = ytt.list(video_id)
            first = next(iter(transcript_list))
            data = first.translate("en").fetch()

        # FetchedTranscript snippets expose .text, .start, .duration as attributes
        result = []
        for e in data.snippets:
            result.append({
                "text":     e.text,
                "start":    e.start,
                "duration": getattr(e, "duration", 0)
            })

        save_cache(video_id, result)
        return result

    except TranscriptsDisabled:
        st.warning("⚡ YouTube captions are disabled for this video — switching to Whisper AI...")
        return get_transcript_whisper(video_id, lang)
    except Exception as e:
        err = str(e).lower()
        if any(x in err for x in ["subtitles are disabled", "no transcript", "no element found",
                                    "too many requests", "429", "could not retrieve", "transcript disabled"]):
            st.warning("⚡ YouTube captions unavailable — switching to Whisper AI...")
        else:
            st.warning(f"Caption fetch issue ({str(e)[:120]}) — trying Whisper AI...")
        return get_transcript_whisper(video_id, lang)

# ── Transcript: Whisper fallback ──────────────────────────────────────────────
def get_transcript_whisper(video_id, lang="auto"):
    try:
        import yt_dlp
        import whisper
        import glob

        current_dir = os.path.dirname(os.path.abspath(__file__))
        cookie_path = os.path.join(current_dir, "cookies.txt")
        audio_file  = os.path.join(current_dir, f"audio_{video_id}")
        url = f"https://www.youtube.com/watch?v={video_id}"

        progress = st.progress(0, text="📥 Downloading audio...")

        # ── yt-dlp options ──────────────────────────────────────────────────
        ydl_opts = {
            "format": "worstaudio/bestaudio/best",
            "outtmpl": audio_file + ".%(ext)s",   # force extension in filename
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": False,
            "socket_timeout": 30,
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
            # Convert to mp3 so Whisper always gets a supported format
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "64",
            }],
        }
        if os.path.exists(cookie_path):
            ydl_opts["cookiefile"] = cookie_path

        # Try download — attempt each format in order
        downloaded = None
        last_error = ""
        for fmt in ["worstaudio/bestaudio", "bestaudio/best", "best"]:
            try:
                ydl_opts["format"] = fmt
                # Clean any leftover partial files
                for f in glob.glob(f"{audio_file}.*"):
                    try:
                        os.remove(f)
                    except Exception:
                        pass

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    if info is None:
                        raise Exception("yt-dlp returned no info (video may be private or age-restricted)")

                # After postprocessor, file will be .mp3
                matches = glob.glob(f"{audio_file}*.mp3")
                if not matches:
                    matches = glob.glob(f"{audio_file}*")
                matches = [f for f in matches if os.path.isfile(f)]

                if matches:
                    downloaded = matches[0]
                    break
            except Exception as dl_err:
                last_error = str(dl_err)
                continue

        if not downloaded:
            st.error(
                f"❌ Audio download failed. This usually means:\n"
                f"- The video is age-restricted or private\n"
                f"- YouTube is blocking downloads in this environment (common on Streamlit Cloud)\n"
                f"- Try adding a `cookies.txt` file next to app.py (export from your browser)\n\n"
                f"**Technical detail:** {last_error[:200] if last_error else 'No output files found'}"
            )
            return None

        progress.progress(45, text="🎙️ Transcribing with Whisper AI...")

        # ── Whisper transcription ──
        whisper_lang = None if lang == "auto" else lang
        model_size   = "small"   # small = best quality/speed balance for multilingual

        model  = whisper.load_model(model_size)
        kwargs = {"verbose": False}
        if whisper_lang:
            kwargs["language"] = whisper_lang

        result = model.transcribe(downloaded, **kwargs)

        progress.progress(90, text="🔧 Processing segments...")

        # Clean up audio
        try:
            os.remove(downloaded)
        except Exception:
            pass

        transcript = [
            {
                "text":     seg["text"].strip(),
                "start":    seg["start"],
                "duration": seg["end"] - seg["start"]
            }
            for seg in result["segments"]
            if seg["text"].strip()
        ]

        save_cache(video_id, transcript)
        progress.progress(100, text="✅ Transcription complete!")
        return transcript

    except ImportError as e:
        st.error(f"Missing package: {e}. Run: pip install yt-dlp openai-whisper")
        return None
    except Exception as e:
        st.error(f"Whisper transcription failed: {str(e)}")
        return None

# ── Chunking ──────────────────────────────────────────────────────────────────
def process_transcript(transcript):
    if not transcript:
        return []
    docs = []
    window, overlap = 8, 3
    for i in range(0, len(transcript), window - overlap):
        chunk = transcript[i: i + window]
        if not chunk:
            continue
        text       = " ".join(e["text"].strip() for e in chunk)
        start_time = chunk[0]["start"]
        end_time   = chunk[-1]["start"] + chunk[-1].get("duration", 0)
        docs.append(Document(
            page_content=text,
            metadata={
                "start_time":      start_time,
                "end_time":        end_time,
                "timestamp":       format_timestamp(start_time),
                "timestamp_range": f"{format_timestamp(start_time)} - {format_timestamp(end_time)}"
            }
        ))
    return docs

# ── Embeddings ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def build_vectorstore(docs):
    try:
        return FAISS.from_documents(documents=docs, embedding=load_embeddings())
    except Exception as e:
        st.error(f"Vector store failed: {e}")
        return None

# ── BM25 ──────────────────────────────────────────────────────────────────────
def build_bm25(docs):
    try:
        from rank_bm25 import BM25Okapi
        return BM25Okapi([d.page_content.lower().split() for d in docs])
    except Exception:
        return None

# ── Hybrid search ─────────────────────────────────────────────────────────────
def hybrid_search(vectorstore, bm25, docs, question, k=8):
    sem = vectorstore.similarity_search_with_score(question, k=k * 2)
    sem_map = {d.page_content: (d, s) for d, s in sem}

    if bm25 is None:
        return sem[:k]

    from rank_bm25 import BM25Okapi
    bm25_scores = bm25.get_scores(question.lower().split())
    top_bm25    = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k * 2]

    sem_rank  = {d.page_content: r for r, (d, _) in enumerate(sem)}
    bm25_rank = {docs[i].page_content: r for r, i in enumerate(top_bm25)}

    all_keys = set(sem_rank) | set(bm25_rank)
    fused    = {c: 1/(60 + sem_rank.get(c, 999)) + 1/(60 + bm25_rank.get(c, 999)) for c in all_keys}

    results, seen = [], set()
    for c in sorted(fused, key=fused.get, reverse=True)[:k]:
        if c in seen:
            continue
        seen.add(c)
        if c in sem_map:
            results.append(sem_map[c])
        else:
            for d in docs:
                if d.page_content == c:
                    results.append((d, 1.0))
                    break
    return results

# ── Reranking ─────────────────────────────────────────────────────────────────
def rerank(question, results, top_k=5):
    try:
        from sentence_transformers import CrossEncoder
        ce     = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        scores = ce.predict([(question, d.page_content) for d, _ in results])
        ranked = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)
        return [(d, s) for (d, _), s in ranked[:top_k]]
    except Exception:
        return results[:top_k]

# ── Ollama AI answer ──────────────────────────────────────────────────────────
# ── Language instruction helper ───────────────────────────────────────────────
def get_lang_instruction(lang):
    if lang in ["hi", "gu", "ur", "bn", "mr", "ta", "te"]:
        return (
            "Reply in Hinglish ONLY — mix Hindi and English naturally, "
            "like: 'Is video mein bataya gaya hai ki...' "
            "Do NOT reply in pure Hindi or pure English."
        )
    elif lang == "auto":
        return (
            "If the transcript is in Hindi/Indian language, reply in Hinglish "
            "(Hindi + English mix). If English transcript, reply in English."
        )
    else:
        return "Reply in English only."

# ── Smart system prompt ───────────────────────────────────────────────────────
def build_prompt(question, docs, lang):
    context = "\n\n".join(
        f"[Timestamp {d.metadata['timestamp_range']}]:\n{d.page_content}"
        for d in docs
    )
    lang_instruction = get_lang_instruction(lang)
    return f"""You are an expert video assistant — like ChatGPT but specialized for YouTube videos.

{lang_instruction}

Your job:
- Give clear, detailed, easy-to-understand answers based on the video transcript
- Use simple language, like explaining to a friend
- Use bullet points or numbered steps when explaining concepts
- Mention timestamps when relevant so user can jump to that part
- If something is not in the transcript, say so honestly

VIDEO TRANSCRIPT:
{context}

USER QUESTION: {question}

Give a helpful, well-structured answer:"""

# ── Gemini API answer ─────────────────────────────────────────────────────────
def gemini_answer(question, docs, api_key, lang="auto", gemini_model="gemini-2.5-flash"):
    import time
    try:
        try:
            from google import genai
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "google-genai"])
            from google import genai

        client = genai.Client(api_key=api_key)
        prompt = build_prompt(question, docs, lang)

        # Retry logic: 3 attempts with backoff, fallback to lite model on 503
        models_to_try = [gemini_model]
        if gemini_model != "gemini-2.5-flash-lite":
            models_to_try.append("gemini-2.5-flash-lite")  # fallback

        last_error = ""
        for attempt, model in enumerate(models_to_try):
            for retry in range(3):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                    label = model + (" (fallback)" if attempt > 0 else "")
                    return response.text, label
                except Exception as e:
                    last_error = str(e)
                    if "503" in last_error or "UNAVAILABLE" in last_error or "high demand" in last_error:
                        if retry < 2:
                            time.sleep(2 ** retry)  # 1s, 2s wait
                            continue
                        break  # try fallback model
                    return None, last_error  # non-503 error, don't retry

        return None, f"Gemini overloaded — try again in a moment. ({last_error[:80]})"
    except Exception as e:
        return None, str(e)

# ── Ollama answer ─────────────────────────────────────────────────────────────
def ollama_answer(question, docs, model_name="llama3", lang="auto"):
    try:
        import ollama
        prompt = build_prompt(question, docs, lang)
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"], "ollama"
    except Exception as e:
        return None, str(e)

# ── Smart AI: Gemini first, Ollama fallback ───────────────────────────────────
def get_ai_answer(question, docs, ai_mode, gemini_api_key, ollama_model, lang, gemini_model="gemini-2.5-flash"):
    ans, source, error = None, None, None

    if ai_mode in ["gemini", "both"]:
        key = gemini_api_key.strip() if gemini_api_key else ""
        if key and (key.startswith("AIza") or key.startswith("AQ.")):
            ans, info = gemini_answer(question, docs, key, lang, gemini_model)
            if ans:
                source = gemini_model
            else:
                error = info
        else:
            error = "Invalid or missing Gemini API key (should start with AIza... or AQ...)"

    if ans is None and ai_mode in ["ollama", "both"]:
        ans, info = ollama_answer(question, docs, ollama_model, lang)
        if ans:
            source = f"Ollama ({ollama_model})"
        elif not error:
            error = info

    return ans, source, error

# ── Build result HTML ─────────────────────────────────────────────────────────
def build_results(results, video_id, question, ai_mode, gemini_api_key, ollama_model, min_match, lang="auto", gemini_model="gemini-2.5-flash"):
    # Filter by min_match — fixed score formula
    filtered = []
    for doc, score in results:
        rel = max(0, min(100, (1 - score) * 100))
        if rel >= min_match:
            filtered.append((doc, rel))

    # If nothing passes threshold, take top 3 anyway
    if not filtered:
        if results:
            filtered = [(doc, max(0, min(100, (1 - score) * 100))) for doc, score in results[:3]]
        else:
            msg = "No segments found. Try a different question or lower the Min Match % slider."
            return msg, msg

    # AI answer — premium redesign
    ai_html, ai_plain = "", ""
    if ai_mode != "none":
        with st.spinner("🤖 Generating AI answer..."):
            ans, source, error = get_ai_answer(
                question, [d for d, _ in filtered],
                ai_mode, gemini_api_key, ollama_model, lang, gemini_model
            )
        if ans:
            source_label = source or "AI"
            ai_html  = (
                f'<div class="vs-ai-box">'
                f'<div class="vs-ai-header">'
                f'<div class="vs-ai-label">🤖 AI Answer</div>'
                f'<span class="vs-ai-source">▶ {source_label}</span>'
                f'</div>'
                f'<div class="vs-ai-text">{ans}</div>'
                f'</div>'
            )
            ai_plain = f"**AI Answer ({source_label}):**\n{ans}\n\n"
        elif error:
            ai_html  = f'<div style="color:#F87171;font-size:0.8rem;margin-bottom:0.7rem;padding:0.6rem 0.8rem;background:rgba(239,68,68,0.06);border:1px solid rgba(239,68,68,0.15);border-radius:8px">⚠️ {error[:120]}</div>'
            ai_plain = f"AI Error: {error[:120]}\n\n"


    # Segment cards — premium redesign
    html  = ai_html + f'<div class="vs-segs-header">📍 {len(filtered)} Relevant Segments</div>'
    plain = ai_plain + f"**{len(filtered)} relevant segments:**\n"

    for i, (doc, rel) in enumerate(filtered, 1):
        ts       = doc.metadata.get("timestamp_range", "N/A")
        start_s  = int(doc.metadata.get("start_time", 0))
        yt_url   = f"https://www.youtube.com/watch?v={video_id}&t={start_s}s"
        text     = doc.page_content

        if rel >= 60:   mc = "vs-match-high"; icon = "↑"
        elif rel >= 35: mc = "vs-match-mid";  icon = "~"
        else:           mc = "vs-match-low";  icon = "↓"

        html += (
            f'<div class="vs-seg-card">'
            f'<div class="vs-seg-meta">'
            f'<span class="vs-seg-num">#{i}</span>'
            f'<a class="vs-ts-pill" href="{yt_url}" target="_blank">⏱ {ts}</a>'
            f'<span class="{mc}">{icon} {rel:.0f}%</span>'
            f'</div>'
            f'<div class="vs-seg-text">{text}</div>'
            f'</div>'
        )
        plain += f"\n**{i}. [{ts}]** ({rel:.0f}%)\n{text}\n"


    return html, plain

# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    # ── Premium Navbar ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="vs-nav">
        <div class="vs-nav-logo">
            <div class="vs-nav-logo-icon">🎬</div>
            VidSearch AI
        </div>
        <div class="vs-nav-pills">
            <span class="vs-pill indigo">⚡ Gemini AI</span>
            <span class="vs-pill cyan">🔍 Semantic Search</span>
            <span class="vs-pill green">● Free</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Session state init ─────────────────────────────────────────────────────
    defaults = {
        "messages":    [],
        "vectorstore": None,
        "bm25":        None,
        "docs":        [],
        "v_id":        None,
        "from_cache":  False,
        "active_tab":  "setup",
        "lang_sel":    "auto",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── 3-column layout ───────────────────────────────────────────────────────
    hist_col, left_col, right_col = st.columns([0.85, 1.4, 2], gap="small")

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORY SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    with hist_col:
        st.markdown('<div class="vs-sidebar">', unsafe_allow_html=True)
        st.markdown('<span class="vs-sidebar-brand">VidSearch AI</span>', unsafe_allow_html=True)

        history = load_history()
        if not history:
            st.markdown("""
            <div class="vs-hist-empty">
                <div style="font-size:1.4rem;opacity:0.25;margin-bottom:0.5rem">🎬</div>
                No history yet.<br>Process a video to begin.
            </div>
            """, unsafe_allow_html=True)
        else:
            if st.button("🗑", key="clear_hist", help="Clear all history"):
                try:
                    if os.path.exists(HISTORY_FILE):
                        os.remove(HISTORY_FILE)
                    st.rerun()
                except Exception:
                    pass

            import datetime
            today     = datetime.datetime.now().strftime("%d %b")
            yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%d %b")

            groups = {}
            for h in history:
                ts = h.get("timestamp", "")
                date_part = ts.split(",")[0].strip() if "," in ts else ts.strip()
                if date_part == today:
                    grp = "Today"
                elif date_part == yesterday:
                    grp = "Yesterday"
                else:
                    grp = "Older"
                groups.setdefault(grp, []).append(h)

            for grp_name in ["Today", "Yesterday", "Older"]:
                if grp_name not in groups:
                    continue
                st.markdown(f'<span class="vs-hist-section">{grp_name}</span>', unsafe_allow_html=True)

                for i, h in enumerate(groups[grp_name]):
                    vid    = h.get("video_id", "")
                    title  = h.get("title", vid)
                    is_active = vid == st.session_state.get("v_id", "")
                    display = vid if ("youtube.com" in title or "youtu.be" in title) else title
                    active_cls = "active" if is_active else ""
                    st.markdown(f"""
                    <div class="vs-hist-item {active_cls}">
                        <div class="vs-hist-dot"></div>
                        <div class="vs-hist-title">{display[:36]}{"…" if len(display)>36 else ""}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if st.button("↩", key=f"hist_{grp_name}_{i}_{vid}", help=f"Load: {display[:30]}"):
                        lang_sel = st.session_state.get("lang_sel", "auto")
                        with st.spinner("Loading..."):
                            raw = get_transcript(vid, lang=lang_sel)
                        if raw:
                            docs = process_transcript(raw)
                            st.session_state.vectorstore = build_vectorstore(docs)
                            st.session_state.bm25        = build_bm25(docs)
                            st.session_state.docs        = docs
                            st.session_state.v_id        = vid
                            st.session_state.messages    = []
                            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # LEFT PANEL — Video + Setup + Settings
    # ══════════════════════════════════════════════════════════════════════════
    with left_col:
        st.markdown('<div class="vs-main">', unsafe_allow_html=True)

        # ── Video display or hero ──────────────────────────────────────────
        if st.session_state.v_id:
            # Video card with embed
            st.markdown('<div class="vs-video-card">', unsafe_allow_html=True)
            st.video(f"https://www.youtube.com/watch?v={st.session_state.v_id}")
            st.markdown(f"""
            <div class="vs-video-info">
                <div class="vs-video-title">🎬 Video Loaded</div>
                <div class="vs-video-meta">
                    <span class="vs-video-channel">youtube.com</span>
                    <span class="vs-video-sep">·</span>
                    <span class="vs-video-duration">ID: {st.session_state.v_id}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if st.session_state.from_cache:
                st.markdown('<div class="vs-cache-badge">⚡ Loaded from cache</div>', unsafe_allow_html=True)

            # Analytics cards
            if st.session_state.docs:
                n_chunks   = len(st.session_state.docs)
                n_trans    = sum(len(d.page_content.split()) for d in st.session_state.docs)
                ai_mode_label = st.session_state.get("_ai_mode", "gemini").title()
                st.markdown(f"""
                <div class="vs-analytics-row">
                    <div class="vs-stat-card">
                        <div class="vs-stat-num indigo">{n_trans:,}</div>
                        <div class="vs-stat-label">Words indexed</div>
                    </div>
                    <div class="vs-stat-card">
                        <div class="vs-stat-num cyan">{n_chunks}</div>
                        <div class="vs-stat-label">Chunks built</div>
                    </div>
                    <div class="vs-stat-card">
                        <div class="vs-stat-num green">✓</div>
                        <div class="vs-stat-label">Vector index</div>
                    </div>
                    <div class="vs-stat-card">
                        <div class="vs-stat-num violet">{ai_mode_label}</div>
                        <div class="vs-stat-label">AI mode</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Hero / empty state
            st.markdown("""
            <div class="vs-hero">
                <div class="vs-hero-badge">
                    <div class="vs-hero-badge-dot"></div>
                    AI-Powered · Free · No Signup
                </div>
                <div class="vs-hero-headline">
                    Chat With Any<br>
                    <span class="vs-hero-gradient-text">YouTube Video</span>
                </div>
                <div class="vs-hero-sub">
                    Paste any YouTube URL. Ask questions. Get instant answers with timestamps.
                </div>
                <div class="vs-feature-grid">
                    <div class="vs-feature-card">
                        <span class="vs-feature-icon">🔍</span>
                        <div class="vs-feature-title">Hybrid Search</div>
                        <div class="vs-feature-desc">Semantic + BM25 retrieval for best results</div>
                    </div>
                    <div class="vs-feature-card">
                        <span class="vs-feature-icon">🤖</span>
                        <div class="vs-feature-title">Gemini AI</div>
                        <div class="vs-feature-desc">Powered by Google's latest models</div>
                    </div>
                    <div class="vs-feature-card">
                        <span class="vs-feature-icon">⏱</span>
                        <div class="vs-feature-title">Timestamps</div>
                        <div class="vs-feature-desc">Jump directly to relevant moments</div>
                    </div>
                    <div class="vs-feature-card">
                        <span class="vs-feature-icon">🌐</span>
                        <div class="vs-feature-title">Multilingual</div>
                        <div class="vs-feature-desc">Hindi, English, and 7 more languages</div>
                    </div>
                </div>
                <div style="font-size:0.65rem;color:#1E293B;margin-bottom:0.4rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase">Try these videos</div>
                <div class="vs-example-urls">
                    <span class="vs-example-url">youtube.com/watch?v=dQw4w9WgXcQ</span>
                    <span class="vs-example-url">youtu.be/...</span>
                    <span class="vs-example-url">YouTube Shorts</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Tabs: Video Setup | Settings ──────────────────────────────────
        tab1, tab2 = st.tabs(["⚙️ Video Setup", "🛠 Settings"])

        with tab1:
            st.markdown('<div class="vs-url-section">', unsafe_allow_html=True)
            st.markdown('<div class="vs-section-label"><div class="vs-section-label-dot"></div>YouTube URL</div>', unsafe_allow_html=True)
            url = st.text_input("url", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed", key="url_input")

            if st.button("⚡ Process Video", key="process_btn"):
                if not url.strip():
                    st.warning("Please enter a YouTube URL.")
                else:
                    try:
                        v_id = extract_video_id(url.strip())
                    except ValueError as e:
                        st.error(str(e))
                        v_id = None

                    if v_id:
                        lang_sel = st.session_state.get("lang_sel", "auto")
                        with st.spinner("Fetching transcript..."):
                            raw = get_transcript(v_id, lang=lang_sel)
                        if raw:
                            with st.spinner("Building search index..."):
                                docs = process_transcript(raw)
                                st.session_state.vectorstore = build_vectorstore(docs)
                                st.session_state.bm25        = build_bm25(docs)
                                st.session_state.docs        = docs
                                st.session_state.v_id        = v_id
                                st.session_state.messages    = []

                            import datetime
                            save_history({
                                "video_id":  v_id,
                                "title":     url.strip(),
                                "timestamp": datetime.datetime.now().strftime("%d %b, %H:%M"),
                                "last_query": ""
                            })
                            st.rerun()

            if st.session_state.vectorstore:
                st.markdown('<div class="vs-ready-badge"><div class="vs-ready-dot"></div>Ready to search</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            # Language glass card
            st.markdown('<div class="vs-glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="vs-card-label">🌐 Video Language</div>', unsafe_allow_html=True)
            lang_map = {
                "auto": "🔍 Auto Detect", "en": "🇬🇧 English",
                "hi": "🇮🇳 Hindi",       "gu": "🇮🇳 Gujarati",
                "ur": "🇵🇰 Urdu",        "ta": "🇮🇳 Tamil",
                "te": "🇮🇳 Telugu",      "bn": "🇧🇩 Bengali",
                "mr": "🇮🇳 Marathi",
            }
            lang_sel = st.selectbox("lang", list(lang_map.keys()), key="lang_sel_box",
                                    format_func=lambda x: lang_map[x],
                                    label_visibility="collapsed")
            st.session_state.lang_sel = lang_sel
            st.markdown('</div>', unsafe_allow_html=True)

            # AI Settings glass card
            st.markdown('<div class="vs-glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="vs-card-label">🤖 AI Model</div>', unsafe_allow_html=True)
            ai_mode = st.selectbox(
                "ai_mode",
                ["gemini", "both", "ollama", "none"],
                format_func=lambda x: {
                    "gemini": "✨ Gemini API (Free, Best)",
                    "both":   "🔄 Gemini + Ollama Fallback",
                    "ollama": "💻 Ollama Only (Local)",
                    "none":   "❌ Disabled"
                }[x],
                label_visibility="collapsed",
                key="ai_mode_sel"
            )
            secret_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""
            gemini_api_key = secret_key
            if ai_mode in ["gemini", "both"]:
                if secret_key:
                    st.markdown(
                        '<div style="background:rgba(16,185,129,0.06);border:1px solid rgba(16,185,129,0.15);'
                        'border-radius:8px;padding:6px 10px;font-size:0.72rem;color:#34D399;margin-top:0.4rem">'
                        '🔑 API key loaded automatically ✓</div>',
                        unsafe_allow_html=True
                    )
                else:
                    gemini_api_key = st.text_input(
                        "Gemini API Key", type="password",
                        placeholder="AIzaSy... or AQ...",
                        help="Free at aistudio.google.com",
                        key="gemini_key_input"
                    )
                    if not gemini_api_key:
                        st.markdown(
                            '🔑 <a href="https://aistudio.google.com/app/apikey" target="_blank" '
                            'style="color:#6366F1;font-size:0.75rem;text-decoration:none">Get FREE Gemini key →</a>',
                            unsafe_allow_html=True
                        )
                # Gemini model selector
                st.markdown('<div class="vs-card-label" style="margin-top:0.7rem">🧠 Gemini Model</div>', unsafe_allow_html=True)
                gemini_model = st.selectbox(
                    "gemini_model",
                    ["gemini-2.5-flash", "gemini-2.5-flash-lite"],
                    format_func=lambda x: {
                        "gemini-2.5-flash":      "⚡ Gemini 2.5 Flash (Stable, Free)",
                        "gemini-2.5-flash-lite": "💡 Gemini 2.5 Flash-Lite (Fastest, Free)",
                    }[x],
                    label_visibility="collapsed",
                    key="gemini_model_sel"
                )
            else:
                gemini_model = "gemini-2.5-flash"
            ollama_model = "llama3"
            if ai_mode in ["ollama", "both"]:
                ollama_model = st.selectbox("model",
                    ["llama3", "mistral", "llama3.2", "phi3", "gemma2"],
                    label_visibility="collapsed", key="ollama_model_sel")
            st.markdown('</div>', unsafe_allow_html=True)

            # Min match slider
            st.markdown('<br>', unsafe_allow_html=True)
            min_match = st.slider("Min match % to show results", 0, 60, 10, 5, key="min_match_slider")

        # Store settings in session
        st.session_state["_ai_mode"]        = ai_mode if "ai_mode" in dir() else "gemini"
        st.session_state["_gemini_api_key"] = gemini_api_key if "gemini_api_key" in dir() else ""
        st.session_state["_gemini_model"]   = gemini_model if "gemini_model" in dir() else "gemini-2.5-flash"
        st.session_state["_ollama_model"]   = ollama_model if "ollama_model" in dir() else "llama3"
        st.session_state["_min_match"]      = min_match if "min_match" in dir() else 10

        st.markdown('</div>', unsafe_allow_html=True)  # vs-main

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL — Chat
    # ══════════════════════════════════════════════════════════════════════════
    with right_col:
        st.markdown('<div class="vs-chat">', unsafe_allow_html=True)

        # Chat header
        ai_mode_display = st.session_state.get("_ai_mode", "gemini").title()
        st.markdown(f"""
        <div class="vs-chat-header">
            <span class="vs-chat-title">💬 Ask About The Video</span>
            <span class="vs-chat-model-badge">
                <span class="vs-chat-model-dot"></span>
                {ai_mode_display}
            </span>
        </div>
        """, unsafe_allow_html=True)

        chat_area = st.container()

        with chat_area:
            if not st.session_state.messages:
                st.markdown("""
                <div class="vs-chat-empty">
                    <div class="vs-chat-empty-ring">🎯</div>
                    <div class="vs-chat-empty-title">Process a video to start</div>
                    <div class="vs-chat-empty-sub">
                        Paste a YouTube URL in the panel<br>to the left, then ask anything.
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for m in st.session_state.messages:
                    with st.chat_message(m["role"]):
                        if m["role"] == "assistant":
                            st.markdown(m.get("html", m["content"]), unsafe_allow_html=True)
                        else:
                            st.markdown(m["content"])

        st.markdown('</div>', unsafe_allow_html=True)

        # Chat input
        if prompt := st.chat_input("Ask anything about the video..."):
            if not st.session_state.vectorstore:
                st.error("⚠️ Please process a video first.")
            else:
                _ai_mode        = st.session_state.get("_ai_mode", "gemini")
                _gemini_api_key = st.session_state.get("_gemini_api_key", "")
                _ollama_model   = st.session_state.get("_ollama_model", "llama3")
                _min_match      = st.session_state.get("_min_match", 10)

                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_area:
                    with st.chat_message("user"):
                        st.markdown(prompt)

                with st.spinner("Searching..."):
                    raw_results = hybrid_search(
                        st.session_state.vectorstore,
                        st.session_state.bm25,
                        st.session_state.docs,
                        prompt, k=8
                    )
                    reranked = rerank(prompt, raw_results, top_k=6)

                html, plain = build_results(
                    reranked,
                    st.session_state.v_id,
                    prompt,
                    ai_mode=_ai_mode,
                    gemini_api_key=_gemini_api_key,
                    ollama_model=_ollama_model,
                    min_match=_min_match,
                    lang=st.session_state.get("lang_sel", "auto"),
                    gemini_model=st.session_state.get("_gemini_model", "gemini-2.5-flash")
                )

                st.session_state.messages.append({"role": "assistant", "content": plain, "html": html})
                with chat_area:
                    with st.chat_message("assistant"):
                        st.markdown(html, unsafe_allow_html=True)

                # Update history
                if st.session_state.v_id:
                    history = load_history()
                    for h in history:
                        if h.get("video_id") == st.session_state.v_id:
                            h["last_query"] = prompt
                            break
                    try:
                        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                            json.dump(history, f, ensure_ascii=False)
                    except Exception:
                        pass

                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()



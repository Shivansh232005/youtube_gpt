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
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #07090f !important; color: #e2e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
    background: radial-gradient(ellipse 90% 50% at 50% -5%, #0a1628 0%, #07090f 55%) !important;
}

/* ── Remove ALL Streamlit top space ── */
#MainMenu, footer, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"],
[data-testid="stHeader"], header { 
    display: none !important; 
    height: 0 !important;
    min-height: 0 !important;
}
.block-container { 
    padding: 0 !important; 
    margin-top: 0 !important;
    max-width: 100% !important; 
}
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stAppViewContainer"] > section { 
    padding-top: 0 !important; 
    margin-top: 0 !important;
}
/* Remove ALL column internal padding */
[data-testid="stColumn"] {
    padding: 0 !important;
    margin: 0 !important;
}
[data-testid="stVerticalBlock"],
[data-testid="stVerticalBlockSizeWrapper"] {
    padding: 0 !important;
    margin: 0 !important;
    gap: 0 !important;
}
[data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    align-items: stretch !important;
}
/* Remove element gaps inside panels */
[data-testid="stColumn"] > div > div {
    gap: 0 !important;
}
/* Remove spacing from element containers */
div[data-testid="element-container"],
div.stMarkdown {
    margin: 0 !important;
    padding: 0 !important;
}
/* But restore padding for actual content elements */
div[data-testid="element-container"]:has(input),
div[data-testid="element-container"]:has(button),
div[data-testid="element-container"]:has(select),
div[data-testid="element-container"]:has([data-testid="stSelectbox"]),
div[data-testid="element-container"]:has([data-testid="stSlider"]) {
    padding: 0.2rem 0.8rem !important;
}

/* ── Top navbar ── */
.navbar { display: flex; align-items: center; justify-content: space-between; padding: 0.65rem 1.5rem; border-bottom: 1px solid rgba(255,255,255,0.06); background: rgba(7,9,15,0.95); position: sticky; top: 0; z-index: 100; backdrop-filter: blur(10px); }
.nav-brand { font-family: 'Syne',sans-serif; font-size: 1.15rem; font-weight: 800; background: linear-gradient(135deg,#38bdf8,#818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.nav-badges { display: flex; gap: 0.5rem; }
.nav-badge { background: rgba(56,189,248,0.08); border: 1px solid rgba(56,189,248,0.18); color: #7dd3fc; font-size: 0.62rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; padding: 0.2rem 0.7rem; border-radius: 100px; }
.nav-badge.green { background: rgba(74,222,128,0.08); border-color: rgba(74,222,128,0.2); color: #4ade80; }

/* ── Layout panels ── */
.left-panel { padding: 0.8rem 1rem 1rem 1.2rem; border-right: 1px solid rgba(255,255,255,0.05); overflow-y: auto; display: flex; flex-direction: column; gap: 0.6rem; }
.right-panel { padding: 0; display: flex; flex-direction: column; }

/* Target Streamlit columns directly for full height */
[data-testid="stColumn"]:nth-child(1) > div { border-right: 1px solid rgba(255,255,255,0.05); min-height: calc(100vh - 48px); }
[data-testid="stColumn"]:nth-child(3) > div { min-height: calc(100vh - 48px); }

/* ── Tabs ── */
.tab-bar { display: flex; border-bottom: 1px solid rgba(255,255,255,0.06); padding: 0 1rem; }
.tab-btn { padding: 0.55rem 1rem; font-size: 0.78rem; font-weight: 600; color: #475569; border: none; background: none; cursor: pointer; border-bottom: 2px solid transparent; transition: all 0.2s; font-family: 'Syne',sans-serif; letter-spacing: 0.04em; }
.tab-btn.active { color: #38bdf8; border-bottom-color: #38bdf8; }

/* ── Video box ── */
.video-placeholder { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; color: #1e293b; font-size: 0.8rem; }
[data-testid="stVideo"] { border-radius: 12px !important; overflow: hidden !important; }
[data-testid="stVideo"] iframe { border-radius: 12px !important; }

/* ── Section boxes ── */
.sbox { background: rgba(99,102,241,0.05); border: 1px solid rgba(99,102,241,0.12); border-radius: 10px; padding: 0.75rem 0.9rem; }
.slabel { font-family: 'Syne',sans-serif; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #818cf8; margin-bottom: 0.45rem; display: block; }
.flabel { font-family: 'Syne',sans-serif; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #38bdf8; margin-bottom: 0.35rem; display: block; }

/* ── Process btn & inputs ── */
[data-testid="stTextInput"] > div > div > input { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 8px !important; color: #e2e8f0 !important; font-size: 0.85rem !important; padding: 0.55rem 0.9rem !important; }
[data-testid="stTextInput"] > div > div > input:focus { border-color: rgba(56,189,248,0.4) !important; box-shadow: 0 0 0 3px rgba(56,189,248,0.06) !important; }
[data-testid="stTextInput"] label { display: none !important; }
[data-testid="stButton"] > button { width: 100% !important; background: linear-gradient(135deg,#0ea5e9,#6366f1) !important; color: white !important; border: none !important; border-radius: 8px !important; padding: 0.6rem 1rem !important; font-family: 'Syne',sans-serif !important; font-size: 0.82rem !important; font-weight: 700 !important; letter-spacing: 0.06em !important; margin-top: 0.35rem !important; }
[data-testid="stButton"] > button:hover { filter: brightness(1.1) !important; }
.pill-ready { display: inline-flex; align-items: center; gap: 0.3rem; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.2); color: #10b981; font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 100px; margin-top: 0.4rem; }
.pill-cache { display: inline-flex; align-items: center; gap: 0.3rem; background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.2); color: #f59e0b; font-size: 0.7rem; padding: 0.2rem 0.6rem; border-radius: 100px; }
[data-testid="stSelectbox"] label { display: none !important; }
[data-testid="stSelectbox"] > div > div { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important; color: #e2e8f0 !important; }
[data-testid="stSlider"] label { color: #64748b !important; font-size: 0.78rem !important; }

/* ── History sidebar — ChatGPT style ── */
.hist-panel-label { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #334155; padding: 0.5rem 0.6rem 0.4rem; display: block; }
.hist-section-date { font-size: 0.6rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: #1e293b; padding: 0.5rem 0.6rem 0.2rem; display: block; }
.hist-item { display: block; padding: 0.4rem 0.6rem; border-radius: 7px; cursor: pointer; transition: background 0.15s; margin-bottom: 1px; text-decoration: none; }
.hist-item:hover { background: rgba(255,255,255,0.05); }
.hist-item.active { background: rgba(56,189,248,0.07); }
.hist-title { font-size: 0.78rem; color: #94a3b8; font-weight: 400; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.4; }
.hist-item.active .hist-title { color: #e2e8f0; }
.hist-empty { color: #1e293b; font-size: 0.78rem; text-align: center; padding: 2rem 0.5rem; line-height: 1.8; }

/* ── Chat area ── */
.chat-scroll { flex: 1; overflow-y: auto; padding: 1rem 1.2rem; }
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0 !important; margin-bottom: 0.8rem !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: rgba(14,165,233,0.07) !important; border: 1px solid rgba(14,165,233,0.12) !important; border-radius: 11px !important; padding: 0.7rem 1rem !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { background: rgba(255,255,255,0.02) !important; border: 1px solid rgba(255,255,255,0.05) !important; border-radius: 11px !important; padding: 0.7rem 1rem !important; }
[data-testid="stChatMessage"] p { color: #e2e8f0 !important; font-size: 0.88rem !important; line-height: 1.7 !important; }
[data-testid="stChatInput"] { background: rgba(255,255,255,0.03) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 10px !important; margin: 0 1rem 0.8rem !important; }
[data-testid="stChatInput"] textarea { color: #e2e8f0 !important; }
.chat-empty { color: #1e293b; font-size: 0.9rem; text-align: center; padding: 3rem 2rem; }
.chat-empty-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }

/* ── AI answer box ── */
.ai-box { background: linear-gradient(135deg,rgba(74,222,128,0.06),rgba(14,165,233,0.03)); border: 1px solid rgba(74,222,128,0.18); border-radius: 12px; padding: 1rem 1.1rem; margin-bottom: 0.8rem; }
.ai-label { font-family: 'Syne',sans-serif; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #4ade80; margin-bottom: 0.45rem; display: flex; align-items: center; gap: 0.45rem; }
.ai-badge { background: rgba(74,222,128,0.1); border: 1px solid rgba(74,222,128,0.2); color: #4ade80; font-size: 0.6rem; padding: 0.1rem 0.5rem; border-radius: 100px; }
.ai-text { color: #cbd5e1; font-size: 0.88rem; line-height: 1.8; }

/* ── Segment cite tags (Perplexity style) ── */
.cite-tag { display: inline-flex; align-items: center; gap: 3px; background: rgba(14,165,233,0.1); border: 1px solid rgba(14,165,233,0.2); color: #38bdf8; font-size: 0.65rem; font-weight: 600; padding: 0.05rem 0.45rem; border-radius: 4px; cursor: pointer; text-decoration: none; vertical-align: middle; margin: 0 2px; }
.cite-tag:hover { background: rgba(14,165,233,0.2); }

/* ── Segment cards ── */
.seg-section { margin-top: 0.6rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 0.6rem; }
.seg-header { font-family: 'Syne',sans-serif; font-size: 0.6rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #334155; margin-bottom: 0.5rem; }
.seg-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.05); border-left: 2px solid rgba(14,165,233,0.5); border-radius: 0 10px 10px 0; padding: 0.7rem 0.9rem; margin-bottom: 0.5rem; }
.seg-meta { display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.35rem; flex-wrap: wrap; }
.seg-num { font-family: 'Syne',sans-serif; font-size: 0.58rem; font-weight: 700; color: #334155; text-transform: uppercase; }
.ts-pill { background: rgba(14,165,233,0.1); border: 1px solid rgba(14,165,233,0.2); color: #38bdf8; font-size: 0.65rem; font-weight: 500; padding: 0.1rem 0.5rem; border-radius: 5px; text-decoration: none; }
.ts-pill:hover { background: rgba(14,165,233,0.2); }
.match-high { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.2); color: #34d399; font-size: 0.62rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 5px; }
.match-mid  { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.2); color: #fbbf24; font-size: 0.62rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 5px; }
.match-low  { background: rgba(239,68,68,0.1); border: 1px solid rgba(239,68,68,0.18); color: #f87171; font-size: 0.62rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 5px; }
.seg-text { color: #64748b; font-size: 0.82rem; line-height: 1.7; }

/* ── Misc ── */
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg,#0ea5e9,#6366f1) !important; }
::-webkit-scrollbar { width: 3px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.07); border-radius: 10px; }
[data-testid="stAlert"] { border-radius: 8px !important; font-size: 0.83rem !important; }
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
    try:
        try:
            from google import genai
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "google-genai"])
            from google import genai
        client = genai.Client(api_key=api_key)
        prompt = build_prompt(question, docs, lang)
        response = client.models.generate_content(
            model=gemini_model,
            contents=prompt,
        )
        return response.text, "gemini"
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

    # AI answer
    ai_html, ai_plain = "", ""
    if ai_mode != "none":
        with st.spinner("🤖 Generating AI answer..."):
            ans, source, error = get_ai_answer(
                question, [d for d, _ in filtered],
                ai_mode, gemini_api_key, ollama_model, lang, gemini_model
            )
        if ans:
            badge_color = "#4ade80" if "Gemini" in (source or "") else "#a78bfa"
            ai_html  = (
                f'<div class="ai-box">'
                f'<div class="ai-label">🤖 AI Answer '
                f'<span style="color:{badge_color};font-size:0.6rem;margin-left:6px">▶ {source}</span>'
                f'</div>'
                f'<div class="ai-text">{ans}</div>'
                f'</div>'
            )
            ai_plain = f"**AI Answer ({source}):**\n{ans}\n\n"
        elif error:
            ai_html  = f'<div style="color:#f87171;font-size:0.8rem;margin-bottom:0.7rem">⚠️ AI Error: {error[:120]}</div>'
            ai_plain = f"AI Error: {error[:120]}\n\n"

    # Segment cards
    html  = ai_html + f'<div style="font-family:Syne,sans-serif;font-size:0.65rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#475569;padding-bottom:0.5rem;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:0.7rem">📍 {len(filtered)} Relevant Segments</div>'
    plain = ai_plain + f"**{len(filtered)} relevant segments:**\n"

    for i, (doc, rel) in enumerate(filtered, 1):
        ts       = doc.metadata.get("timestamp_range", "N/A")
        start_s  = int(doc.metadata.get("start_time", 0))
        yt_url   = f"https://www.youtube.com/watch?v={video_id}&t={start_s}s"
        text     = doc.page_content

        if rel >= 60:   mc = "match-high"; icon = "↑"
        elif rel >= 35: mc = "match-mid";  icon = "~"
        else:           mc = "match-low";  icon = "↓"

        html += (
            f'<div class="seg-card">'
            f'<div class="seg-meta">'
            f'<span class="seg-num">#{i}</span>'
            f'<a class="ts-pill" href="{yt_url}" target="_blank">⏱ {ts}</a>'
            f'<span class="{mc}">{icon} {rel:.0f}%</span>'
            f'</div>'
            f'<div class="seg-text">{text}</div>'
            f'</div>'
        )
        plain += f"\n**{i}. [{ts}]** ({rel:.0f}%)\n{text}\n"

    return html, plain

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    # ── Navbar ────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="navbar">
        <div class="nav-brand">🎬 VidSearch AI</div>
        <div class="nav-badges">
            <span class="nav-badge">⚡ Gemini AI</span>
            <span class="nav-badge">🔍 Semantic Search</span>
            <span class="nav-badge green">● Free</span>
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

    # ── 3-column layout: History | Video+Setup | Chat ─────────────────────────
    hist_col, left_col, right_col = st.columns([0.85, 1.4, 2], gap="small")

    # ══════════════════════════════════════════════════════════════════════════
    # HISTORY SIDEBAR
    # ══════════════════════════════════════════════════════════════════════════
    with hist_col:
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)
        st.markdown('<span class="hist-panel-label">VidSearch AI</span>', unsafe_allow_html=True)

        history = load_history()
        if not history:
            st.markdown('<div class="hist-empty">No history yet.<br>Process a video to get started.</div>', unsafe_allow_html=True)
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
                st.markdown(f'<span class="hist-section-date">{grp_name}</span>', unsafe_allow_html=True)

                for i, h in enumerate(groups[grp_name]):
                    vid    = h.get("video_id", "")
                    title  = h.get("title", vid)
                    is_active = vid == st.session_state.get("v_id", "")

                    if "youtube.com" in title or "youtu.be" in title:
                        display = vid
                    else:
                        display = title

                    active_cls = "active" if is_active else ""
                    st.markdown(f"""
                    <div class="hist-item {active_cls}">
                        <div class="hist-title">🎬 {display[:38]}{"…" if len(display)>38 else ""}</div>
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
    # LEFT PANEL — Video Player + Tabbed Setup/Settings
    # ══════════════════════════════════════════════════════════════════════════
    with left_col:
        st.markdown('<div class="left-panel">', unsafe_allow_html=True)

        # ── Embedded YouTube player ──────────────────────────────────────────
        if st.session_state.v_id:
            st.video(f"https://www.youtube.com/watch?v={st.session_state.v_id}")
            if st.session_state.from_cache:
                st.markdown('<div class="pill-cache">⚡ Loaded from cache</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="video-placeholder">
                <div style="text-align:center">
                    <div style="font-size:2rem;margin-bottom:0.5rem">🎬</div>
                    <div style="color:#334155;font-size:0.8rem">Video will appear here after processing</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Tabs: Video Setup | Settings ────────────────────────────────────
        tab1, tab2 = st.tabs(["⚙️ Video Setup", "🛠 Settings"])

        with tab1:
            st.markdown('<span class="flabel">YouTube URL</span>', unsafe_allow_html=True)
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

                            # Save to history
                            import datetime
                            save_history({
                                "video_id":  v_id,
                                "title":     url.strip(),
                                "timestamp": datetime.datetime.now().strftime("%d %b, %H:%M"),
                                "last_query": ""
                            })
                            st.rerun()

            if st.session_state.vectorstore:
                st.markdown('<div class="pill-ready">● Ready to search</div>', unsafe_allow_html=True)

        with tab2:
            # Language
            st.markdown('<div class="sbox">', unsafe_allow_html=True)
            st.markdown('<span class="slabel">🌐 Video Language</span>', unsafe_allow_html=True)
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

            # AI Settings
            st.markdown('<div class="sbox" style="margin-top:0.6rem">', unsafe_allow_html=True)
            st.markdown('<span class="slabel">🤖 AI Model</span>', unsafe_allow_html=True)
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
            # Auto-load from Streamlit secrets if available
            secret_key = st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else ""

            gemini_api_key = secret_key  # default to secret
            if ai_mode in ["gemini", "both"]:
                if secret_key:
                    st.markdown(
                        '<div style="background:rgba(74,222,128,0.08);border:0.5px solid '
                        'rgba(74,222,128,0.2);border-radius:6px;padding:5px 8px;font-size:0.72rem;color:#4ade80;margin-bottom:8px">'
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
                            'style="color:#4ade80;font-size:0.75rem">Get FREE key →</a>',
                            unsafe_allow_html=True
                        )
                # Gemini model selector
                st.markdown('<span class="slabel">🧠 Gemini Model</span>', unsafe_allow_html=True)
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

        # Store settings in session for chat to use
        st.session_state["_ai_mode"]        = ai_mode if "ai_mode" in dir() else "gemini"
        st.session_state["_gemini_api_key"] = gemini_api_key if "gemini_api_key" in dir() else ""
        st.session_state["_gemini_model"]   = gemini_model if "gemini_model" in dir() else "gemini-2.5-flash"
        st.session_state["_ollama_model"]   = ollama_model if "ollama_model" in dir() else "llama3"
        st.session_state["_min_match"]      = min_match if "min_match" in dir() else 10

        st.markdown('</div>', unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # RIGHT PANEL — Chat
    # ══════════════════════════════════════════════════════════════════════════
    with right_col:
        st.markdown('<div class="right-panel">', unsafe_allow_html=True)
        st.markdown('<div class="chat-scroll">', unsafe_allow_html=True)

        # Chat header
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:space-between;
             padding-bottom:0.6rem;border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:0.8rem">
            <span style="font-family:'Syne',sans-serif;font-size:0.65rem;font-weight:700;
                  letter-spacing:0.12em;text-transform:uppercase;color:#475569">
                💬 Ask About The Video
            </span>
            <span style="font-size:0.65rem;color:#1e293b">
                Powered by Gemini 1.5 Flash
            </span>
        </div>
        """, unsafe_allow_html=True)

        chat_area = st.container()

        with chat_area:
            if not st.session_state.messages:
                st.markdown("""
                <div class="chat-empty">
                    <div class="chat-empty-icon">🎯</div>
                    <div style="color:#334155;font-size:0.9rem;font-weight:600">Process a video to start</div>
                    <div style="color:#1e293b;font-size:0.78rem;margin-top:0.5rem">
                        Paste a YouTube URL → Process → Ask anything!
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
                # Retrieve settings from session
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

                # Update history with last query
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



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
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
.block-container { padding: 1.5rem 2.5rem 4rem !important; max-width: 1500px !important; }

/* Hero */
.hero { text-align: center; padding: 2rem 0 1.5rem; }
.hero-badge { display: inline-block; background: rgba(56,189,248,0.1); border: 1px solid rgba(56,189,248,0.25); color: #38bdf8; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.18em; text-transform: uppercase; padding: 0.3rem 1rem; border-radius: 100px; margin-bottom: 0.8rem; }
.hero h1 { font-family: 'Syne', sans-serif !important; font-size: clamp(2rem, 4.5vw, 3.2rem) !important; font-weight: 800 !important; background: linear-gradient(135deg, #f8fafc 20%, #38bdf8 60%, #818cf8 100%); -webkit-background-clip: text !important; -webkit-text-fill-color: transparent !important; background-clip: text !important; margin-bottom: 0.5rem !important; }
.hero p { color: #64748b; font-size: 0.95rem; }
.divider { height: 1px; background: linear-gradient(90deg,transparent,rgba(56,189,248,0.15),transparent); margin: 0.5rem 0 1.5rem; }

/* Cards & Labels */
.card { background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 1.2rem; }
.field-label { font-family: 'Syne',sans-serif; font-size: 0.68rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #38bdf8; margin-bottom: 0.4rem; display: block; }
.section-box { background: rgba(99,102,241,0.06); border: 1px solid rgba(99,102,241,0.15); border-radius: 10px; padding: 0.8rem 1rem; margin-top: 0.7rem; }
.section-label { font-family: 'Syne',sans-serif; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #818cf8; margin-bottom: 0.3rem; }
.chat-label { font-family: 'Syne',sans-serif; font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em; text-transform: uppercase; color: #475569; padding-bottom: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05); margin-bottom: 0.7rem; }

/* Input */
[data-testid="stTextInput"] > div > div > input { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 9px !important; color: #e2e8f0 !important; font-family: 'DM Sans',sans-serif !important; font-size: 0.9rem !important; padding: 0.65rem 1rem !important; }
[data-testid="stTextInput"] > div > div > input:focus { border-color: rgba(56,189,248,0.4) !important; box-shadow: 0 0 0 3px rgba(56,189,248,0.06) !important; }
[data-testid="stTextInput"] label { display: none !important; }

/* Button */
[data-testid="stButton"] > button { width: 100% !important; background: linear-gradient(135deg, #0ea5e9, #6366f1) !important; color: white !important; border: none !important; border-radius: 9px !important; padding: 0.65rem 1.5rem !important; font-family: 'Syne',sans-serif !important; font-size: 0.85rem !important; font-weight: 700 !important; letter-spacing: 0.06em !important; margin-top: 0.4rem !important; transition: all 0.2s !important; }
[data-testid="stButton"] > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 20px rgba(14,165,233,0.3) !important; }

/* Status pills */
.pill-ready { display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.25); color: #10b981; font-size: 0.72rem; font-weight: 500; padding: 0.25rem 0.7rem; border-radius: 100px; margin-top: 0.6rem; }
.pill-cache { display: inline-flex; align-items: center; gap: 0.35rem; background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.25); color: #f59e0b; font-size: 0.72rem; font-weight: 500; padding: 0.25rem 0.7rem; border-radius: 100px; margin-top: 0.3rem; }

/* Selectbox & Checkbox */
[data-testid="stSelectbox"] label { display: none !important; }
[data-testid="stSelectbox"] > div > div { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.08) !important; border-radius: 8px !important; color: #e2e8f0 !important; }
[data-testid="stCheckbox"] label { color: #94a3b8 !important; font-size: 0.86rem !important; }
[data-testid="stSlider"] label { color: #94a3b8 !important; font-size: 0.82rem !important; }

/* Chat */
[data-testid="stChatMessage"] { background: transparent !important; border: none !important; padding: 0 !important; margin-bottom: 0.8rem !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) { background: rgba(14,165,233,0.07) !important; border: 1px solid rgba(14,165,233,0.13) !important; border-radius: 11px !important; padding: 0.75rem 1rem !important; }
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) { background: rgba(255,255,255,0.025) !important; border: 1px solid rgba(255,255,255,0.06) !important; border-radius: 11px !important; padding: 0.75rem 1rem !important; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] strong { color: #e2e8f0 !important; font-family: 'DM Sans',sans-serif !important; font-size: 0.88rem !important; line-height: 1.7 !important; }
[data-testid="stChatInput"] { background: rgba(255,255,255,0.04) !important; border: 1px solid rgba(255,255,255,0.09) !important; border-radius: 11px !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: #e2e8f0 !important; font-family: 'DM Sans',sans-serif !important; }

/* Segment cards */
.seg-card { background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.07); border-left: 3px solid #0ea5e9; border-radius: 0 10px 10px 0; padding: 0.8rem 1rem; margin-bottom: 0.6rem; }
.seg-meta { display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.4rem; flex-wrap: wrap; }
.seg-num { font-family: 'Syne',sans-serif; font-size: 0.62rem; font-weight: 700; color: #334155; text-transform: uppercase; letter-spacing: 0.1em; }
.ts-pill { background: rgba(14,165,233,0.12); border: 1px solid rgba(14,165,233,0.2); color: #38bdf8; font-size: 0.7rem; font-weight: 500; padding: 0.1rem 0.5rem; border-radius: 5px; text-decoration: none; }
.ts-pill:hover { background: rgba(14,165,233,0.2); }
.match-high { background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.2); color: #34d399; font-size: 0.66rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 5px; }
.match-mid  { background: rgba(245,158,11,0.1);  border: 1px solid rgba(245,158,11,0.2);  color: #fbbf24; font-size: 0.66rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 5px; }
.match-low  { background: rgba(239,68,68,0.1);   border: 1px solid rgba(239,68,68,0.18);  color: #f87171; font-size: 0.66rem; font-weight: 600; padding: 0.1rem 0.5rem; border-radius: 5px; }
.seg-text { color: #94a3b8; font-size: 0.84rem; line-height: 1.7; }

/* AI Answer */
.ai-box { background: linear-gradient(135deg,rgba(99,102,241,0.08),rgba(14,165,233,0.05)); border: 1px solid rgba(99,102,241,0.2); border-radius: 11px; padding: 0.9rem 1.1rem; margin-bottom: 0.9rem; }
.ai-label { font-family: 'Syne',sans-serif; font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; color: #818cf8; margin-bottom: 0.4rem; }
.ai-text { color: #cbd5e1; font-size: 0.88rem; line-height: 1.75; }

/* Progress & misc */
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg,#0ea5e9,#6366f1) !important; }
::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: transparent; } ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.08); border-radius: 10px; }
[data-testid="stHorizontalBlock"] { gap: 2rem !important; }
[data-testid="stVideo"], iframe { border-radius: 11px !important; border: 1px solid rgba(255,255,255,0.07) !important; margin-top: 0.7rem !important; }
[data-testid="stAlert"] { border-radius: 9px !important; font-family: 'DM Sans',sans-serif !important; font-size: 0.84rem !important; }
</style>
""", unsafe_allow_html=True)

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
def gemini_answer(question, docs, api_key, lang="auto"):
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model  = genai.GenerativeModel("gemini-1.5-flash")
        prompt = build_prompt(question, docs, lang)
        response = model.generate_content(prompt)
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
def get_ai_answer(question, docs, ai_mode, gemini_api_key, ollama_model, lang):
    ans, source, error = None, None, None

    if ai_mode in ["gemini", "both"]:
        if gemini_api_key and gemini_api_key.strip().startswith("AIza"):
            ans, info = gemini_answer(question, docs, gemini_api_key.strip(), lang)
            if ans:
                source = "Gemini 1.5 Flash"
            else:
                error = info
        else:
            error = "Invalid or missing Gemini API key (should start with AIza...)"

    if ans is None and ai_mode in ["ollama", "both"]:
        ans, info = ollama_answer(question, docs, ollama_model, lang)
        if ans:
            source = f"Ollama ({ollama_model})"
        elif not error:
            error = info

    return ans, source, error

# ── Build result HTML ─────────────────────────────────────────────────────────
def build_results(results, video_id, question, ai_mode, gemini_api_key, ollama_model, min_match, lang="auto"):
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
                ai_mode, gemini_api_key, ollama_model, lang
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
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🎬 Local AI · No API Key · Fully Private</div>
        <h1>VidSearch AI</h1>
        <p>Semantic search + AI answers for any YouTube video</p>
    </div>
    <div class="divider"></div>
    """, unsafe_allow_html=True)

    # Session state init
    defaults = {
        "messages":   [],
        "vectorstore": None,
        "bm25":        None,
        "docs":        [],
        "v_id":        None,
        "from_cache":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    col1, col2 = st.columns([1, 1.65])

    # ── LEFT PANEL ──
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<span class="field-label">YouTube URL</span>', unsafe_allow_html=True)
        url = st.text_input("url", placeholder="https://youtube.com/watch?v=...", label_visibility="collapsed")

        if st.button("⚡ Process Video"):
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

        if st.session_state.vectorstore:
            st.markdown('<div class="pill-ready">● Ready to search</div>', unsafe_allow_html=True)
            if st.session_state.from_cache:
                st.markdown('<div class="pill-cache">⚡ Loaded from cache</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Language selector
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🌐 Video Language</div>', unsafe_allow_html=True)
        lang_map = {
            "auto": "🔍 Auto Detect",
            "en":   "🇬🇧 English",
            "hi":   "🇮🇳 Hindi",
            "gu":   "🇮🇳 Gujarati",
            "ur":   "🇵🇰 Urdu",
            "ta":   "🇮🇳 Tamil",
            "te":   "🇮🇳 Telugu",
            "bn":   "🇧🇩 Bengali",
            "mr":   "🇮🇳 Marathi",
        }
        lang_sel = st.selectbox("lang", list(lang_map.keys()),
                                format_func=lambda x: lang_map[x],
                                label_visibility="collapsed")
        st.session_state.lang_sel = lang_sel
        st.markdown("</div>", unsafe_allow_html=True)

        # AI Settings
        st.markdown('<div class="section-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">🤖 AI Answer Settings</div>', unsafe_allow_html=True)

        ai_mode = st.selectbox(
            "ai_mode",
            ["gemini", "both", "ollama", "none"],
            format_func=lambda x: {
                "gemini": "✨ Gemini API (Free, Best Quality)",
                "both":   "🔄 Gemini + Ollama (Gemini first, Ollama fallback)",
                "ollama": "💻 Ollama Only (Local, No internet)",
                "none":   "❌ No AI Answer"
            }[x],
            label_visibility="collapsed"
        )

        gemini_api_key = ""
        if ai_mode in ["gemini", "both"]:
            gemini_api_key = st.text_input(
                "Gemini API Key",
                type="password",
                placeholder="AIzaSy...",
                help="Free key at aistudio.google.com"
            )
            if not gemini_api_key:
                st.markdown(
                    '<div style="color:#f59e0b;font-size:0.75rem;margin-top:4px">'
                    '🔑 <a href="https://aistudio.google.com/app/apikey" target="_blank" '
                    'style="color:#4ade80">Get FREE Gemini API key here</a></div>',
                    unsafe_allow_html=True
                )

        ollama_model = "llama3"
        if ai_mode in ["ollama", "both"]:
            ollama_model = st.selectbox(
                "model",
                ["llama3", "mistral", "llama3.2", "phi3", "gemma2"],
                label_visibility="collapsed"
            )

        st.markdown("</div>", unsafe_allow_html=True)

        # Min match slider
        st.markdown("<br>", unsafe_allow_html=True)
        min_match = st.slider("Min match % to show results", 0, 60, 10, 5)

        # Video preview
        if st.session_state.v_id:
            st.video(f"https://www.youtube.com/watch?v={st.session_state.v_id}")

    # ── RIGHT PANEL ──
    with col2:
        st.markdown('<div class="chat-label">💬 Ask about the video</div>', unsafe_allow_html=True)
        chat_area = st.container()

        with chat_area:
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    if m["role"] == "assistant":
                        st.markdown(m.get("html", m["content"]), unsafe_allow_html=True)
                    else:
                        st.markdown(m["content"])

        if prompt := st.chat_input("Ask anything about the video..."):
            if not st.session_state.vectorstore:
                st.error("⚠️ Please process a video first.")
            else:
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
                    ai_mode=ai_mode,
                    gemini_api_key=gemini_api_key,
                    ollama_model=ollama_model,
                    min_match=min_match,
                    lang=st.session_state.get("lang_sel", "auto")
                )

                st.session_state.messages.append({"role": "assistant", "content": plain, "html": html})
                with chat_area:
                    with st.chat_message("assistant"):
                        st.markdown(html, unsafe_allow_html=True)

                st.rerun()

if __name__ == "__main__":
    main()

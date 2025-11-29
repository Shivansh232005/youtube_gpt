import streamlit as st
import re
import warnings
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from typing import List, Dict, Any


# Suppress warnings
warnings.filterwarnings("ignore")


def extract_video_id(youtube_url):
    """Extract YouTube video ID from URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/)([a-zA-Z0-9_-]{11})',
        r'([a-zA-Z0-9_-]{11})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            video_id = match.group(1).strip()
            if len(video_id) == 11:
                return video_id
    raise ValueError("Could not extract YouTube video ID from URL")


def get_transcript(video_id):
    """Get transcript with timestamps from YouTube video"""
    try:
        transcript_api = YouTubeTranscriptApi()
        transcript = transcript_api.fetch(video_id)
        return transcript
    except Exception as e:
        try:
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            return transcript
        except Exception as e2:
            st.error(f"Error fetching transcript: {str(e2)}")
            return None


def format_timestamp(seconds):
    """Convert seconds to MM:SS format"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def process_transcript_advanced(transcript):
    """Process transcript with better chunking for semantic search"""
    if not transcript:
        return []
    
    documents = []
    transcript_list = []
    
    # Normalize transcript format
    for entry in transcript:
        if hasattr(entry, 'text'):
            transcript_entry = {
                'text': entry.text,
                'start': entry.start,
                'duration': getattr(entry, 'duration', 0)
            }
        elif isinstance(entry, dict):
            transcript_entry = entry
        else:
            continue
        transcript_list.append(transcript_entry)
    
    # Create overlapping chunks for better context
    window_size = 5  # Number of segments per chunk
    overlap = 2      # Overlap between chunks
    
    for i in range(0, len(transcript_list), window_size - overlap):
        text_parts = []
        start_time = transcript_list[i].get('start', 0)
        end_idx = min(i + window_size, len(transcript_list))
        
        for j in range(i, end_idx):
            text = transcript_list[j].get('text', '').strip()
            if text:
                text_parts.append(text)
        
        if text_parts:
            combined_text = " ".join(text_parts)
            
            # Calculate end time
            end_time = transcript_list[end_idx - 1].get('start', start_time) + \
                      transcript_list[end_idx - 1].get('duration', 0)
            
            doc = Document(
                page_content=combined_text,
                metadata={
                    'start_time': start_time,
                    'end_time': end_time,
                    'timestamp': format_timestamp(start_time),
                    'timestamp_range': f"{format_timestamp(start_time)} - {format_timestamp(end_time)}",
                    'segment_count': len(text_parts)
                }
            )
            documents.append(doc)
    
    return documents


def setup_semantic_search(documents):
    """Setup semantic search system using FAISS (NO LLM required)"""
    try:
        if not documents:
            st.error("No documents to process")
            return None
        
        # Use FREE HuggingFace embeddings (runs locally, no API calls needed)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        st.info("🔄 Creating embeddings locally (this may take a moment on first run)...")
        
        # Create FAISS vector store
        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        st.success("✅ Vector database created successfully!")
        
        return vectorstore
        
    except Exception as e:
        st.error(f"Error setting up semantic search: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return None


def semantic_search_answer(vectorstore, question, k=5):
    """
    Perform semantic search and format results as an answer.
    No LLM needed - just retrieve and format the most relevant segments.
    """
    try:
        # Perform similarity search
        docs = vectorstore.similarity_search_with_score(question, k=k)
        
        if not docs:
            return "No relevant information found for your question.", []
        
        # Format the answer from retrieved documents
        answer_parts = []
        source_docs = []
        
        answer_parts.append(f"**Based on the video content, here are {len(docs)} most relevant segments:**\n")
        
        for i, (doc, score) in enumerate(docs, 1):
            timestamp = doc.metadata.get('timestamp_range', doc.metadata.get('timestamp', 'N/A'))
            content = doc.page_content.strip()
            
            # Calculate relevance percentage (cosine similarity converted to percentage)
            relevance = max(0, min(100, (1 - score) * 100))
            
            answer_parts.append(f"\n**{i}. [{timestamp}]** (Relevance: {relevance:.0f}%)")
            answer_parts.append(f"{content}\n")
            
            source_docs.append(doc)
        
        answer_parts.append("\n💡 **Tip:** Click on timestamps to jump to that part in the video!")
        
        return "\n".join(answer_parts), source_docs
        
    except Exception as e:
        return f"Error performing search: {str(e)}", []


def get_video_summary(vectorstore, num_segments=8):
    """Generate a summary by retrieving diverse segments from the video"""
    try:
        # Search for general summary-related queries
        summary_queries = [
            "main points and key takeaways",
            "important information and conclusions",
            "summary overview"
        ]
        
        all_docs = []
        seen_timestamps = set()
        
        for query in summary_queries:
            docs = vectorstore.similarity_search(query, k=3)
            for doc in docs:
                timestamp = doc.metadata.get('timestamp', '')
                if timestamp not in seen_timestamps:
                    all_docs.append(doc)
                    seen_timestamps.add(timestamp)
        
        # Sort by timestamp
        all_docs = sorted(all_docs, key=lambda x: x.metadata.get('start_time', 0))
        
        # Take top segments
        all_docs = all_docs[:num_segments]
        
        if not all_docs:
            return "Unable to generate summary."
        
        summary_parts = ["**Video Summary (Key Segments):**\n"]
        
        for i, doc in enumerate(all_docs, 1):
            timestamp = doc.metadata.get('timestamp_range', doc.metadata.get('timestamp', 'N/A'))
            content = doc.page_content.strip()
            
            summary_parts.append(f"\n**{i}. [{timestamp}]**")
            summary_parts.append(f"{content}\n")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        return f"Error generating summary: {str(e)}"


def main():
    st.set_page_config(
        page_title="YouTube Semantic Search",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 YouTube Transcript Semantic Search")
    st.markdown("**Powered by 100% FREE Local Embeddings - No LLM, No API Costs!**")
    
    # Initialize session state
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "transcript" not in st.session_state:
        st.session_state.transcript = None
    if "video_id" not in st.session_state:
        st.session_state.video_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        st.success("✅ 100% Local - No API Key Required!")
        
        st.divider()
        
        st.subheader("✨ Features")
        st.markdown("""
        - 🔍 **Pure Semantic Search** (No LLM)
        - 🚀 **Lightning Fast** responses
        - 💰 **Completely FREE**
        - 🔒 **100% Private** (runs locally)
        - 📊 **FAISS Vector Search**
        - 🎯 **Semantic Understanding**
        - ⚡ **No API Limits**
        - 🌐 **Works Offline** (after model download)
        """)
        
        st.divider()
        
        st.subheader("💡 Question Examples")
        st.markdown("""
        - "What are the main points?"
        - "Explain the key concepts"
        - "What advice is given?"
        - "How to do X?"
        - "Tell me about Y"
        - "What's discussed at 5:30?"
        """)
        
        st.divider()
        
        if st.session_state.vectorstore:
            st.subheader("📊 Vector DB Stats")
            try:
                doc_count = len(st.session_state.vectorstore.docstore._dict)
                st.metric("Documents Indexed", doc_count)
                st.success("✅ Ready for Search")
            except:
                st.info("Vector store active")
        
        st.divider()
        st.subheader("🔬 How It Works")
        st.markdown("""
        1. **Extract** transcript from YouTube
        2. **Embed** text using local AI model
        3. **Index** with FAISS vector database
        4. **Search** semantically (meaning-based)
        5. **Retrieve** most relevant segments
        
        No LLM = Faster, Cheaper, More Reliable!
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📺 Video Input")
        
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://www.youtube.com/watch?v=VIDEO_ID",
            help="Enter the complete YouTube video URL"
        )
        
        if st.button("🚀 Load & Process Video", type="primary", use_container_width=True):
            if not youtube_url.strip():
                st.error("❌ Please enter a YouTube URL")
            else:
                # Extract video ID
                with st.spinner("🔍 Extracting video ID..."):
                    try:
                        video_id = extract_video_id(youtube_url)
                        st.session_state.video_id = video_id
                        st.success(f"✅ Video ID: `{video_id}`")
                    except ValueError as e:
                        st.error(f"❌ {str(e)}")
                        st.stop()
                
                # Fetch transcript
                with st.spinner("📥 Fetching transcript..."):
                    transcript = get_transcript(video_id)
                    if transcript is None:
                        st.error("❌ Could not fetch transcript")
                        st.stop()
                    
                    st.session_state.transcript = transcript
                    st.success(f"✅ Loaded {len(transcript)} segments")
                
                # Process transcript
                with st.spinner("⚙️ Processing transcript..."):
                    documents = process_transcript_advanced(transcript)
                    if not documents:
                        st.error("❌ No valid content found")
                        st.stop()
                    
                    st.success(f"✅ Created {len(documents)} searchable chunks")
                
                # Setup semantic search (NO LLM!)
                with st.spinner("🤖 Setting up semantic search (first run downloads ~90MB model)..."):
                    vectorstore = setup_semantic_search(documents)
                    if vectorstore is None:
                        st.error("❌ Failed to setup semantic search")
                        st.stop()
                    
                    st.session_state.vectorstore = vectorstore
                    st.success("🎉 Semantic Search Ready!")
                    
                    # Initialize chat
                    st.session_state.messages = [{
                        "role": "assistant", 
                        "content": "Hello! I've indexed the video using semantic search. Ask me anything and I'll find the most relevant segments instantly! 🚀\n\n*Note: I retrieve actual video segments without generating text, so responses are fast and accurate.*"
                    }]
        
        # Video info
        if st.session_state.transcript:
            st.markdown("---")
            st.subheader("📊 Video Analysis")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Video ID", st.session_state.video_id[:8] + "...")
            
            with col_b:
                st.metric("Segments", len(st.session_state.transcript))
            
            with col_c:
                if st.session_state.transcript:
                    last_segment = st.session_state.transcript[-1]
                    if isinstance(last_segment, dict) and 'start' in last_segment:
                        duration = format_timestamp(last_segment['start'])
                    else:
                        duration = "Unknown"
                    st.metric("Duration", f"~{duration}")
            
            # Embed video
            st.markdown("---")
            st.subheader("🎥 Video Preview")
            video_url = f"https://www.youtube.com/embed/{st.session_state.video_id}"
            st.markdown(f'<iframe width="100%" height="315" src="{video_url}" frameborder="0" allowfullscreen></iframe>', 
                       unsafe_allow_html=True)
    
    with col2:
        st.subheader("💬 Semantic Search Q&A")
        
        # Display messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask anything about the video..."):
            if not st.session_state.vectorstore:
                st.warning("⚠️ Please load a video first")
            else:
                # Add user message
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Perform semantic search (instant!)
                with st.chat_message("assistant"):
                    with st.spinner("🔍 Searching semantic database..."):
                        try:
                            # Get answer using semantic search only
                            answer, source_docs = semantic_search_answer(
                                st.session_state.vectorstore, 
                                prompt, 
                                k=5
                            )
                            
                            # Display answer
                            st.markdown(answer)
                            
                            # Show advanced options in expander
                            if source_docs:
                                with st.expander("⚙️ Search Details"):
                                    st.markdown("**Search Algorithm:** Cosine Similarity (FAISS)")
                                    st.markdown(f"**Embedding Model:** sentence-transformers/all-MiniLM-L6-v2")
                                    st.markdown(f"**Results Returned:** {len(source_docs)}")
                                    st.markdown(f"**Vector Dimension:** 384")
                            
                            # Add to chat history
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": answer
                            })
                            
                        except Exception as e:
                            error_msg = f"Error performing search: {str(e)}"
                            st.error(error_msg)
                            
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": "I encountered an error. Please try again."
                            })
        
        # Quick action buttons
        if st.session_state.vectorstore:
            st.markdown("---")
            st.subheader("🎯 Quick Actions")
            
            col_q1, col_q2, col_q3 = st.columns(3)
            
            with col_q1:
                if st.button("📋 Main Points", use_container_width=True):
                    try:
                        with st.spinner("Searching..."):
                            answer, _ = semantic_search_answer(
                                st.session_state.vectorstore,
                                "main points key takeaways important information",
                                k=6
                            )
                            st.session_state.messages.append({"role": "user", "content": "What are the main points?"})
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with col_q2:
                if st.button("💡 Summary", use_container_width=True):
                    try:
                        with st.spinner("Generating summary..."):
                            answer = get_video_summary(st.session_state.vectorstore, num_segments=6)
                            st.session_state.messages.append({"role": "user", "content": "Summarize the video"})
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            
            with col_q3:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.rerun()


if __name__ == "__main__":
    main()

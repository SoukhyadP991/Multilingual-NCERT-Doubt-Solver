import streamlit as st
import time
import os
from src.pipeline import RAGPipeline
import config

# Page Config
st.set_page_config(
    page_title="NCERT Doubt Solver",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Pipeline (Cached to avoid reloading model)
# Initialize Pipeline (Cached to avoid reloading model)
# Renamed to force cache invalidation after code updates
@st.cache_resource
def get_pipeline_v3():
    return RAGPipeline()

try:
    rag_pipeline = get_pipeline_v3()
except Exception as e:
    st.error(f"Failed to initialize RAG Pipeline: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.title("📚 Configuration")
    
    st.markdown("### Filters")
    selected_grade = st.selectbox("Select Grade", ["All", "Grade 5", "Grade 6", "Grade 7", "Grade 8", "Grade 9", "Grade 10"])
    selected_subject = st.selectbox("Select Subject", ["All", "Science", "Maths", "Social Science", "English", "Hindi"])
    
    st.markdown("---")
    st.markdown("### System Status")
    if rag_pipeline.retriever.vector_store:
        st.success("Vector DB Loaded")
    else:
        st.error("Vector DB Not Found")
        
    if rag_pipeline.generator.llm:
        st.success("LLM Loaded")
    else:
        st.warning("LLM Not Loaded (Check models/)")

    st.markdown("---")
    st.info("Note: Ensure NCERT PDFs are in `data/raw` and `ingestion.py` has been run.")

# Main Interface
st.title("🤖 Multilingual NCERT Doubt Solver")
st.markdown("Ask questions from your NCERT textbooks. Answers are strictly grounded in the content.")

# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "latency" in message:
            st.caption(f"Latency: {message['latency']:.2f}s | Language: {message.get('language', 'Unknown')}")
        if "sources" in message:
            with st.expander("View Sources"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:** {doc.metadata.get('source', 'Unknown')} (Page {doc.metadata.get('page', 'Unknown')})")
                    st.markdown(f"> {doc.page_content[:300]}...")

# User Input
if prompt := st.chat_input("What is your doubt?"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Answer
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        # Prepare filters (mapping UI to metadata keys if applicable)
        filters = {}
        if selected_grade != "All":
            filters["grade"] = selected_grade
        if selected_subject != "All":
            filters["subject"] = selected_subject
            
        # Run Pipeline
        try:
            result = rag_pipeline.process_query(prompt, filters=filters)
            answer = result["answer"]
            sources = result["source_documents"]
            latency = result["latency"]
            lang = result["language"]
            
            message_placeholder.markdown(answer)
            
            # Feedback
            # Use small columns for buttons to keep them close
            col1, col2, col3 = st.columns([1, 1, 10])
            with col1:
                st.button("👍", key=f"up_{len(st.session_state.messages)}")
            with col2:
                st.button("👎", key=f"down_{len(st.session_state.messages)}")

            # Update history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "latency": latency,
                "language": lang
            })
            
        except Exception as e:
            message_placeholder.error(f"An error occurred: {e}")

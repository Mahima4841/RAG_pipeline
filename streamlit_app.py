import streamlit as st
import sys
import os
from query_rag import query_rag, print_wrapped, generate_response
import textwrap

# Set page config
st.set_page_config(
    page_title="RAG Query System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .query-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .score-box {
        background-color: #fff3cd;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #ffc107;
    }
    .response-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 RAG Query System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("📚 About")
        st.markdown("""
        This RAG (Retrieval-Augmented Generation) system allows you to query the UFS (Universal Flash Storage) documentation.
        
        **How it works:**
        1. Enter your question in the text box
        2. The system finds the most relevant text chunks
        3. An AI model generates a response based on the context
        
        **Features:**
        - 🔍 Semantic search using embeddings
        - 🤖 AI-powered response generation
        - 📊 Similarity scoring
        - 💾 Persistent storage with ChromaDB
        """)
        
        st.header("⚙️ Settings")
        top_k = st.slider("Number of top results", min_value=1, max_value=5, value=1)
        
        st.header("📖 Sample Questions")
        sample_questions = [
            "What is RPMB region in UFS?",
            "How does UFS boot work?",
            "What are the security features of UFS?",
            "Explain UFS logical units",
            "What is the RPMB well-known logical unit?"
        ]
        
        for question in sample_questions:
            if st.button(question, key=question):
                st.session_state.query = question
                st.rerun()
    
    # Main content
    st.markdown('<div class="query-box">', unsafe_allow_html=True)
    
    # Query input
    query = st.text_input(
        "Enter your question:",
        value=st.session_state.get('query', ''),
        placeholder="e.g., What is RPMB region in UFS?",
        key="query_input"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process query
    if st.button("🔍 Search", type="primary", use_container_width=True):
        if query.strip():
            with st.spinner("Searching for relevant information..."):
                try:
                    # Query the RAG system
                    results = query_rag(query, top_k=top_k)
                    
                    if results is None:
                        st.error("❌ Failed to load embeddings. Please run `generate_embeddings.py` first.")
                        return
                    
                    top_results_dot_product, documents = results
                    
                    st.markdown('<div class="result-box">', unsafe_allow_html=True)
                    st.subheader("📋 Search Results")
                    
                    for i, (score, idx) in enumerate(zip(top_results_dot_product[0][0], top_results_dot_product[1][0])):
                        st.markdown(f'<div class="score-box">', unsafe_allow_html=True)
                        st.markdown(f"**Result {i+1}** - Similarity Score: {score:.4f}")
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Display context
                        context = documents[idx]
                        st.markdown("**Context:**")
                        st.text_area(
                            f"Context {i+1}",
                            value=textwrap.fill(context, width=80),
                            height=150,
                            key=f"context_{i}",
                            disabled=True
                        )
                        
                        # Generate response
                        with st.spinner(f"Generating response for result {i+1}..."):
                            response = generate_response(context, query)
                            
                            st.markdown('<div class="response-box">', unsafe_allow_html=True)
                            st.markdown("**🤖 AI Response:**")
                            st.write(response)
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"❌ An error occurred: {str(e)}")
                    st.info("💡 Make sure you have run `generate_embeddings.py` first to create the embeddings.")
        else:
            st.warning("⚠️ Please enter a question.")

if __name__ == "__main__":
    main() 
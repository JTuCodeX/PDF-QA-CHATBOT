import streamlit as st
import tempfile, os
from main import process_file_and_build_index, run_qa_mode
from processing.embeddings import build_index

st.set_page_config(page_title="PDF Q&A Chatbot", layout="wide")
st.title("📚 PDF Q&A Chatbot")

# --- File Upload ---
uploaded_files = st.file_uploader(
    "Upload one or more PDF/TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    # Rebuild index only if uploaded files change
    if (
        "uploaded_names" not in st.session_state
        or [f.name for f in uploaded_files] != st.session_state.uploaded_names
    ):
        with st.spinner("⏳ Processing uploaded files..."):
            try:
                all_chunks = []
                for file in uploaded_files:
                    text, chunks, _ = process_file_and_build_index(file)
                    all_chunks.extend(chunks)

                if not all_chunks:
                    st.error("❌ No text extracted from uploaded files. Try another PDF/TXT.")
                    st.stop()

                st.session_state.index = build_index(all_chunks)
                st.session_state.chunks = all_chunks
                st.session_state.uploaded_names = [f.name for f in uploaded_files]

                # reset chat history
                st.session_state.history = []
                st.success("✅ Files processed successfully! You can now start asking questions.")

            except Exception as e:
                st.error(f"Error processing file(s): {e}")
                st.stop()

    # --- Q&A Chatbot ---
    st.subheader("💬 Ask Questions")

    if "history" not in st.session_state:
        st.session_state.history = []

    # Clear conversation button
    if st.button("🗑️ Clear Conversation"):
        st.session_state.history = []
        st.success("Conversation cleared!")

    # Display chat history
    for q, ans, ctx in st.session_state.history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(ans)
            with st.expander("Show supporting context"):
                for i, passage in enumerate(ctx, 1):
                    st.markdown(f"**Passage {i}:** {passage}")

    # Input box always comes last
    query = st.chat_input("Ask a question...")
    if query:
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = run_qa_mode(
                    st.session_state.index,
                    query,
                    history=st.session_state.history  # ✅ pass history here
                )
            st.markdown(result["answer"])
            with st.expander("Show supporting context"):
                for i, passage in enumerate(result["context"], 1):
                    st.markdown(f"**Passage {i}:** {passage}")

        # Append to history
        st.session_state.history.append((query, result["answer"], result["context"]))

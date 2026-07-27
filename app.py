import streamlit as st
import os
import tempfile
import fitz
from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL_NAME, MAX_PAGES, DEFAULT_GROQ_API_KEY
from src.ingest import load_pdf_pages, chunk_pages
from src.embed import build_faiss_index
from src.retrieve import retrieve_top_k
from src.generate import (
    generate_answer,
    generate_summary,
    generate_practice_questions,
    generate_flashcards,
    group_chunks_by_pages,
)

st.set_page_config(page_title="Notes RAG Assistant", layout="wide")
st.title("📚 Notes RAG Assistant")
st.caption("Upload a PDF and ask questions, get summaries, practice questions, or flashcards — all grounded in your document.")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def process_pdf(pdf_path: str):
    pages = load_pdf_pages(pdf_path)
    chunks = chunk_pages(pages)
    model = load_embedding_model()
    index = build_faiss_index(chunks, model)
    return chunks, index, model


# ---------- Sidebar: API Key ----------
with st.sidebar:
    st.subheader("🔑 API Key")
    user_api_key = st.text_input(
        "Your Groq API key (optional)",
        type="password",
        help="Leave blank to use the demo key. For heavy use, get your own free key at console.groq.com/keys"
    )
    st.session_state.groq_api_key = user_api_key if user_api_key else DEFAULT_GROQ_API_KEY
    st.divider()

    # ---------- Sidebar: Upload ----------
    st.header("Upload your PDF")
    uploaded_file = st.file_uploader("Choose a PDF (max 100 pages)", type="pdf")

    if uploaded_file is not None:
        if "processed_filename" not in st.session_state or st.session_state.processed_filename != uploaded_file.name:
            with st.spinner("Checking PDF..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                page_count = fitz.open(tmp_path).page_count

                if page_count > MAX_PAGES:
                    st.error(f"❌ This PDF has {page_count} pages. Please upload a PDF with {MAX_PAGES} pages or fewer.")
                    os.unlink(tmp_path)
                else:
                    with st.spinner("Processing PDF..."):
                        try:
                            chunks, index, model = process_pdf(tmp_path)
                        except ValueError as e:
                            st.error(str(e))
                            os.unlink(tmp_path)
                            st.stop()

                        st.session_state.chunks = chunks
                        st.session_state.index = index
                        st.session_state.model = model
                        st.session_state.processed_filename = uploaded_file.name
                        st.session_state.pop("practice_qs", None)
                        st.session_state.pop("flashcards", None)
                        st.session_state.pop("qa_answer", None)
                        st.session_state.pop("summary_text", None)

                    os.unlink(tmp_path)
                    st.success(f"Processed '{uploaded_file.name}' — {len(chunks)} chunks created.")
        else:
            st.info(f"'{uploaded_file.name}' already processed.")


# ---------- Main area ----------
if "chunks" not in st.session_state:
    st.info("👈 Upload a PDF from the sidebar to get started.")
else:
    chunks = st.session_state.chunks
    index = st.session_state.index
    model = st.session_state.model

    tab1, tab2, tab3, tab4 = st.tabs(["💬 Ask a Question", "📝 Summary", "❓ Practice Questions", "🗂️ Flashcards"])

    # --- Tab 1: Q&A ---
    with tab1:
        query = st.text_input("Ask a question about your document:")
        st.caption("💡 Ask specific questions for more accurate answers.")

        if "qa_generating" not in st.session_state:
            st.session_state.qa_generating = False

        ask_clicked = st.button("Get Answer", key="qa_btn", disabled=st.session_state.qa_generating)

        if ask_clicked:
            if query.strip():
                st.session_state.qa_generating = True
                st.session_state.qa_query = query
                st.rerun()
            else:
                st.warning("Please enter a question first.")

        if st.session_state.qa_generating:
            with st.spinner("Retrieving relevant sections and generating answer..."):
                top_chunks = retrieve_top_k(st.session_state.qa_query, index, chunks, model, k=6)
                st.session_state.qa_answer = generate_answer(
                    st.session_state.qa_query, top_chunks, api_key=st.session_state.groq_api_key
                )
            st.session_state.qa_generating = False
            st.rerun()

        if "qa_answer" in st.session_state:
            st.markdown(st.session_state.qa_answer)

    # --- Tab 2: Summary ---
    with tab2:
        length = st.radio("Summary length:", ["short", "medium", "detailed"], index=1, horizontal=True)

        if "summary_generating" not in st.session_state:
            st.session_state.summary_generating = False

        summary_clicked = st.button("Generate Summary", key="summary_btn", disabled=st.session_state.summary_generating)

        if summary_clicked:
            st.session_state.summary_generating = True
            st.session_state.summary_length = length
            st.rerun()

        if st.session_state.summary_generating:
            with st.spinner("Generating summary..."):
                st.session_state.summary_text = generate_summary(
                    chunks, api_key=st.session_state.groq_api_key, length=st.session_state.summary_length
                )
            st.session_state.summary_generating = False
            st.rerun()

        if "summary_text" in st.session_state:
            st.markdown(st.session_state.summary_text)

    # --- Tab 3: Practice Questions (section-wise) ---
    with tab3:
        st.caption("⚠️ Questions and answers are AI-generated — occasionally may contain minor inaccuracies. Cross-check key concepts.")
        sections = group_chunks_by_pages(chunks, pages_per_section=20)
        section_labels = [s["label"] for s in sections]

        selected_label = st.selectbox("Choose a section:", section_labels, key="practice_section")
        selected_section = next(s for s in sections if s["label"] == selected_label)

        num_q = st.slider("Number of questions:", 3, 10, 5)

        if "practice_generating" not in st.session_state:
            st.session_state.practice_generating = False

        regenerate = st.button(
            "🔄 Generate New Practice Questions",
            key="practice_btn",
            disabled=st.session_state.practice_generating
        )

        if regenerate:
            st.session_state.practice_generating = True
            st.rerun()

        if st.session_state.practice_generating:
            with st.spinner(f"Generating fresh practice questions for {selected_label}..."):
                seed_note = f"Random seed: {os.urandom(4).hex()}"
                st.session_state.practice_qs = generate_practice_questions(
                    selected_section["chunks"], api_key=st.session_state.groq_api_key,
                    num_questions=num_q, seed_note=seed_note
                )
                st.session_state.practice_qs_label = selected_label
            st.session_state.practice_generating = False
            st.rerun()

        if "practice_qs" in st.session_state:
            st.caption(f"Showing questions for {st.session_state.practice_qs_label}")
            for i, q in enumerate(st.session_state.practice_qs):
                st.markdown(f"**Q{i+1}: {q['question']}**")
                st.markdown(f"{q['answer']}")
                st.markdown("---")
        else:
            st.info("Select a section and click the button above to generate practice questions.")

    # --- Tab 4: Flashcards (section-wise) ---
    with tab4:
        st.caption("⚠️ Flashcards are AI-generated — occasionally may contain minor inaccuracies. Cross-check key concepts.")
        sections = group_chunks_by_pages(chunks, pages_per_section=20)
        section_labels = [s["label"] for s in sections]

        selected_label_fc = st.selectbox("Choose a section:", section_labels, key="flashcard_section")
        selected_section_fc = next(s for s in sections if s["label"] == selected_label_fc)

        num_cards = st.slider("Number of flashcards:", 5, 12, 8)

        if "flashcards_generating" not in st.session_state:
            st.session_state.flashcards_generating = False

        regenerate_cards = st.button(
            "🔄 Generate New Flashcards",
            key="flashcards_btn",
            disabled=st.session_state.flashcards_generating
        )

        if regenerate_cards:
            st.session_state.flashcards_generating = True
            st.rerun()

        if st.session_state.flashcards_generating:
            with st.spinner(f"Generating fresh flashcards for {selected_label_fc}..."):
                seed_note = f"Random seed: {os.urandom(4).hex()}"
                st.session_state.flashcards = generate_flashcards(
                    selected_section_fc["chunks"], api_key=st.session_state.groq_api_key,
                    num_cards=num_cards, seed_note=seed_note
                )
                st.session_state.flashcards_label = selected_label_fc
            st.session_state.flashcards_generating = False
            st.rerun()

        if "flashcards" in st.session_state:
            st.caption(f"Showing flashcards for {st.session_state.flashcards_label}")

            st.markdown("""
                <style>
                .flip-card {
                    background-color: transparent;
                    width: 100%;
                    height: 160px;
                    perspective: 1000px;
                    margin-bottom: 20px;
                }
                .flip-card-inner {
                    position: relative;
                    width: 100%;
                    height: 100%;
                    text-align: center;
                    transition: transform 0.6s;
                    transform-style: preserve-3d;
                    cursor: pointer;
                }
                .flip-card:hover .flip-card-inner {
                    transform: rotateY(180deg);
                }
                .flip-card-front, .flip-card-back {
                    position: absolute;
                    width: 100%;
                    height: 100%;
                    backface-visibility: hidden;
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 15px;
                    font-size: 16px;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                }
                .flip-card-front {
                    background-color: #4a4a8a;
                    color: white;
                    font-weight: bold;
                }
                .flip-card-back {
                    background-color: #2e2e52;
                    color: white;
                    transform: rotateY(180deg);
                    font-size: 14px;
                }
                </style>
            """, unsafe_allow_html=True)

            cols = st.columns(2)
            for i, card in enumerate(st.session_state.flashcards):
                with cols[i % 2]:
                    st.markdown(f"""
                        <div class="flip-card">
                            <div class="flip-card-inner">
                                <div class="flip-card-front">{card['term']}</div>
                                <div class="flip-card-back">{card['definition']}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Select a section and click the button above to generate flashcards.")
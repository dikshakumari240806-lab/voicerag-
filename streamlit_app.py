import os
import tempfile
import streamlit as st

from app.rag import RAG
from app.stt import Transcriber


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="VoiceRAG AI",
    page_icon="🎙️",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "rag" not in st.session_state:
    st.session_state.rag = None

if "indexed_file" not in st.session_state:
    st.session_state.indexed_file = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "voice_question" not in st.session_state:
    st.session_state.voice_question = ""


# ============================================================
# HEADER
# ============================================================

st.title("🎙️ VoiceRAG AI")

st.caption(
    "Voice-powered Retrieval Augmented Generation"
)

st.success(
    "✅ VoiceRAG AI is running successfully!"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    language = st.selectbox(
        "🌐 Language",
        [
            "English",
            "Hindi",
            "Hinglish"
        ]
    )

    st.divider()

    st.subheader("📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload your document",
        type=[
            "pdf",
            "txt",
            "docx"
        ],
        help="Upload a PDF, TXT or DOCX document."
    )

    if uploaded_file:

        st.success(
            f"📄 {uploaded_file.name}"
        )

        st.write(
            f"Size: {uploaded_file.size / 1024:.1f} KB"
        )

        # ----------------------------------------------------
        # INDEX DOCUMENT
        # ----------------------------------------------------

        if st.button(
            "🧠 Index Document",
            use_container_width=True
        ):

            with st.spinner(
                "📚 Reading and indexing document..."
            ):

                try:

                    # Create data folder
                    os.makedirs(
                        "data",
                        exist_ok=True
                    )

                    # Save uploaded document
                    file_path = os.path.join(
                        "data",
                        uploaded_file.name
                    )

                    with open(
                        file_path,
                        "wb"
                    ) as f:

                        f.write(
                            uploaded_file.getbuffer()
                        )

                    # Create RAG
                    st.session_state.rag = RAG()

                    # Ingest document
                    result = st.session_state.rag.ingest(
                        directory="data",
                        strategy="all"
                    )

                    st.session_state.indexed_file = (
                        uploaded_file.name
                    )

                    # Clear previous answer
                    st.session_state.answer = None

                    st.success(
                        "✅ Document indexed successfully!"
                    )

                    st.info(
                        f"📄 Documents: "
                        f"{result['documents']}\n\n"
                        f"🧩 Chunks: "
                        f"{result['chunks']}"
                    )

                except Exception as e:

                    st.session_state.rag = None

                    st.error(
                        f"❌ Indexing failed:\n\n{e}"
                    )


# ============================================================
# MAIN LAYOUT
# ============================================================

col1, col2 = st.columns(
    [2, 1]
)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    st.subheader(
        "🎤 Ask VoiceRAG"
    )

    st.info(
        "Upload and index your document, "
        "then ask a question using your voice "
        "or type it manually."
    )


# ==================================================
# VOICE INPUT
# ==================================================

st.markdown("### 🎙️ Voice Input")
st.caption("🎤 Click the microphone and speak your question.")

audio = st.audio_input(
    "🎤 Click and speak your question",
    key="voice_recording"
)

if audio is not None:

    st.audio(audio, format="audio/wav")

    with st.spinner("🎧 Converting your voice to text..."):

        temp_path = None

        try:
            # Save recorded audio
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as tmp:
                tmp.write(audio.getvalue())
                temp_path = tmp.name

            # Import transcriber
            from app.stt import Transcriber

            transcriber = Transcriber()

            # Convert voice → text
            transcript, confidence = transcriber.transcribe_file(
                temp_path
            )

            if transcript and transcript.strip():

                st.success("✅ Voice converted successfully!")

                # IMPORTANT:
                # Save transcript so it appears in question box
                st.session_state.voice_question = transcript.strip()

                st.markdown("### 📝 Your Voice Text")

                st.text_area(
                    "Transcribed Question",
                    value=st.session_state.voice_question,
                    height=100,
                    key="transcribed_text"
                )

            else:
                st.warning(
                    "⚠️ I couldn't understand the audio. "
                    "Please speak clearly and try again."
                )

        except Exception as e:

            st.error(
                f"❌ Voice processing error: {e}"
            )

        finally:

            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass


# ==================================================
# TEXT QUESTION
# ==================================================

st.markdown("### 📝 Your Question")

question = st.text_area(
    "Ask something about your document",
    value=st.session_state.get("voice_question", ""),
    height=120,
    placeholder="Example: What is this document about?",
    key="question_box"
)

    # ========================================================
    # SPEECH TO TEXT
    # ========================================================

    if audio is not None:

        st.audio(
            audio
        )

        with st.spinner(
            "🎧 Converting speech to text..."
        ):

            temp_path = None

            try:

                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".wav"
                ) as temp_file:

                    temp_file.write(
                        audio.getvalue()
                    )

                    temp_path = temp_file.name


                # Load Whisper
                transcriber = Transcriber()


                # Convert speech → text
                transcript, confidence = (
                    transcriber.transcribe_file(
                        temp_path
                    )
                )


                if transcript:

                    st.session_state.voice_question = (
                        transcript
                    )

                    st.success(
                        "✅ Speech converted to text!"
                    )

                    st.text_area(
                        "📝 Transcribed Question",
                        value=transcript,
                        height=100,
                        key="voice_transcript"
                    )

                    st.caption(
                        "🎯 Confidence: "
                        f"{confidence * 100:.1f}%"
                    )

                else:

                    st.warning(
                        "⚠️ I couldn't understand "
                        "the audio. Please speak again."
                    )


            except Exception as e:

                st.error(
                    "❌ Voice processing error:\n\n"
                    f"{e}"
                )


            finally:

                # Delete temporary file
                if temp_path:

                    try:

                        os.remove(
                            temp_path
                        )

                    except Exception:

                        pass


    # ========================================================
    # QUESTION BOX
    # ========================================================

    st.markdown(
        "### 📝 Your Question"
    )

    question = st.text_area(
        "Ask something about your document",
        value=st.session_state.voice_question,
        placeholder=(
            "Example: What is this document about?"
        ),
        height=120
    )


    # ========================================================
    # AI ANSWER BUTTON
    # ========================================================

    if st.button(
        "🤖 Get AI Answer",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "⚠️ Please speak or type a question."
            )

        elif st.session_state.rag is None:

            st.warning(
                "⚠️ Please upload and index "
                "a document first."
            )

        else:

            with st.spinner(
                "🔍 Searching documents and "
                "generating AI answer..."
            ):

                try:

                    result = (
                        st.session_state.rag.ask(
                            question.strip()
                        )
                    )

                    st.session_state.answer = result

                except Exception as e:

                    st.error(
                        f"❌ RAG error:\n\n{e}"
                    )


    # ========================================================
    # AI ANSWER DISPLAY
    # ========================================================

    if st.session_state.answer:

        result = st.session_state.answer

        st.divider()

        st.subheader(
            "🤖 AI Answer"
        )


        if result.answer:

            st.success(
                result.answer
            )

        else:

            st.warning(
                "I don't have enough information "
                "in the provided documents to answer "
                "this reliably."
            )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.metric(
            "🎯 Confidence",
            f"{result.confidence * 100:.1f}%"
        )


        # ----------------------------------------------------
        # SOURCES
        # ----------------------------------------------------

        if result.sources:

            st.subheader(
                "📚 Sources"
            )

            for i, source in enumerate(
                result.sources,
                start=1
            ):

                st.write(
                    f"**{i}.** "
                    f"{source.get('source', 'Unknown')}"
                )

                st.caption(
                    f"Similarity Score: "
                    f"{source.get('score', 0)}"
                )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    st.subheader(
        "📊 System Status"
    )


    # Streamlit
    st.success(
        "🟢 Streamlit"
    )


    # Voice
    if audio is not None:

        st.success(
            "🟢 Voice Input"
        )

    else:

        st.warning(
            "🟡 Voice Input"
        )


    # Document RAG
    if st.session_state.rag:

        st.success(
            "🟢 Document RAG"
        )

    else:

        st.warning(
            "🟡 Document RAG"
        )


    # AI
    if st.session_state.answer:

        st.success(
            "🟢 AI Answer"
        )

    else:

        st.warning(
            "🟡 AI Answer"
        )


    # TTS
    st.warning(
        "🟡 Text-to-Speech"
    )


    # ========================================================
    # INDEXED DOCUMENT
    # ========================================================

    st.divider()

    st.subheader(
        "📄 Current Document"
    )

    if st.session_state.indexed_file:

        st.success(
            st.session_state.indexed_file
        )

    else:

        st.info(
            "No document indexed yet."
        )


    # ========================================================
    # PIPELINE
    # ========================================================

    st.divider()

    st.subheader(
        "🚀 VoiceRAG Pipeline"
    )

    st.write("🎤 Voice Input")

    st.write("↓")

    st.write("📝 Speech-to-Text")

    st.write("↓")

    st.write("📄 Document")

    st.write("↓")

    st.write("✂️ Chunking")

    st.write("↓")

    st.write("🧠 Embeddings")

    st.write("↓")

    st.write("🔎 Vector Search")

    st.write("↓")

    st.write("🤖 LLM")

    st.write("↓")

    st.write("💬 AI Answer")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚀 VoiceRAG AI | Hackathon Project | "
    "Voice + RAG + Generative AI"
)
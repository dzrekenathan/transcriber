# streamlit_app.py
import os
import uuid
from openai import OpenAI
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ─── UI ────────────────────────────────────────────────────────────────────────
st.title("📢 Audio Transcription & Q&A")

uploaded_files = st.file_uploader(
    "Upload audio files",
    type=["mp3", "wav", "m4a", "flac"],
    accept_multiple_files=True
)

if uploaded_files:
    transcriptions = []
    with st.spinner("Transcribing…"):
        for audio in uploaded_files:
            # each audio is a Streamlit UploadedFile -> file-like
            resp = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=audio,
                response_format="text"
            )
            transcriptions.append({
                "id": str(uuid.uuid4()),
                "file_name": audio.name,
                "transcription": resp
            })

    st.subheader("Transcriptions")
    for t in transcriptions:
        st.markdown(f"**{t['file_name']}**")
        st.text_area("Transcript", t["transcription"], height=150)

    question = st.text_input("❓ Ask a question about these transcripts")
    if question:
        with st.spinner("Thinking…"):
            # Combine all transcripts into a single context
            context = "\n\n".join(t["transcription"] for t in transcriptions)
            messages = [
                {"role": "system",
                 "content": "You are a helpful assistant. Use the provided transcripts to answer the user’s question."},
                {"role": "user",
                 "content": f"Transcripts:\n{context}\n\nQuestion: {question}"}
            ]
            chat = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            answer = chat.choices[0].message.content

        st.subheader("Answer")
        st.write(answer)

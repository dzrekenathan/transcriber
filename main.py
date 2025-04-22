# main.py
import os
import uuid
from typing import List, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from openai import OpenAI

# ─── Configuration ────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set the OPENAI_API_KEY environment variable")

client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(title="Audio Transcription & Q&A API")

# ─── Schemas ──────────────────────────────────────────────────────────────────
# (you could also use Pydantic models here for stricter typing)

# ─── Endpoint ─────────────────────────────────────────────────────────────────
@app.post("/transcribe")
async def transcribe_and_ask(
    files: List[UploadFile] = File(...),
    question: Optional[str] = Form(None)
):
    # 1) Transcribe each file
    transcriptions = []
    for upload in files:
        try:
            # pass the raw file-stream to OpenAI
            result = client.audio.transcriptions.create(
                model="gpt-4o-transcribe",
                file=upload.file,
                response_format="text"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

        transcriptions.append({
            "id": str(uuid.uuid4()),
            "file_name": upload.filename,
            "transcription": result
        })

    payload = {"transcriptions": transcriptions}

    # 2) If a question is provided, ask the LLM using the transcripts as context
    if question:
        context = "\n\n".join(t["transcription"] for t in transcriptions)
        messages = [
            {"role": "system",
             "content": "You are a helpful assistant. Use the following transcripts to answer the question."},
            {"role": "user",
             "content": f"Transcripts:\n{context}\n\nQuestion: {question}"}
        ]
        try:
            chat = client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            answer = chat.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"LLM completion failed: {e}")

        payload["answer"] = answer

    return JSONResponse(payload)


# from openai import OpenAI
# import os
# import uuid
# import json


# api_key = OPENAI_API_KEY = "OPENAI_API_KEY"

# client = OpenAI(api_key=api_key)


# # Directory containing your audio files
# audio_folder = "C:/Users/Nathan/Downloads/Music"

# # List to hold transcription results
# transcriptions = []

# def transcribe_audio(file_path):
#     # Loop through all mp3 files in the folder
#     for filename in os.listdir(audio_folder):
#         if filename.endswith(".mp3"):
#             audio_path = os.path.join(audio_folder, filename)

#             with open(audio_path, "rb") as audio_file:
#                 transcription = client.audio.transcriptions.create(
#                     model="gpt-4o-transcribe",
#                     file=audio_file,
#                     response_format="text"
#                 )

#             transcriptions.append({
#                 "id": str(uuid.uuid4()),  
#                 "file_name": filename,
#                 "transcription": transcription
#             })
#     return json.dumps(transcriptions, indent=2)


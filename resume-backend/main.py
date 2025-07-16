from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pdfminer.high_level import extract_text
from starlette.responses import StreamingResponse
import requests
import tempfile
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    resume_text = extract_text(tmp_path)

    def stream_response():
        with requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "resume-analyst",
                "stream": True,
                "prompt": f"""
                    This is a resume:

                    {resume_text.strip()}

                    Please analyze the resume and respond in the following structured format:

                    1. Overall Structure and Clarity
                    2. Strengths
                    3. Weaknesses
                    4. Recommendations for improvement (bullet points)
                    5. Suitable job titles based on skills and experience (at least 3 roles, e.g. 'Full Stack Developer', 'AI Engineer', etc.)
                    6. Final Thoughts

                    Keep the tone professional and concise. Use Markdown formatting.
                """,
            },
            stream=True,
        ) as response:
            for line in response.iter_lines():
                if line:
                    try:
                        # Ollama stream responses start with `data: { ... }`
                        line = line.decode("utf-8").removeprefix("data: ")
                        parsed = json.loads(line)
                        yield parsed.get("response", "")
                    except Exception as e:
                        yield f"\n[Stream error: {e}]"

    return StreamingResponse(stream_response(), media_type="text/plain")

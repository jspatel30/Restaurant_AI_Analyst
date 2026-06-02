from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PromptRequest(BaseModel):
    system: str
    userContent: str
    maxTokens: int = 2000

@app.post("/analyze")
async def analyze(req: PromptRequest):
    prompt = f"""
SYSTEM:
{req.system}

USER:
{req.userContent}
"""

    response = model.generate_content(prompt)

    return {
        "text": response.text
    }
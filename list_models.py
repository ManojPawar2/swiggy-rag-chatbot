"""
Quick script to list all Gemini models available for your API key.
Run with: venv\Scripts\python.exe list_models.py
"""
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY", "")
if not api_key:
    print("ERROR: GEMINI_API_KEY not set in .env")
    exit(1)

client = genai.Client(api_key=api_key)

print("\nModels available for generateContent:\n")
for m in client.models.list():
    if "generateContent" in (m.supported_actions or []):
        print(f"  {m.name}")

import os
import uvicorn
import httpx
import asyncio 
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# ----------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# ----------------------------------------------------------------------

# --- Load environment variables from .env file for local development ---
load_dotenv()

app = FastAPI()

# 1. TEMPORARY DEBUG KEY (!!! MUST BE REMOVED BEFORE COMMIT/DEPLOYMENT !!!)
# PASTE YOUR KEY HERE ONLY FOR LOCAL DEBUGGING. Use an empty string "" otherwise.

# CRITICAL SECURITY LINE: Prioritizes environment variable/load_dotenv, falls back to hardcode for debug.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- WARNING/DEBUGGING BLOCK ---
# if GEMINI_API_KEY and GEMINI_API_KEY != HARDCODED_API_KEY:
#     print("DEBUG: GEMINI_API_KEY successfully loaded from environment or .env file.")
# elif GEMINI_API_KEY and GEMINI_API_KEY == HARDCODED_API_KEY:
#     # New Warning for hardcoded key
#     print("!!! WARNING !!!: USING HARDCODED API KEY FOR DEBUGGING. REMOVE BEFORE DEPLOYMENT.")
# else:
#     print("DEBUG: WARNING! GEMINI_API_KEY is still NOT loaded. Check .env file or export command.")
# # ----------------------------------

GEMINI_MODEL = "gemini-2.5-flash-preview-09-2025"
GEMINI_API_URL_TEMPLATE = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key="


# Pydantic model for the incoming request body from the HTML file
class ChatRequest(BaseModel):
    user_prompt: str
    resume_context: str

# ----------------------------------------------------------------------
# 2. CHAT PROXY ENDPOINT (The Secure Part)
# ----------------------------------------------------------------------

@app.post("/api/chat")
async def chat_proxy(request: ChatRequest):
    """
    Handles the request from the frontend, securely calls the Gemini API, 
    and returns the response. The API key is used here, server-side.
    """
    # 1. Security Check: Ensure the API Key is available from the environment
    if not GEMINI_API_KEY:
        # If the key is missing, deployment is misconfigured.
        raise HTTPException(
            status_code=500, 
            detail="Server Error: The GEMINI_API_KEY environment variable is not set securely."
        )

    # Construct the full API URL with the secure key
    gemini_api_url = GEMINI_API_URL_TEMPLATE + GEMINI_API_KEY

    # 2. RAG Setup: Define the AI's persona and context
    system_prompt = f"""You are a helpful, professional, and concise AI assistant for Manoj M Kadaramandalgi. 
        """

    # Combine user prompt with resume data for Retrieval-Augmented Generation (RAG) style response
    user_query = f"Context (Manoj's Resume):\n---\n{request.resume_context}\n---\n\nUser Question: {request.user_prompt}"

    payload = {
        "contents": [{"parts": [{"text": user_query}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }

    # 3. Secure Server-to-Server API Call with Exponential Backoff
    for i in range(3): # Retry up to 3 times
        try:
            # Use httpx for asynchronous HTTP requests
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    gemini_api_url,
                    headers={"Content-Type": "application/json"},
                    json=payload
                )
                
                # Check for API failure codes (4xx or 5xx)
                if response.status_code >= 400:
                    # Print the detailed JSON response from the Google API for inspection
                    print(f"--- FAILED API CALL DEBUG INFO (Attempt {i+1}) ---")
                    print(f"Status Code: {response.status_code}")
                    print(f"Google API Error Response Body:")
                    print(response.text)
                    print("-------------------------------------------------")
                
                response.raise_for_status() # Raise exception for bad status codes (4xx or 5xx)

                # Extract the generated text
                result = response.json()
                candidate = result.get("candidates", [{}])[0]
                response_text = candidate.get("content", {}).get("parts", [{}])[0].get("text", "Error: No text generated.")

                return {"response_text": response_text}

        except httpx.HTTPStatusError as e:
            # Handle specific API errors
            # The detailed response is already printed above, but we keep this for structure
            if i == 2: # Last attempt failed
                error_detail = f"Gemini API Error (HTTP Status {e.response.status_code}): Check your key and model name."
                raise HTTPException(status_code=e.response.status_code, detail=error_detail)
            await asyncio.sleep(2 ** i) # Exponential backoff delay

        except Exception as e:
            # Handle network or parsing errors
            print(f"General Error during API call (Attempt {i+1}): {e}")
            if i == 2:
                 raise HTTPException(status_code=500, detail="Internal Server Error during Gemini call (network/JSON parse issue).")
            await asyncio.sleep(2 ** i)
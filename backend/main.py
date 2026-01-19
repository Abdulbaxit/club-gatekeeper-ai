from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# Application configuration
APP_TITLE = "The Doorman Game API"
REQUEST_TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", "30"))
WIN_THRESHOLD = int(os.getenv("WIN_THRESHOLD", "100"))

# OpenRouter attribution headers
APP_PUBLIC_URL = (os.getenv("APP_PUBLIC_URL") or "").strip() 
APP_NAME = (os.getenv("APP_NAME") or "Doorman Game").strip()

# LLM Provider Configuration (OpenRouter)
LLM_PROVIDER = "openrouter"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")

# Model selection
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "meta-llama/llama-3.1-8b-instruct")
DOORMAN_MODEL = os.getenv("DOORMAN_MODEL", "meta-llama/llama-3.3-70b-instruct:free")


# Agent A: The Doorman - Conversational persona with rich backstory
DOORMAN_SYSTEM_PROMPT = """You are Arthur: a doorman at an exclusive Dubai nightclub, and a former philosophy professor.

Tone: calm, polite, professional, and warm.

Behavior:
- Stay in character as a real doorman.
- Prefer curiosity over confrontation; ask thoughtful questions.
- If you refuse entry, do so gently and offer a clear way to try again.
- Keep replies concise: 1–3 short sentences and at most one question.

You evaluate people by character, not wealth or status.
"""

# Agent B: The Judge - Hidden evaluator that scores persuasiveness
JUDGE_SYSTEM_PROMPT = """You are the Judge for a persuasion game. Score how convincing the user's last message is at getting into the club.

You are NOT the doorman. You are a hidden evaluator.

Return ONLY valid JSON with exactly these keys:
{"reasoning":"one short sentence","score":<integer>}

Rules:
- score is an integer from -20 to +20.
- Be slightly generous to keep the game moving: normal polite messages should usually score +5 to +12.
- Reserve negative scores for arrogance, entitlement, insults, threats, or obvious manipulation.

No markdown. No extra text.
"""


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatResponse(BaseModel):
    doorman_reply: str
    influence_score: int
    total_influence: int
    reasoning: Optional[str] = None
    game_won: bool = False


class SessionResponse(BaseModel):
    session_id: str
    total_influence: int
    conversation: List[Message]


def _extract_json_object(text: str) -> str:
    """Extract JSON from markdown code blocks if present."""
    t = (text or "").strip()
    if "```json" in t:
        return t.split("```json", 1)[1].split("```", 1)[0].strip()
    if "```" in t:
        return t.split("```", 1)[1].split("```", 1)[0].strip()
    return t


def _parse_judge_response(raw: str) -> Tuple[int, str]:
    """
    Parse Judge's JSON response and extract score + reasoning.
    Falls back to default score (+5) if parsing fails to keep game moving.
    """
    json_str = _extract_json_object(raw)
    try:
        data = json.loads(json_str)
        score = int(data.get("score", 0))
        reasoning = str(data.get("reasoning", "")).strip()
        score = max(-20, min(20, score))  # Clamp to valid range
        return score, reasoning
    except Exception:
        # Default positive score ensures game continues even if Judge response is malformed
        return 5, (raw or "")[:200]


async def _post_json(url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generic HTTP POST helper with error handling."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        r = await client.post(url, headers=headers, json=payload)
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, detail=r.text[:500])
    return r.json()


async def call_openrouter(model: str, messages: List[dict], system_prompt: Optional[str]) -> str:
    """
    Call OpenRouter API using OpenAI-compatible format.
    Includes attribution headers for API usage tracking (optional).
    """
    if not OPENROUTER_API_KEY:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")

    api_messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(messages)

    headers: Dict[str, str] = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    if APP_PUBLIC_URL:
        headers["HTTP-Referer"] = APP_PUBLIC_URL
    if APP_NAME:
        headers["X-Title"] = APP_NAME

    data = await _post_json(f"{OPENROUTER_API_URL}/chat/completions", headers, {"model": model, "messages": api_messages})
    return data["choices"][0]["message"]["content"]


async def call_llm(model: str, messages: List[dict], system_prompt: Optional[str]) -> str:
    """Unified LLM call interface (currently OpenRouter only)."""
    return await call_openrouter(model, messages, system_prompt)


# In-memory session storage
game_sessions: Dict[str, Dict[str, Any]] = {}

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_conversation_history(session_id: str) -> List[dict]:
    """Retrieve conversation history for a session in LLM message format."""
    session = game_sessions.get(session_id)
    if not session:
        return []
    return [{"role": msg["role"], "content": msg["content"]} for msg in session["conversation"]]


@app.get("/")
def root() -> Dict[str, Any]:
    return {"message": APP_TITLE, "status": "running", "provider": LLM_PROVIDER, "judge_model": JUDGE_MODEL, "doorman_model": DOORMAN_MODEL}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:

    session = game_sessions.setdefault(request.session_id, {"conversation": [], "total_influence": 0})
    session["conversation"].append({"role": "user", "content": request.message})

    history = get_conversation_history(request.session_id)

    # Step 1: Judge evaluates user message
    judge_raw = await call_llm(JUDGE_MODEL, history, JUDGE_SYSTEM_PROMPT)
    influence_score, reasoning = _parse_judge_response(judge_raw)

    # Step 2: Update cumulative influence (never goes below 0)
    session["total_influence"] = max(0, int(session["total_influence"]) + int(influence_score))
    game_won = session["total_influence"] >= WIN_THRESHOLD

    # Step 3: Doorman responds (with win instruction if threshold reached)
    doorman_prompt = DOORMAN_SYSTEM_PROMPT + ("\nThe user has convinced you. Let them in now, naturally and politely." if game_won else "")
    doorman_reply = await call_llm(DOORMAN_MODEL, history, doorman_prompt)
    session["conversation"].append({"role": "assistant", "content": doorman_reply})

    return ChatResponse(
        doorman_reply=doorman_reply,
        influence_score=influence_score,
        total_influence=session["total_influence"],
        reasoning=reasoning,
        game_won=game_won,
    )


@app.get("/api/session/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    session = game_sessions.get(session_id)
    if not session:
        return SessionResponse(session_id=session_id, total_influence=0, conversation=[])
    return SessionResponse(session_id=session_id, total_influence=session["total_influence"], conversation=session["conversation"])


@app.post("/api/session/{session_id}/reset")
def reset_session(session_id: str) -> Dict[str, str]:
    game_sessions[session_id] = {"conversation": [], "total_influence": 0}
    return {"message": "Session reset", "session_id": session_id}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)



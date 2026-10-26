import os
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI, RateLimitError, APIStatusError

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


class Turn(BaseModel):
    user: str
    assistant: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[Turn] = Field(default_factory=list)
    system_prompt: str = "You are a helpful assistant."


def build_messages(history, user_msg, system_prompt):
    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": "user", "content": turn["user"]})
        messages.append({"role": "assistant", "content": turn["assistant"]})
    messages.append({"role": "user", "content": user_msg})
    return messages


@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    if not api_key or not base_url:
        raise HTTPException(status_code=500, detail="Server configuration error.")

    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = build_messages(
        [{"user": t.user, "assistant": t.assistant} for t in req.history],
        req.message,
        req.system_prompt,
    )

    async def generate():
        try:
            stream = client.chat.completions.create(
                model="gemma-4-E4B-it",
                messages=messages,
                stream=True,
            )
            for chunk in stream:
                if await request.is_disconnected():
                    break
                token = chunk.choices[0].delta.content or ""
                if token:
                    # JSON-encode payload so newlines and special characters are safe in SSE.
                    import json
                    yield f"data: {json.dumps(token, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except RateLimitError:
            yield 'event: error\ndata: {"message":"Rate limited; please retry later."}\n\n'
            yield "data: [DONE]\n\n"
        except APIStatusError:
            yield 'event: error\ndata: {"message":"Upstream model service failed."}\n\n'
            yield "data: [DONE]\n\n"
        except Exception:
            yield 'event: error\ndata: {"message":"Unexpected server error."}\n\n'
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

# HW2 Streaming LLM App

## Files
- `main.py`: FastAPI `/api/chat` SSE endpoint and `/api/health`
- `static/index.html`: same-origin browser chat UI
- `requirements.txt`: Python dependencies

## Local setup
1. Install dependencies: `pip install -r requirements.txt`
2. Set environment variables `OPENAI_API_KEY` and `OPENAI_BASE_URL`.
3. Run: `uvicorn main:app --reload`
4. Open http://127.0.0.1:8000

Never commit your API key or `.env` file to GitHub. Set the two environment variables in Render's Environment settings.

The model is configured as `gemma-4-E4B-it` to match the homework prompt. If your instructor's proxy specifies a different model name, update the `model=` value in `main.py`.

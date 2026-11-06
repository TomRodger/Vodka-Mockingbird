# Project name

One-line pitch goes here.

## Running locally

Needs [uv](https://docs.astral.sh/uv/) and Node 20+.

**Backend** (http://localhost:8000, docs at /docs):

```
cd backend
cp ../.env.example .env
uv sync
uv run uvicorn app.main:app --reload
```

**Frontend** (http://localhost:5173), in a second terminal:

```
cd frontend
npm install
npm run dev
```

The page should say `Backend: ok`. If it doesn't, the backend isn't running.

## Workflow

- `git pull --rebase` before you push. Push small and often.
- Don't force push.
- Keep to your own area (`backend/` or `frontend/`) where possible to avoid conflicts.
- Format before committing: `uv run ruff format` in `backend/`.

## Team

- 

"""Live local demo: run a team's stand-ups one at a time from the JSON database and watch each step.

    standup-exp board demo            ->  http://127.0.0.1:8765  (local only; nothing is deployed)

Press **Run next stand-up**: the context pipeline runs on that team's next stand-up, reading the board from
the stand-in database (``results/board/db-demo-<model>.json``). The server sends each step as an event
straight away: the mention (Step 1), the cards pulled from the board (Step 2), the pick and rule (Step 3)
and the card change. The page plays them back at the viewer's pace (step by step, slow, normal or fast).
The stand-up is committed to the database before the page is told it is done. Model replies are cached,
so replaying a stand-up that has run before costs nothing.

One POST starts work (guarded, one run at a time); the SSE stream only listens, so reconnecting never re-runs.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path

from ..ontology import Ontology
from ..paths import BOARD_DOCS
from .context_pipeline import ContextPipeline
from .runner import SERIES_DIR, db_path, load_series
from .schema import Card, apply
from .store import BoardStore

LIVE_PAGE = BOARD_DOCS / "live-demo" / "index.html"


def _card_row(c: Card, store: BoardStore) -> dict:
    meta = store.meta.get(c.id, {})
    return {"card_id": c.id, "number": meta.get("number"), **c.model_dump(mode="json", exclude={"id", "history"})}


def create_app(model: str = "gpt-5.6-luna", data: Path = SERIES_DIR):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, StreamingResponse

    onto = Ontology.load()
    series = {s[0].series_id: s for s in load_series(data)}
    pipe = ContextPipeline(onto, model)
    store = BoardStore(db_path("demo", model))
    subscribers: set[asyncio.Queue] = set()
    state: dict = {"loop": None, "running": None}
    guard = threading.Lock()

    def publish(kind: str, **payload) -> None:
        event = json.loads(json.dumps({"type": kind, **payload}, default=str))
        for q in list(subscribers):
            state["loop"].call_soon_threadsafe(q.put_nowait, event)

    def remaining(sid: str):
        done = set(store.standups_run(sid))
        return [d for d in series[sid] if d.id not in done]

    def worker(sid: str, days) -> None:
        try:
            for day in days:
                publish("standup", series_id=sid, standup_id=day.id)
                board = store.board(sid)
                t0 = time.perf_counter()
                run = pipe.run_day(day, board)
                publish("mentions", standup_id=day.id, mentions=run.why["mentions"])
                live = [c.model_copy(deep=True) for c in board]
                for dec in run.why["decisions"]:
                    publish("mention", index=dec["mention"])
                    publish("context", index=dec["mention"], cards=dec["context_card_ids"])
                    events = [run.events[k] for k in dec["events"]]
                    live = apply(live, events, day)
                    changed = {e.card_id for e in events}
                    publish("decision", index=dec["mention"], decision={k: v for k, v in dec.items() if k != "events"},
                            cards=[_card_row(c, store) for c in live if c.id in changed])
                cost = sum((s.cost_usd or 0) for s in run.steps.values())
                store.commit(day, run.board_after, run.why, {"status": "done", "error": None, "model": model,
                                                             "cost_usd": round(cost, 6),
                                                             "seconds": round(time.perf_counter() - t0, 2)})
                publish("committed", series_id=sid, standup_id=day.id, db=store.to_dict())
        except Exception as exc:  # noqa: BLE001 - show it on the page rather than crash the server
            publish("error", message=str(exc)[:300])
        finally:
            state["running"] = None
            publish("idle")

    @asynccontextmanager
    async def lifespan(app):
        state["loop"] = asyncio.get_running_loop()
        yield

    app = FastAPI(title="Stand-up board: live demo", lifespan=lifespan)

    @app.get("/")
    def page():
        return FileResponse(LIVE_PAGE)

    @app.get("/api/state")
    def get_state():
        return {
            "model": model, "running": state["running"],
            "people": {p.id: p.name for p in onto.people},
            "sites": {s.id: s.name for s in onto.sites},
            "assets": {a.id: {"name": a.name, "parent": a.parent} for a in onto.assets},
            "series": [{"id": sid, "team": days[0].team, "days": [
                {"id": d.id, "day": d.day, "date": d.date.isoformat(), "chair": d.chair_id, "attendees": d.attendees,
                 "transcript": [{"t": round(u.start), "who": u.speaker_id, "text": u.text} for u in d.utterances]}
                for d in days]} for sid, days in series.items()],
            "db": store.to_dict(),
        }

    @app.post("/api/run/{sid}")
    def run_next(sid: str, rest: bool = False):
        """Run the team's next stand-up, or all remaining ones with ``?rest=true``."""
        if sid not in series:
            raise HTTPException(404, f"no team {sid}")
        with guard:
            if state["running"]:
                raise HTTPException(409, f"already running {state['running']}")
            todo = remaining(sid)
            if not todo:
                raise HTTPException(400, "every stand-up for this team has run; reset it to start again")
            days = todo if rest else todo[:1]
            state["running"] = days[0].id
        threading.Thread(target=worker, args=(sid, days), daemon=True).start()
        return {"started": [d.id for d in days]}

    @app.post("/api/reset/{sid}")
    def reset(sid: str):
        with guard:
            if state["running"]:
                raise HTTPException(409, "wait for the running stand-up to finish")
            store.reset(sid)
        publish("reset", series_id=sid, db=store.to_dict())
        return {"reset": sid}

    @app.get("/api/events")
    async def events():
        q: asyncio.Queue = asyncio.Queue()
        subscribers.add(q)

        async def stream():
            try:
                yield "retry: 2000\n\n"
                while True:
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=15)
                        yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
            finally:
                subscribers.discard(q)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app


def serve(model: str = "gpt-5.6-luna", data: Path = SERIES_DIR, port: int = 8765, open_browser: bool = False) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise SystemExit('The live demo needs the demo extra: uv pip install -e ".[demo]"') from exc
    url = f"http://127.0.0.1:{port}"
    print(f"Live demo on {url}  (Ctrl+C to stop). Database: {db_path('demo', model)}")
    if open_browser:
        threading.Timer(1.0, webbrowser.open, args=(url,)).start()
    uvicorn.run(create_app(model, data), host="127.0.0.1", port=port, log_level="warning")

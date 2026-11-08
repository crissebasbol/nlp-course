"""FastAPI app: REST API + small web interface to try the pipeline.

Run: uvicorn medec.serving.app:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from medec.config import get_output_dir, load_config

STATIC_DIR = Path(__file__).parent / "static"
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    from medec.serving.pipeline import JobManager, MedecService

    cfg = load_config(os.environ.get("MEDEC_CONFIG"))
    service = MedecService(cfg)
    state.update(cfg=cfg, service=service, jobs=JobManager(service))
    yield


app = FastAPI(title="MEDEC pipeline", lifespan=lifespan)


class AnalyzeRequest(BaseModel):
    text: str | None = Field(default=None, description="Clinical text (prose or numbered sentences)")
    text_id: str | None = Field(default=None, description="ID of a test example")
    use_bert: bool = True
    use_dspy: bool = True
    correct: bool = True
    use_claude: bool = Field(default=False, description="Also ask Claude Opus 5.5 (costs money; needs ANTHROPIC_API_KEY)")
    weight: float | None = Field(default=None, ge=0, le=1)
    threshold: float | None = Field(default=None, ge=0, le=1)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", **state["service"].status()}


@app.get("/api/examples")
def examples() -> list[dict]:
    return state["service"].example_list()


@app.get("/api/examples/{text_id}")
def example(text_id: str) -> dict:
    ex = state["service"].example(text_id)
    if ex is None:
        raise HTTPException(404, "Example not found")
    return ex


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict:
    if not req.text and not req.text_id:
        raise HTTPException(422, "Send `text` or `text_id`")
    payload = {k: v for k, v in req.model_dump().items() if v is not None}
    return {"job_id": state["jobs"].submit(payload)}


@app.get("/api/jobs/{job_id}")
def job(job_id: str) -> dict:
    data = state["jobs"].get(job_id)
    if data is None:
        raise HTTPException(404, "Job not found")
    return data


@app.get("/api/metrics")
def metrics() -> dict:
    reports = get_output_dir(state["cfg"]) / "reports"
    out = {}
    for split in ("test", "val"):
        path = reports / f"metrics_{split}.json"
        if path.exists():
            out[split] = json.loads(path.read_text())
    combine = get_output_dir(state["cfg"]) / "artifacts" / "combine.json"
    if combine.exists():
        out["combine"] = json.loads(combine.read_text())
    return out


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

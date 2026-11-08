"""API smoke test with a fake service (no models needed)."""

import time

from fastapi.testclient import TestClient

from medec.serving import app as app_module
from medec.serving.pipeline import JobManager, word_diff


class FakeService:
    def status(self):
        return {"bert": True, "llm": False, "dspy_compiled": False, "corrector": False, "examples": 0,
                "defaults": {"weight": 0.7, "threshold": 0.5, "bert_threshold": 0.5}}

    def example_list(self):
        return []

    def example(self, text_id):
        return None

    def analyze(self, payload, report):
        report("prepare", "done", 0.0)
        return {"error_flag": 0, "sentences": [payload["text"]]}


def make_client():
    client = TestClient(app_module.app)
    service = FakeService()
    app_module.state.update(cfg={"paths": {"output_dir": "outputs"}}, service=service, jobs=JobManager(service))
    return client


def test_analyze_job_flow(monkeypatch):
    monkeypatch.setattr(app_module, "lifespan", None, raising=False)
    client = make_client()
    assert client.get("/api/health").json()["status"] == "ok"
    job_id = client.post("/api/analyze", json={"text": "He has fever."}).json()["job_id"]
    for _ in range(50):
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] == "done":
            break
        time.sleep(0.05)
    assert job["status"] == "done" and job["result"]["error_flag"] == 0


def test_analyze_requires_text():
    assert make_client().post("/api/analyze", json={}).status_code == 422


def test_word_diff():
    ops = word_diff("was H. influenzae today", "was S. pneumoniae today")
    assert {"op": "insert", "text": "S. pneumoniae"} in ops

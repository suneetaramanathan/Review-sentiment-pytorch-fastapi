from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_positive_review():
    r = client.post("/predict", json={"text": "Amazing hotel, friendly staff and a spotless room!"})
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment"] == "positive"
    assert 0 <= body["confidence"] <= 1


def test_negative_review():
    r = client.post("/predict", json={"text": "Dirty bathroom, rude reception, never coming back."})
    assert r.status_code == 200
    assert r.json()["sentiment"] == "negative"


def test_missing_text_is_rejected():
    r = client.post("/predict", json={})
    assert r.status_code == 422

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _mock_chat_pipeline(mocker, answer="Paris.", sources=None, condensed=None):
    sources = sources or ["Paris is the capital of France."]
    mocker.patch("app.api.routes.get_embedding", return_value=[0.1, 0.2, 0.3])
    mocker.patch("app.api.routes.hybrid_search", return_value=sources)
    mocker.patch("app.api.routes.rerank", return_value=sources)
    mocker.patch("app.api.routes.ask_llm", return_value=answer)
    mocker.patch(
        "app.api.routes.condense_question",
        return_value=condensed or "What is the capital of France?",
    )


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_endpoint_returns_answer_and_sources(mocker):
    _mock_chat_pipeline(mocker)

    response = client.post("/chat", json={"question": "What is the capital of France?", "k": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Paris."
    assert body["sources"] == ["Paris is the capital of France."]


def test_chat_generates_new_session_when_none_provided(mocker):
    _mock_chat_pipeline(mocker)

    response = client.post("/chat", json={"question": "What is the capital of France?"})

    body = response.json()
    assert body["session_id"]  # a session id was generated


def test_chat_reuses_provided_session_id(mocker):
    _mock_chat_pipeline(mocker)

    response = client.post(
        "/chat", json={"question": "What is the capital of France?", "session_id": "my-session"}
    )

    assert response.json()["session_id"] == "my-session"


def test_chat_persists_messages_to_history(mocker):
    _mock_chat_pipeline(mocker, answer="Paris.")

    import uuid

    session_id = f"history-test-session-{uuid.uuid4()}"
    client.post(
        "/chat",
        json={"question": "What is the capital of France?", "session_id": session_id},
    )

    from app.services.chat_history import get_history

    history = get_history(session_id, max_turns=5)
    assert history == [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]


def test_status_endpoint_for_unknown_job_returns_error():
    response = client.get("/status/does-not-exist")
    assert response.status_code == 200
    assert response.json() == {"error": "job not found"}


def test_upload_rejects_non_pdf_file():
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"just some text", "text/plain")},
    )
    assert response.status_code == 400

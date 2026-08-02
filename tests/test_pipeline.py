from app.rag.pipeline import process_document


def test_process_document_stops_cleanly_if_job_vanishes_before_processing(mocker, caplog):
    # update_job returning False simulates the job record already being
    # gone (e.g. storage wiped mid-flight) before any work started.
    mocker.patch("app.rag.pipeline.update_job", return_value=False)
    extract_text_mock = mocker.patch("app.rag.pipeline.extract_text")

    with caplog.at_level("WARNING"):
        process_document("/tmp/fake.pdf", "ghost-job-id")

    # Should bail out immediately rather than doing (paid) embedding work
    # against a job that can never be marked done.
    extract_text_mock.assert_not_called()
    assert "vanished mid-processing" in caplog.text


def test_process_document_logs_clearly_if_job_vanishes_after_processing(mocker, caplog):
    mocker.patch("app.rag.pipeline.extract_text", return_value="some extracted text")
    mocker.patch("app.rag.pipeline.chunk_text", return_value=["chunk one", "chunk two"])
    mocker.patch("app.rag.pipeline.get_embeddings", return_value=[[0.1], [0.2]])
    mocker.patch("app.rag.pipeline.store_chunks")
    mocker.patch("app.rag.pipeline.index_chunks")

    # First call (status="processing") succeeds; second call (status="done")
    # fails, simulating the job record disappearing partway through.
    mocker.patch("app.rag.pipeline.update_job", side_effect=[True, False])

    with caplog.at_level("WARNING"):
        process_document("/tmp/fake.pdf", "ghost-job-id")

    assert "vanished mid-processing" in caplog.text


def test_process_document_continues_normally_when_job_exists(mocker):
    mocker.patch("app.rag.pipeline.extract_text", return_value="some extracted text")
    mocker.patch("app.rag.pipeline.chunk_text", return_value=["chunk one"])
    mocker.patch("app.rag.pipeline.get_embeddings", return_value=[[0.1]])
    store_chunks_mock = mocker.patch("app.rag.pipeline.store_chunks")
    mocker.patch("app.rag.pipeline.index_chunks")
    update_job_mock = mocker.patch("app.rag.pipeline.update_job", return_value=True)

    process_document("/tmp/fake.pdf", "real-job-id")

    store_chunks_mock.assert_called_once()
    update_job_mock.assert_any_call("real-job-id", status="done", chunk_count=1)

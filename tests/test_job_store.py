import uuid

from app.services.job_store import create_job, get_job, update_job


def test_update_job_returns_true_when_job_exists():
    job_id = str(uuid.uuid4())
    create_job(job_id, "/tmp/fake.pdf")

    result = update_job(job_id, status="processing")

    assert result is True
    assert get_job(job_id)["status"] == "processing"


def test_update_job_returns_false_when_job_does_not_exist():
    result = update_job(str(uuid.uuid4()), status="processing")
    assert result is False


def test_update_job_returns_false_for_empty_kwargs():
    job_id = str(uuid.uuid4())
    create_job(job_id, "/tmp/fake.pdf")

    assert update_job(job_id) is False


def test_update_job_returns_false_when_no_allowed_fields_given():
    job_id = str(uuid.uuid4())
    create_job(job_id, "/tmp/fake.pdf")

    # "not_a_real_field" isn't in the allowed set, so nothing to update
    assert update_job(job_id, not_a_real_field="x") is False

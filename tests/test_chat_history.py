import time
import uuid

from app.services.chat_history import (
    _connect,
    append_message,
    cleanup_expired_messages,
    get_history,
)


def test_history_round_trip():
    session_id = str(uuid.uuid4())
    append_message(session_id, "user", "hello")
    append_message(session_id, "assistant", "hi there")

    history = get_history(session_id, max_turns=5)

    assert history == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_history_is_isolated_per_session():
    session_a = str(uuid.uuid4())
    session_b = str(uuid.uuid4())

    append_message(session_a, "user", "session a message")
    append_message(session_b, "user", "session b message")

    assert get_history(session_a, max_turns=5) == [
        {"role": "user", "content": "session a message"}
    ]
    assert get_history(session_b, max_turns=5) == [
        {"role": "user", "content": "session b message"}
    ]


def test_history_respects_max_turns():
    session_id = str(uuid.uuid4())
    for i in range(5):
        append_message(session_id, "user", f"q{i}")
        append_message(session_id, "assistant", f"a{i}")

    history = get_history(session_id, max_turns=2)

    assert len(history) == 4
    assert history[0]["content"] == "q3"
    assert history[-1]["content"] == "a4"


def test_unknown_session_returns_empty_history():
    assert get_history(str(uuid.uuid4()), max_turns=5) == []


def _insert_message_with_timestamp(session_id: str, created_at: float) -> None:
    """Bypasses append_message (which stamps `now()` and would also
    trigger the throttled cleanup) to insert a message with a specific,
    possibly-backdated, created_at — needed to test expiry directly."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_messages (session_id, role, content, created_at) "
            "VALUES (?, ?, ?, ?)",
            (session_id, "user", "backdated message", created_at),
        )


def test_cleanup_expired_messages_removes_old_rows_only(mocker):
    mocker.patch("app.services.chat_history.settings.CHAT_HISTORY_TTL_DAYS", 30)

    old_session = str(uuid.uuid4())
    recent_session = str(uuid.uuid4())

    forty_days_ago = time.time() - (40 * 86400)
    _insert_message_with_timestamp(old_session, forty_days_ago)
    _insert_message_with_timestamp(recent_session, time.time())

    deleted = cleanup_expired_messages()

    assert deleted >= 1
    assert get_history(old_session, max_turns=5) == []
    assert get_history(recent_session, max_turns=5) != []


def test_cleanup_expired_messages_disabled_when_ttl_is_zero(mocker):
    mocker.patch("app.services.chat_history.settings.CHAT_HISTORY_TTL_DAYS", 0)

    session_id = str(uuid.uuid4())
    _insert_message_with_timestamp(session_id, time.time() - (999 * 86400))

    deleted = cleanup_expired_messages()

    assert deleted == 0
    assert get_history(session_id, max_turns=5) != []

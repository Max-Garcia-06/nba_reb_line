"""Scan summary email must not hang a scan when the SMTP connection stalls."""
import notify


class _FakeSMTP:
    calls = []

    def __init__(self, host, port, timeout=None):
        _FakeSMTP.calls.append({"host": host, "port": port, "timeout": timeout})

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        pass

    def login(self, *a):
        pass

    def send_message(self, msg):
        pass


def test_smtp_connection_has_a_timeout(monkeypatch):
    # A Mac that sleeps mid-send leaves a dead socket; without a timeout the
    # scan blocks forever and its lock skips every later scan that evening.
    _FakeSMTP.calls.clear()
    monkeypatch.setattr(notify.smtplib, "SMTP", _FakeSMTP)
    monkeypatch.setattr(notify, "SCAN_EMAIL_ENABLED", True)
    monkeypatch.setattr(notify, "GMAIL_APP_PASSWORD", "pw")
    monkeypatch.setattr(notify, "GMAIL_SENDER", "a@example.com")
    monkeypatch.setattr(notify, "SCAN_EMAIL_RECIPIENT", "b@example.com")
    notify.send_email("s", "b")
    assert len(_FakeSMTP.calls) == 1
    t = _FakeSMTP.calls[0]["timeout"]
    assert t is not None and 0 < t <= 60


def test_smtp_timeout_is_swallowed(monkeypatch):
    def boom(*a, **k):
        raise TimeoutError("timed out")

    monkeypatch.setattr(notify.smtplib, "SMTP", boom)
    monkeypatch.setattr(notify, "SCAN_EMAIL_ENABLED", True)
    monkeypatch.setattr(notify, "GMAIL_APP_PASSWORD", "pw")
    monkeypatch.setattr(notify, "GMAIL_SENDER", "a@example.com")
    monkeypatch.setattr(notify, "SCAN_EMAIL_RECIPIENT", "b@example.com")
    notify.send_email("s", "b")  # must not raise

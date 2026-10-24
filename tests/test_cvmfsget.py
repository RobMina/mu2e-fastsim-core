"""cvmfsget retries: timeouts and 5xx are retried with backoff, 4xx are final, data objects fall back to other servers."""
import io
import urllib.error
import zlib

import pytest

from mu2efastsim import cvmfsget as C


class _Resp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _opener(script):
    calls = []

    def op(url):
        calls.append(url)
        r = script.pop(0)
        if isinstance(r, Exception):
            raise r
        return _Resp(r)
    return op, calls


def test_retry_then_success():
    op, calls = _opener([TimeoutError("timed out"), urllib.error.URLError("reset"), b"ok"])
    slept = []
    assert C._get("u", opener=op, sleep=slept.append) == b"ok"
    assert len(calls) == 3 and slept == [C.BACKOFF_S, 2 * C.BACKOFF_S]


def test_gives_up_after_attempts():
    op, calls = _opener([TimeoutError("t")] * C.ATTEMPTS)
    with pytest.raises(TimeoutError):
        C._get("u", opener=op, sleep=lambda s: None)
    assert len(calls) == C.ATTEMPTS


def test_404_is_final():
    op, calls = _opener([urllib.error.HTTPError("u", 404, "nf", {}, None), b"never"])
    with pytest.raises(urllib.error.HTTPError):
        C._get("u", opener=op, sleep=lambda s: None)
    assert len(calls) == 1


def test_object_falls_back_to_next_server(monkeypatch):
    seen = []

    def fake_get(url, **kw):
        seen.append(url)
        if url.startswith("http://a"):
            raise TimeoutError("a down")
        return zlib.compress(b"payload")
    monkeypatch.setattr(C, "_get", fake_get)
    r = C.Repo.__new__(C.Repo)
    r.repo, r.servers, r.server, r._cat = "x.org", ("http://a", "http://b"), "http://a", {}
    assert r._object("abcdef") == b"payload"
    assert seen[0].startswith("http://a") and seen[1].startswith("http://b")

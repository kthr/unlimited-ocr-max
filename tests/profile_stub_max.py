"""A stand-in for the ``max`` executable, used only by ``tests/test_profile_cli.py``.

Not a test module: pytest collects ``test_*.py``, and nothing is served on import. The tests
point ``unlimited-ocr-max profile`` at a wrapper that runs this file as ``max serve ... --port P``
(every other flag is ignored). It then behaves like the real server as far as ``profile`` can
see:

* ``GET /v1/models`` lists ``unlimited-ocr-max``;
* ``POST /v1/chat/completions`` checks the request body is exactly the research harness's
  shape (HTTP 400 otherwise), identifies the page by the sha256 of the decoded image against
  the bundled corpus, and streams that page's bf16 reference as SSE: an empty role delta, the
  text in 7-character deltas (at most ``max_tokens`` of them), a ``content: null`` delta with the
  finish reason, a usage chunk and ``data: [DONE]``;
* per request it prints scheduler lines like MAX's to stdout: one ``CE`` batch (30 s for the
  warmup, 5 s after), three ``TG`` batches of 50 ms, and -- once -- a whole-second ``TG`` line
  (2.50 s) that decode statistics must exclude.

It also spawns one child of its own (``sleep``), so a test can prove the whole tree goes away.
``PROFILE_STUB_DIR`` receives ``pids.json`` (``pid``, ``child``, ``argv``, ``ngram``) and one
``requests.jsonl`` line per request; ``PROFILE_STUB_PREFILL_S`` sets the delay before the first
text delta (default 0.05 s). ``PROFILE_STUB_MODE`` selects a misbehaviour:

``wrong-page``        one page's text has its first character changed
``exit-early``        exits 1 before serving, leaving its child behind in the process group
``exit-after-pages``  exits 0 right after answering the last page, leaving its child behind
``never-ready``       ``/v1/models`` never lists the model
``detach``            the child starts its own session, i.e. leaves the process group
``ignore-term``       ignores SIGTERM, and so does its child (the disposition is inherited)
``choices-dict``      the text deltas carry ``choices`` as an object instead of a list
``choices-empty``     the text deltas carry ``choices: []`` (and no usage)
``content-not-text``  the text deltas carry a number as ``delta.content``
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # the checkout these tests belong to

from unlimited_ocr_max import profile_corpus  # noqa: E402  (stdlib-only)

MODE_ENV = "PROFILE_STUB_MODE"
DIR_ENV = "PROFILE_STUB_DIR"
NGRAM_ENV = "_UNLIMITED_OCR_MAX_NGRAM_SIZE"
PROMPT = "<|grounding|>Convert the document to markdown."
BODY_KEYS = ["model", "temperature", "max_tokens", "stream", "stream_options", "messages"]
IMAGE_PREFIX = "data:image/png;base64,"
WRONG_PAGE = "dense_body"
CHUNK_CHARS = 7
TG_PER_REQUEST = 3
PREFILL_S = float(os.environ.get("PROFILE_STUB_PREFILL_S", "0.05"))

_LINE = (
    "{ts} INFO: Executed {stage} batch with 1 reqs | Terminated: 0 reqs, Pending: 0 reqs | "
    "Input Tokens: 1/8192 toks | Prompt Tput: 21.5 tok/s, Generation Tput: 21.5 tok/s | "
    "Batch creation: {creation}, Execution: {execution} | KVCache usage: 18.8% of 16 blocks | "
    "All Preemptions: 0 reqs"
)

MODE = os.environ.get(MODE_ENV, "")
OUT = Path(os.environ[DIR_ENV]) if os.environ.get(DIR_ENV) else None
PAGES_BY_SHA = {hashlib.sha256(profile_corpus.page_png(name)).hexdigest(): name for name in profile_corpus.page_names()}
_requests = 0
_lock = threading.Lock()


def _scheduler(stage: str, creation: str, execution: str) -> None:
    now = time.time()
    ts = time.strftime("%H:%M:%S", time.localtime(now)) + f".{int(now * 1000) % 1000:03d}"
    print(_LINE.format(ts=ts, stage=stage, creation=creation, execution=execution), flush=True)


def _page_of(body: object) -> str:
    """The corpus page a request body carries; ``ValueError`` for anything but the exact shape."""
    if not isinstance(body, dict) or list(body) != BODY_KEYS:
        raise ValueError(f"body keys {list(body) if isinstance(body, dict) else body!r}, expected {BODY_KEYS}")
    if (body["model"], body["temperature"], body["stream"], body["stream_options"]) != (
        "unlimited-ocr-max", 0, True, {"include_usage": True}
    ) or type(body["temperature"]) is not int or type(body["max_tokens"]) is not int:
        raise ValueError(f"unexpected sampling fields: {({k: body[k] for k in BODY_KEYS[:5]})}")
    messages = body["messages"]
    if len(messages) != 1 or list(messages[0]) != ["role", "content"] or messages[0]["role"] != "user":
        raise ValueError("expected exactly one user message")
    text_part, image_part = messages[0]["content"]  # text first, then the image
    if text_part != {"type": "text", "text": PROMPT} or list(text_part) != ["type", "text"]:
        raise ValueError(f"unexpected text part {text_part!r}")
    if list(image_part) != ["type", "image_url"] or image_part["type"] != "image_url" or list(image_part["image_url"]) != ["url"]:
        raise ValueError("unexpected image part")
    url = image_part["image_url"]["url"]
    if not url.startswith(IMAGE_PREFIX):
        raise ValueError("image is not a data:image/png;base64 URL")
    digest = hashlib.sha256(base64.b64decode(url[len(IMAGE_PREFIX):], validate=True)).hexdigest()
    if digest not in PAGES_BY_SHA:
        raise ValueError("image is not a page of the bundled corpus")
    return PAGES_BY_SHA[digest]


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 -- keep the log to scheduler lines
        pass

    def _json(self, status: int, payload: object) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/models":
            self._json(404, {"error": "not found"})
            return
        ids = [] if MODE == "never-ready" else [{"id": "unlimited-ocr-max", "object": "model"}]
        self._json(200, {"object": "list", "data": ids})

    def do_POST(self) -> None:  # noqa: N802
        global _requests
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            page = _page_of(body)
        except (ValueError, KeyError, TypeError) as e:
            self._json(400, {"error": str(e)})
            return
        with _lock:
            index = _requests
            _requests += 1
        if OUT is not None:
            with open(OUT / "requests.jsonl", "a") as log:
                log.write(json.dumps({"page": page, "max_tokens": body["max_tokens"]}) + "\n")

        text = profile_corpus.reference(page, "bf16")
        if MODE == "wrong-page" and page == WRONG_PAGE:
            text = ("Y" if text[0] == "X" else "X") + text[1:]
        chunks = [text[i:i + CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]
        streamed = chunks[: body["max_tokens"]]

        _scheduler("CE", "1.00ms", "30.00s" if index == 0 else "5.00s")
        for _ in range(TG_PER_REQUEST):
            _scheduler("TG", "0.10ms", "50.00ms")
        if index == 1:
            _scheduler("TG", "0.10ms", "2.50s")

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        # The empty role delta comes first and the text only after the "prefill", so a TTFT of at
        # least PREFILL_S proves the client timed the first *non-empty* delta.
        self._event({"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        time.sleep(PREFILL_S)
        for piece in streamed:
            self._event(_text_delta(piece))
        finish = "length" if len(streamed) < len(chunks) else "stop"
        self._event({"choices": [{"index": 0, "delta": {"content": None}, "finish_reason": finish}]})
        self._event({"choices": [], "usage": {"prompt_tokens": 276, "completion_tokens": len(streamed),
                                              "total_tokens": 276 + len(streamed)}})
        self.wfile.write(b"data: [DONE]\n\n")  # unbuffered: in the socket before the exit below
        if MODE == "exit-after-pages" and index == len(profile_corpus.page_names()):  # the warmup, then every page
            os._exit(0)

    def _event(self, payload: object) -> None:
        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())


def _text_delta(piece: str) -> dict[str, object]:
    choice = {"index": 0, "delta": {"content": piece}, "finish_reason": None}
    if MODE == "choices-dict":
        return {"choices": {"0": choice}}
    if MODE == "choices-empty":
        return {"choices": []}
    if MODE == "content-not-text":
        return {"choices": [{**choice, "delta": {"content": 7}}]}
    return {"choices": [choice]}


def main(argv: list[str]) -> int:
    if not argv or argv[0] != "serve":
        print(f"stub max: only `serve` is implemented, got {argv!r}", file=sys.stderr)
        return 2
    port = int(argv[argv.index("--port") + 1])
    if MODE == "ignore-term":
        signal.signal(signal.SIGTERM, signal.SIG_IGN)  # before the child, so it inherits the disposition
    child = subprocess.Popen(["sleep", "3600"], start_new_session=(MODE == "detach"))
    threading.Thread(target=child.wait, daemon=True).start()
    if OUT is not None:
        (OUT / "pids.json").write_text(json.dumps({
            "pid": os.getpid(), "child": child.pid, "argv": argv, "ngram": os.environ.get(NGRAM_ENV),
        }))
    print(f"stub max: starting on 127.0.0.1:{port} (mode {MODE or 'normal'!r})", flush=True)
    if MODE == "exit-early":
        print("stub max: failing before the server is up (exit-early)", flush=True)
        return 1
    http.server.ThreadingHTTPServer(("127.0.0.1", port), _Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

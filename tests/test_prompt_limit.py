"""KON-237: a prompt longer than ``MAX_PROMPT_TOKENS`` is refused in the API process, as an HTTP 400.

The model worker's KV page pool holds ``--max-batch-size`` slots of
``pages_for(MAX_PROMPT_TOKENS + window)`` pages, so the API process must not
hand it a longer prompt: a refusal inside the worker ends the worker process
and every request in flight with it. ``UnlimitedOcrTokenizer.new_context`` --
what MAX's ``TokenGeneratorPipeline.next_token_chunk`` awaits per request,
before the hand-off -- raises MAX's ``PromptTooLongError``, an ``InputError``,
which MAX's OpenAI routes answer with a 400 carrying its message
(``max/serve/router/openai_routes.py``: ``except InputError`` ->
``HTTPException(status_code=400, detail=str(e))``).

Model-free: the tokenizer is the real class over a stand-in delegate that
encodes one id per character, so a page plus ``n`` characters of text is a
``274 + n``-token prompt (BOS, 273 image placeholders, the text); the pipeline
is MAX's own ``TokenGeneratorPipeline`` over a model-worker stand-in that
records what reaches it.
"""

from __future__ import annotations

import asyncio
import base64
import io
from collections.abc import AsyncGenerator
from types import SimpleNamespace
from typing import Any

import pytest
from max.pipelines.context.exceptions import InputError, PromptTooLongError
from max.pipelines.modeling.types import TextGenerationRequest, TextGenerationRequestMessage
from max.pipelines.request import RequestID
from max.serve.pipelines.llm import TokenGeneratorPipeline
from PIL import Image

from unlimited_ocr_max import tokenizer as tk
from unlimited_ocr_max.kv_cache import MAX_PROMPT_TOKENS

#: BOS plus the 273 placeholder ids of one ``base``-mode page.
PAGE_PROMPT = 1 + 273


class _CharDelegate:
    """A ``PreTrainedTokenizerFast`` stand-in: one id per character, none of them special."""

    model_max_length = 32768
    eos_token_id = tk.EOS_ID

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [1000 + ord(char) % 1000 for char in text]

    def decode(self, ids: Any, **kwargs: Any) -> str:
        return ""

    def __len__(self) -> int:
        return 129280


def _tokenizer(monkeypatch: pytest.MonkeyPatch) -> tk.UnlimitedOcrTokenizer:
    """The real ``__init__``, over the stand-in delegate instead of ``tokenizer.json``."""
    monkeypatch.setattr(tk, "load_delegate", lambda *args, **kwargs: _CharDelegate())
    monkeypatch.setattr(tk, "skipped_special_token_ids", lambda delegate: {tk.EOS_ID, tk.IMAGE_TOKEN_ID})
    model = SimpleNamespace(
        huggingface_config=SimpleNamespace(candidate_resolutions=[[1024, 1024]]),
        kv_cache=SimpleNamespace(enable_prefix_caching=False),
    )
    return tk.UnlimitedOcrTokenizer("unused", SimpleNamespace(model=model))


def _page() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _request(text_chars: int) -> TextGenerationRequest:
    """A chat request as the OpenAI route builds it: one user message, the page and ``text_chars`` characters."""
    content = [{"type": "image"}, {"type": "text", "text": "x" * text_chars}]
    return TextGenerationRequest(
        request_id=RequestID(f"r{text_chars}"),
        model_name="unlimited-ocr",
        messages=[TextGenerationRequestMessage(role="user", content=content)],
        images=[_page()],
    )


def test_a_prompt_of_max_prompt_tokens_is_admitted_and_one_token_more_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    """At the limit: a context of exactly ``MAX_PROMPT_TOKENS`` prompt tokens. One over: ``PromptTooLongError``,
    an ``InputError``, whose message names the limit and the prompt's length."""
    assert MAX_PROMPT_TOKENS == 512
    tokenizer = _tokenizer(monkeypatch)

    context = asyncio.run(tokenizer.new_context(_request(MAX_PROMPT_TOKENS - PAGE_PROMPT)))
    assert context.tokens.prompt_length == MAX_PROMPT_TOKENS
    assert sum(int(token) == tk.IMAGE_TOKEN_ID for token in context.tokens.prompt) == 273

    with pytest.raises(PromptTooLongError) as refused:
        asyncio.run(tokenizer.new_context(_request(MAX_PROMPT_TOKENS - PAGE_PROMPT + 1)))
    assert isinstance(refused.value, InputError)
    assert (refused.value.num_tokens, refused.value.max_length) == (MAX_PROMPT_TOKENS + 1, MAX_PROMPT_TOKENS)
    assert str(refused.value) == (
        "Prompt is too long: 513 tokens exceeds the per-request prompt limit of this server (the page's image "
        "tokens count toward it) of 512 tokens. Please shorten your prompt."
    )


class _ModelWorker:
    """MAX's model-worker proxy as ``next_token_chunk`` uses it; records every hand-off."""

    def __init__(self) -> None:
        self.awaiting_admission: list[int] = []
        self.handed_off: list[Any] = []

    def note_awaiting_admission(self, delta: int) -> None:
        self.awaiting_admission.append(delta)

    async def stream(self, request_id: Any, context: Any) -> AsyncGenerator[Any, None]:
        self.handed_off.append(context)

        async def nothing() -> AsyncGenerator[Any, None]:
            return
            yield

        return nothing()


@pytest.mark.parametrize("stream", [False, True])
def test_the_chat_completions_route_answers_a_too_long_prompt_with_a_400(
    monkeypatch: pytest.MonkeyPatch, stream: bool
) -> None:
    """MAX's own ``/v1/chat/completions`` route, request middleware and HTTP-error handler, over the real tokenizer.

    What a client sees: a 400 with the OpenAI error envelope carrying the
    refusal's message, streaming or not, and the model worker never sees the
    request. The app is only the pieces of ``max serve``'s that this path
    reads (``api_server.fastapi_app`` would start a model worker); the
    pipeline config is a stand-in for the two parts the route reads
    (sampling and runtime defaults, and the main model's sampling defaults).
    """
    from fastapi import FastAPI, HTTPException
    from fastapi.testclient import TestClient
    from max.pipelines.context.sampling_params import SamplingParamsGenerationConfigDefaults
    from max.pipelines.lib import PipelineConfig
    from max.serve.api_server import _openai_http_exception_handler
    from max.serve.config import Settings
    from max.serve.request import register_request
    from max.serve.router import openai_routes

    worker = _ModelWorker()
    app = FastAPI()
    register_request(app)
    app.add_exception_handler(HTTPException, _openai_http_exception_handler)
    app.include_router(openai_routes.router)
    app.state.pipeline = TokenGeneratorPipeline(
        model_name="unlimited-ocr", tokenizer=_tokenizer(monkeypatch), model_worker=worker
    )
    main = SimpleNamespace(sampling_params_defaults=SamplingParamsGenerationConfigDefaults())
    app.state.pipeline_config = PipelineConfig.model_construct(models={"main": main})
    app.state.settings = Settings()

    page = "data:image/png;base64," + base64.b64encode(_page()).decode()
    text = "x" * (MAX_PROMPT_TOKENS - PAGE_PROMPT + 1)
    content = [{"type": "image_url", "image_url": {"url": page}}, {"type": "text", "text": text}]
    body = {"model": "unlimited-ocr", "messages": [{"role": "user", "content": content}], "stream": stream}
    response = TestClient(app).post("/v1/chat/completions", json=body)

    assert response.status_code == 400
    error = response.json()["error"]
    assert (error["code"], error["type"]) == ("400", "invalid_request_error")
    assert error["message"].startswith("Prompt is too long: 513 tokens exceeds the per-request prompt limit")
    assert error["message"].endswith("of 512 tokens. Please shorten your prompt.")
    assert worker.handed_off == []

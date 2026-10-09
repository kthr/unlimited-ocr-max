"""KON-215: N requests per scheduler step -- the batch processor, ``execute``, load-independent padding, eager warm-up.

With ``--max-batch-size N > 1`` a prefill batch runs as one batch-1 prefill
per context and a decode batch as ONE ``pipeline.decode_rows`` execute; every
decode step runs a graph of at least two rows (a lone request is padded with a
dummy row), so a request's output does not depend on load, and the
``B = 2 .. N`` decode graphs are loaded at startup. ``N == 1`` is the batch-1
path exactly as before.

Model-free except the slow test: ``execute`` runs against a fake pipeline
(what is under test is which pipeline calls it makes, in what order, with which
per-context inputs), ``__init__`` over a fake base class and pipeline, the
batch processor against stand-in contexts, and the padding against a stub
``_execute`` on caches in a CPU page pool, reusing ``test_batched_decode``'s.
A padding row is a row on the pool's null page (KON-237): it takes no pages and
is the same row every step. The slow test is the invariant the padding exists
for, on an accelerator at a small config with random weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.graph import DeviceRef
from max.pipelines.lib.interfaces.pipeline_model import PipelineModelWithKVCache
from max.pipelines.request import RequestID

import unlimited_ocr_max.model as model_module
import unlimited_ocr_max.pipeline as pipeline_module
from unlimited_ocr_max import cli
from unlimited_ocr_max.batch_processor import UnlimitedOcrBatchProcessor
from unlimited_ocr_max.graphs import DecodeGraph
from unlimited_ocr_max.kv_cache import KvPagePool
from unlimited_ocr_max.model import (
    MAX_BATCH_SIZE_ENV,
    ServedRequest,
    UnlimitedOcrArchConfig,
    UnlimitedOcrInputs,
    UnlimitedOCRModel,
)
from unlimited_ocr_max.pipeline import PADDING_POSITION, PADDING_TOKEN_ID, PrefillResult, UnlimitedOcrPipeline

from test_batched_decode import (
    VOCAB,
    _UNREAD,
    _assert_pool_inputs,
    _cpu_pipeline,
    _host_cache,
    _paged_cache,
    _random_rows,
    _small_pipeline,
    _stub_graph,
    _whole,
)
from test_weight_sharing import gpu_only

# --------------------------------------------------------------------------
# stand-ins: contexts for the batch processor, a pipeline for execute
# --------------------------------------------------------------------------

PAGE_SHAPE = (1, 3, 4, 4)


def _context(request_id: str, tokens: list[int], *, page: np.ndarray | None = None, indices: Any = None) -> Any:
    """What the batch processor reads off a ``TextAndVisionContext``; a page means it still needs vision encoding."""
    active = np.asarray(tokens, dtype=np.int64)
    return SimpleNamespace(
        request_id=request_id,
        tokens=SimpleNamespace(active=active, active_length=int(active.shape[0])),
        needs_vision_encoding=page is not None,
        next_images=[] if page is None else [SimpleNamespace(pixel_values=page)],
        extra_model_args={} if indices is None else {"image_token_indices": indices},
    )


def _prompt(b: int) -> list[int]:
    """Context ``b``'s prompt: its own length, and a first token that names it."""
    return [100 + b, *range(1, 4 + b)]


def _page(b: int) -> np.ndarray:
    """Context ``b``'s page, every pixel ``b + 1``; context 1's is rank 3, which the processor must lift to rank 4."""
    page = np.full(PAGE_SHAPE, float(b + 1), dtype=np.float32)
    return page[0] if b == 1 else page


def _placeholders(b: int) -> np.ndarray:
    """Context ``b``'s placeholder rows: ``b + 2`` of them, so a misaligned list fails the model's row check."""
    return np.arange(5, 5 + b + 2, dtype=np.int64)


def _prefill_contexts(n: int) -> list[Any]:
    return [_context(f"r{b}", _prompt(b), page=_page(b), indices=_placeholders(b)) for b in range(n)]


def _decode_contexts(tokens: dict[str, int]) -> list[Any]:
    return [_context(request_id, [token]) for request_id, token in tokens.items()]


def _processor() -> UnlimitedOcrBatchProcessor:
    return UnlimitedOcrBatchProcessor(SimpleNamespace(), SimpleNamespace(devices=[CPU()]))


def _inputs(contexts: list[Any]) -> UnlimitedOcrInputs:
    return _processor().prepare_initial_token_inputs([contexts])


def _row_logits(first_token: int) -> np.ndarray:
    """A logits row that names the token it answers."""
    return np.arange(VOCAB, dtype=np.float32) + np.float32(1000 * first_token)


class _FakePipeline:
    """The pipeline as ``execute`` sees it in the served bf16 accelerator configuration; records every call."""

    releases_language_graphs = False

    def __init__(self, blocker: Any = None) -> None:
        self.ngram_blocker = blocker
        self.calls: list[tuple[Any, ...]] = []
        #: Every ``decode_rows`` call's histories, copied as they were at the call.
        self.histories: list[list[list[int]]] = []

    def retain_only_prefill(self, seq_len: int) -> None:
        self.calls.append(("retain_only_prefill", seq_len))

    def release_decode(self) -> None:
        raise AssertionError("the served bf16 path holds its graphs")

    def release_prefill(self) -> None:
        raise AssertionError("the served bf16 path holds its graphs")

    def run_vision(self, pixels: np.ndarray) -> dict[str, np.ndarray]:
        page = int(pixels.flat[0])  # `_page(b)` is filled with b + 1
        self.calls.append(("run_vision", page, pixels.shape))
        return {"image_embeds": np.zeros((page + 1, 4), dtype=np.float32)}  # as many rows as `_placeholders(page - 1)`

    def drop_vision_weights(self) -> None:
        self.calls.append(("drop_vision_weights",))

    def run_prefill(self, tokens: np.ndarray, image_embeds: np.ndarray) -> PrefillResult:
        ids = [int(token) for token in tokens]
        self.calls.append(("run_prefill", ids))
        name = f"cache-{ids[0]}"
        cache = SimpleNamespace(name=name, release=lambda: self.calls.append(("release", name)))
        return PrefillResult(logits=_row_logits(ids[0]), cache=cache)

    def decode_rows(self, caches: list[Any], token_ids: list[int], histories: list[Any]) -> np.ndarray:
        self.calls.append(("decode_rows", [cache.name for cache in caches], list(token_ids)))
        self.histories.append([list(history) for history in histories])
        return np.stack([_row_logits(token) for token in token_ids])

    def decode_step(self, *_args: Any) -> np.ndarray:
        raise AssertionError("the model decodes through decode_rows")


def _model(pipeline: Any, *, devices: list[Any] | None = None, sample_on_host: bool = True) -> UnlimitedOCRModel:
    model = object.__new__(UnlimitedOCRModel)
    model._pipeline = pipeline
    model._served = {}
    model.devices = [CPU()] if devices is None else devices
    model.pipeline_config = SimpleNamespace(sampling=SimpleNamespace(sample_on_host=sample_on_host))
    return model


def _logits(outputs: Any) -> np.ndarray:
    assert outputs.logits is outputs.next_token_logits
    return outputs.next_token_logits.to(CPU()).to_numpy()


def _expected_prefill_calls(n: int) -> list[tuple[Any, ...]]:
    """Today's batch-1 ``_prefill`` sequence (hold-both ``retain_only_prefill`` first), once per context, in order."""
    calls: list[tuple[Any, ...]] = []
    for b in range(n):
        prompt = _prompt(b)
        calls += [
            ("retain_only_prefill", len(prompt)),
            ("run_vision", b + 1, PAGE_SHAPE),
            ("drop_vision_weights",),
            ("run_prefill", prompt),
        ]
    return calls


# --------------------------------------------------------------------------
# the batch processor: N contexts, kept per context
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 3])
def test_the_batch_processor_keeps_a_prefill_batch_per_context(n: int) -> None:
    """Concatenated tokens split by ``token_counts``, request ids, and one pixel / placeholder buffer per context."""
    inputs = _inputs(_prefill_contexts(n))
    prompts = [_prompt(b) for b in range(n)]

    assert inputs.request_ids == tuple(f"r{b}" for b in range(n))
    assert inputs.token_counts == tuple(len(prompt) for prompt in prompts)
    tokens = inputs.tokens.to_numpy()
    assert tokens.dtype == np.int64 and tokens.tolist() == [token for prompt in prompts for token in prompt]

    assert inputs.has_vision_inputs
    assert inputs.pixel_values is not None and len(inputs.pixel_values) == n
    for b, buffer in enumerate(inputs.pixel_values):
        pixels = buffer.to_numpy()
        assert pixels.dtype == np.float32 and pixels.shape == PAGE_SHAPE, b
        assert np.all(pixels == b + 1), b
    # Each context's own indices, into its own tokens: no offset across the batch.
    assert inputs.image_token_indices is not None and len(inputs.image_token_indices) == n
    for b, buffer in enumerate(inputs.image_token_indices):
        indices = buffer.to_numpy()
        assert indices.dtype == np.int32 and indices.tolist() == _placeholders(b).tolist(), b


def test_the_batch_processor_builds_a_decode_batch_without_vision_inputs() -> None:
    inputs = _inputs(_decode_contexts({"r2": 7, "r0": 8, "r1": 9}))
    assert inputs.request_ids == ("r2", "r0", "r1")
    assert inputs.token_counts == (1, 1, 1)
    assert inputs.tokens.to_numpy().tolist() == [7, 8, 9]
    assert inputs.pixel_values is None and inputs.image_token_indices is None
    assert not inputs.has_vision_inputs


def test_the_batch_processor_refuses_what_it_cannot_split() -> None:
    processor = _processor()
    with pytest.raises(ValueError, match="data parallelism"):
        processor.prepare_initial_token_inputs([_prefill_contexts(1), _prefill_contexts(1)])
    with pytest.raises(ValueError, match="empty batch"):
        processor.prepare_initial_token_inputs([[]])
    # A prefill batch whose contexts do not all carry a page cannot be run one prefill per context.
    mixed = [*_prefill_contexts(1), _context("text-only", [1, 2, 3])]
    with pytest.raises(ValueError, match="1 of 2 requests in a prefill batch carry no page"):
        processor.prepare_initial_token_inputs([mixed])
    two_pages = _prefill_contexts(2)
    two_pages[1].next_images = two_pages[1].next_images * 2
    with pytest.raises(ValueError, match="exactly one page per request, got 2"):
        processor.prepare_initial_token_inputs([two_pages])
    partial = _prefill_contexts(2)
    partial[0].extra_model_args = {}
    with pytest.raises(ValueError, match="1 of 2 requests in a prefill batch carry no image_token_indices"):
        processor.prepare_initial_token_inputs([partial])


# --------------------------------------------------------------------------
# execute: a CE batch of N prefills, a TG batch of one decode_rows call
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_prefill_batch_runs_one_batch1_prefill_per_context_in_order(n: int) -> None:
    """N prefills, in batch order, each today's batch-1 path, context ``b`` with its own pixels and placeholders."""
    pipeline = _FakePipeline()
    model = _model(pipeline)
    logits = _logits(model.execute(_inputs(_prefill_contexts(n))))

    assert pipeline.calls == _expected_prefill_calls(n)
    assert logits.shape == (n, VOCAB)
    for b in range(n):
        assert np.array_equal(logits[b], _row_logits(_prompt(b)[0])), b
        served = model._served[f"r{b}"]
        assert served.cache.name == f"cache-{_prompt(b)[0]}" and served.sequence == _prompt(b)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_a_decode_batch_calls_decode_rows_once_in_context_order(n: int) -> None:
    """ONE ``decode_rows`` call, rows in the step's context order (not the prefill order), each sequence one longer."""
    pipeline = _FakePipeline()
    model = _model(pipeline)
    model.execute(_inputs(_prefill_contexts(n)))
    pipeline.calls.clear()

    step = {f"r{b}": 50 + b for b in reversed(range(n))}
    logits = _logits(model.execute(_inputs(_decode_contexts(step))))

    order = list(step)
    assert pipeline.calls == [("decode_rows", [f"cache-{int(name[1:]) + 100}" for name in order], list(step.values()))]
    assert logits.shape == (n, VOCAB)
    for b, (request_id, token) in enumerate(step.items()):
        assert np.array_equal(logits[b], _row_logits(token)), b
        assert model._served[request_id].sequence == [*_prompt(int(request_id[1:])), token]
    # Row b's history is its own request's sequence, the new token already appended.
    assert pipeline.histories == [[model._served[request_id].sequence for request_id in order]]


def test_a_decode_batch_checks_every_request_before_any_sequence_moves() -> None:
    """Today's two refusals, word for word; whichever row trips them, no request's sequence has moved."""
    pipeline = _FakePipeline()
    model = _model(pipeline)
    model.execute(_inputs(_prefill_contexts(2)))
    pipeline.calls.clear()
    before = {request_id: list(state.sequence) for request_id, state in model._served.items()}

    with pytest.raises(ValueError, match="request r9 asked for a decode step with no prefill on record"):
        model.execute(_inputs(_decode_contexts({"r0": 5, "r9": 6})))
    two_tokens = [_context("r0", [5]), _context("r1", [6, 7])]
    with pytest.raises(ValueError, match="this decoder steps one token at a time; the scheduler asked for 2"):
        model.execute(_inputs(two_tokens))
    with pytest.raises(ValueError, match="request r0 appears twice in one decode batch"):
        model.execute(_inputs([_context("r0", [5]), _context("r0", [6])]))
    assert pipeline.calls == []
    assert {request_id: state.sequence for request_id, state in model._served.items()} == before


def test_release_gives_a_requests_pages_back_once() -> None:
    """``release`` hands the finished request's cache back to the pool, once; an unknown or released id is a no-op.

    A request prefilled again releases its first cache before the new prefill
    allocates, so a re-prefill neither strands pages nor needs twice as many.
    """
    pipeline = _FakePipeline()
    model = _model(pipeline)
    model.execute(_inputs(_prefill_contexts(2)))
    pipeline.calls.clear()

    model.release("r0")
    model.release("r0")
    model.release("never-seen")
    assert pipeline.calls == [("release", "cache-100")]
    assert list(model._served) == ["r1"]

    pipeline.calls.clear()
    model.execute(_inputs([_prefill_contexts(2)[1]]))
    assert [call for call in pipeline.calls if call[0] in ("release", "run_prefill")] == [
        ("release", "cache-101"),
        ("run_prefill", _prompt(1)),
    ]


class _PoolPipeline(_FakePipeline):
    """:class:`_FakePipeline` whose prefill takes real pages: a CPU pool of exactly two one-page slots."""

    def __init__(self) -> None:
        super().__init__()
        self.kv_pool = KvPagePool(num_layers=1, num_kv_heads=1, head_dim=2, pages=2 + 1)

    def run_prefill(self, tokens: np.ndarray, image_embeds: np.ndarray) -> PrefillResult:
        ids = [int(token) for token in tokens]
        self.calls.append(("run_prefill", ids))
        return PrefillResult(logits=_row_logits(ids[0]), cache=self.kv_pool.allocate(prefill_len=len(ids), window=4))


def test_release_by_maxs_request_id_gives_the_pages_back() -> None:
    """MAX releases a request by its ``RequestID`` object, not the ``str`` the model keys it by; the pages come back.

    The contexts carry ``RequestID``s, as MAX's do, and ``release`` gets the
    same objects, as ``TextGenerationPipeline.release`` passes them on. Three
    rounds of two requests through a pool of exactly two slots: were a
    ``RequestID`` release a no-op (the ``request_key`` leak fixed before
    KON-237), the second round's prefill would find no free page.
    """
    pipeline = _PoolPipeline()
    model = _model(pipeline)
    for round_ in range(3):
        request_ids = [RequestID(f"round{round_}-r{b}") for b in range(2)]
        contexts = [
            _context(request_id, _prompt(b), page=_page(b), indices=_placeholders(b))
            for b, request_id in enumerate(request_ids)
        ]
        model.execute(_inputs(contexts))
        assert pipeline.kv_pool.free_pages == 0
        assert sorted(model._served) == [str(request_id) for request_id in request_ids]
        for request_id in request_ids:
            model.release(request_id)
        assert pipeline.kv_pool.free_pages == 2 and model._served == {}


def test_execute_refuses_inputs_the_batch_processor_did_not_build() -> None:
    model = _model(_FakePipeline())
    tokens = Buffer.from_numpy(np.asarray([1, 2, 3], dtype=np.int64))
    with pytest.raises(ValueError, match="carries no request ids"):
        model.execute(UnlimitedOcrInputs(tokens=tokens, token_counts=(3,)))
    with pytest.raises(ValueError, match="token_counts"):
        model.execute(UnlimitedOcrInputs(tokens=tokens, request_ids=("r0",)))
    with pytest.raises(ValueError, match="token_counts"):
        model.execute(UnlimitedOcrInputs(tokens=tokens, token_counts=(1, 1), request_ids=("r0", "r1")))


class _RecordingBlocker:
    """The n-gram guard's ``apply``: records each row's sequence and stamps the row with that sequence's sum."""

    def __init__(self) -> None:
        self.seen: list[list[int]] = []

    def apply(self, logits: np.ndarray, sequence: list[int]) -> np.ndarray:
        assert np.asarray(logits).shape == (VOCAB,)
        self.seen.append(list(sequence))
        return np.full(VOCAB, float(sum(sequence)), dtype=np.float32)


def test_the_guard_sees_only_its_own_rows_sequence() -> None:
    """Per row, against that row's own request's sequence: prefill rows through ``blocker.apply``, the
    result landing in that row; decode rows never through it -- ``decode_rows`` gets each row's own
    sequence for the decode graph's in-graph guard, and its logits come back untouched."""
    blocker = _RecordingBlocker()
    pipeline = _FakePipeline(blocker)
    model = _model(pipeline)

    prefill = _logits(model.execute(_inputs(_prefill_contexts(3))))
    assert blocker.seen == [_prompt(b) for b in range(3)]
    assert [row[0] for row in prefill] == [float(sum(_prompt(b))) for b in range(3)]

    blocker.seen.clear()
    step = {"r1": 61, "r2": 62, "r0": 60}
    decode = _logits(model.execute(_inputs(_decode_contexts(step))))
    own = [[*_prompt(int(request_id[1:])), token] for request_id, token in step.items()]
    assert blocker.seen == []
    assert pipeline.histories == [own]
    for b, token in enumerate(step.values()):
        assert np.array_equal(decode[b], _row_logits(token)), b


# --------------------------------------------------------------------------
# _logits_buffer: --sample-on-host picks the host CPU over model.devices[0]
# --------------------------------------------------------------------------


class _RecordingBuffer:
    """Stands in for ``max.driver.Buffer``: ``from_numpy(...).to(target)`` records ``target``, moves nothing."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.target: Any = None

    @staticmethod
    def from_numpy(array: np.ndarray) -> _RecordingBuffer:
        return _RecordingBuffer(array)

    def to(self, target: Any) -> _RecordingBuffer:
        self.target = target
        return self


@pytest.mark.parametrize("sample_on_host", [True, False])
def test_logits_buffer_targets_host_or_first_device_per_sample_on_host(
    monkeypatch: pytest.MonkeyPatch, sample_on_host: bool
) -> None:
    """``_logits_buffer``'s target: the host CPU when ``sample_on_host``, else ``model.devices[0]`` unchanged."""
    monkeypatch.setattr(model_module, "Buffer", _RecordingBuffer)
    device_stand_in = object()
    model = _model(_FakePipeline(), devices=[device_stand_in], sample_on_host=sample_on_host)

    buffer = model._logits_buffer(np.stack([_row_logits(0)]))

    assert buffer.target is (CPU() if sample_on_host else device_stand_in)


# --------------------------------------------------------------------------
# N == 1: the batch-1 graph, as before
# --------------------------------------------------------------------------


def test_batch_size_one_decodes_on_the_batch1_graph_with_no_padding() -> None:
    """``min_decode_rows`` defaults to 1: a lone request's decode is one row through the batch-1 graph."""
    pipeline = _cpu_pipeline(window=4)
    assert pipeline.min_decode_rows == 1
    execute = _stub_graph(pipeline, 1)
    model = _model(pipeline)
    cache = _host_cache(pipeline, 3, seed=0)
    model._served["r0"] = ServedRequest(cache=cache, sequence=[1, 2, 3])

    logits = _logits(model.execute(_inputs(_decode_contexts({"r0": 5}))))
    (call,) = execute.calls
    assert call[0] == "model-b1" and call[2][0].tolist() == [5] and call[2][1].tolist() == [3]
    assert logits.shape == (1, VOCAB) and np.array_equal(logits[0], execute.outputs["logits"][0])
    assert model._served["r0"].sequence == [1, 2, 3, 5]
    assert cache.position == 4


# --------------------------------------------------------------------------
# padding: a lone row runs the 2-row graph beside a row on the null page
# --------------------------------------------------------------------------


def _padded_pipeline() -> tuple[UnlimitedOcrPipeline, Any]:
    pipeline = _cpu_pipeline(window=4)
    pipeline.min_decode_rows = 2
    return pipeline, _stub_graph(pipeline, 2)


def test_a_lone_row_runs_the_two_row_graph_and_gets_one_row_back() -> None:
    """The padding row comes after the real one: :data:`PADDING_TOKEN_ID` at :data:`PADDING_POSITION`, on the null page."""
    pipeline, execute = _padded_pipeline()
    cache = _host_cache(pipeline, 3, seed=0)
    free = pipeline.kv_pool.free_pages
    want = pipeline.kv_pool.step([cache], padding=1)

    logits = pipeline.decode_rows([cache], [5], [_UNREAD])

    assert logits.shape == (1, VOCAB)
    assert np.array_equal(logits[0], execute.outputs["logits"][0])
    (call,) = execute.calls
    model, names, arrays = call
    assert (model, names, len(arrays)) == ("model-b2", ("logits",), 11)
    assert arrays[0].tolist() == [5, PADDING_TOKEN_ID]
    assert arrays[1].tolist() == [3, PADDING_POSITION]
    _assert_pool_inputs(arrays, pipeline.kv_pool, want)
    null = pipeline.kv_pool.null_page
    assert want.lookup_table[0, 0] == cache.pages[0] and (want.lookup_table[1] == null).all()
    assert want.attend_lengths.tolist() == [3, 0]  # attend_len - 1: the padding row attends its one row
    assert want.write_index.tolist() == [3, 0]  # write index: the padding row stores at the null page's slot 0
    assert (cache.length, cache.position) == (4, 4)
    assert pipeline.kv_pool.free_pages == free  # padding takes no pages


def test_a_padding_row_gets_an_all_zero_history() -> None:
    """Guard on: the real row's history in row 0, zeros in the padding row 1 (its logits are discarded), one ``n``."""
    pipeline, execute = _padded_pipeline()
    window = pipeline.window
    pipeline.ngram_size = 3  # after construction: no compile
    sequence = [*range(20, 20 + window + 2), 5]

    logits = pipeline.decode_rows([_host_cache(pipeline, 3, seed=0)], [5], [sequence])

    assert logits.shape == (1, VOCAB)
    history, ngram = execute.calls[0][2][-2:]
    assert history.dtype == np.int32 and history.shape == (2, window)
    assert history[0].tolist() == sequence[-window:] and not history[1].any()
    assert ngram.tolist() == [3]


def test_no_padding_at_or_above_min_decode_rows() -> None:
    pipeline, execute = _padded_pipeline()
    caches = [_host_cache(pipeline, 3, seed=0), _host_cache(pipeline, 5, seed=1)]
    want = pipeline.kv_pool.step(caches)
    logits = pipeline.decode_rows(caches, [5, 6], [_UNREAD] * 2)
    assert logits is execute.outputs["logits"]
    arrays = execute.calls[0][2]
    assert arrays[0].tolist() == [5, 6]
    _assert_pool_inputs(arrays, pipeline.kv_pool, want)
    assert not (want.lookup_table == pipeline.kv_pool.null_page).all(axis=1).any()  # no row sits on the null page alone


def test_decode_rows_refuses_the_same_cache_twice() -> None:
    pipeline, execute = _padded_pipeline()
    cache, other = _host_cache(pipeline, 3, seed=0), _host_cache(pipeline, 4, seed=1)
    for caches in ([cache, cache], [cache, other, cache]):
        with pytest.raises(ValueError, match="the same cache appears more than once"):
            pipeline.decode_rows(caches, [1] * len(caches), [_UNREAD] * len(caches))
    assert execute.calls == []
    assert (cache.position, other.position) == (3, 4)


# --------------------------------------------------------------------------
# eager warm-up: the pipeline's, and the model's init
# --------------------------------------------------------------------------


def test_warm_decode_graphs_loads_b2_to_n_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _cpu_pipeline(window=4)
    built: list[int] = []

    def build(config, decoder, **kwargs):
        built.append(kwargs["batch"])
        return DecodeGraph(graph=f"graph-b{kwargs['batch']}", output_names=(), batch=kwargs["batch"])

    monkeypatch.setattr(pipeline_module, "build_decode_graph", build)
    pipeline._decoder = lambda: SimpleNamespace(state_dict=dict)
    pipeline.session = SimpleNamespace(load=lambda graph, *, weights_registry: f"model-{graph}")

    for below_two in (1, 0):
        pipeline.warm_decode_graphs(below_two)
    assert built == []
    pipeline.warm_decode_graphs(4)
    assert built == [2, 3, 4]
    assert list(pipeline._decode) == [2, 3, 4]
    pipeline.decode_graph(3)  # warmed: served from the cache, not built again
    assert built == [2, 3, 4]


class _RecordingPipeline:
    """Stands in for ``UnlimitedOcrPipeline`` in ``UnlimitedOCRModel.__init__``; ``events`` is what it was asked, in order."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.min_decode_rows = 1
        self.warmed: list[tuple[int, int]] = []
        self.events: list[str] = []

    def warm_decode_graphs(self, max_batch: int) -> None:
        self.warmed.append((max_batch, self.min_decode_rows))
        self.events.append("warm_decode_graphs")

    @property
    def kv_pool(self) -> str:
        self.events.append("kv_pool")
        return "pool"


def _init_model(
    monkeypatch: pytest.MonkeyPatch,
    max_batch_size: int,
    scheduler_max_batch_size: int | None = None,
    *,
    int8: bool = False,
    device: DeviceRef = DeviceRef.GPU(0),
) -> UnlimitedOCRModel:
    """``UnlimitedOCRModel.__init__`` itself, over a fake base init, arch config, prompt length and
    pipeline. ``int8`` stands in for an int8 checkpoint without needing real expert tensors: it
    patches ``_weights_are_int8`` directly and gives the fake arch config the member ``__init__``
    reads only on that path (``model.with_int8_experts()``). ``device`` is the one the fake base
    init resolves: a GPU unless a test says otherwise."""

    def base_init(self, *args: Any, **kwargs: Any) -> None:
        self.max_batch_size = max_batch_size if scheduler_max_batch_size is None else scheduler_max_batch_size
        self.pipeline_config = SimpleNamespace()
        self.devices = [CPU()]
        self.device_refs = [device]
        self.weights = {}
        self.adapter = lambda weights, *, config, image_size: {"vision": {}, "language_model": {}}

    model_config = SimpleNamespace()
    model_config.with_int8_experts = lambda: model_config
    arch = SimpleNamespace(model=model_config, decoder=SimpleNamespace(sliding_window_size=16))
    monkeypatch.setenv(MAX_BATCH_SIZE_ENV, str(max_batch_size))
    monkeypatch.setattr(PipelineModelWithKVCache, "__init__", base_init)
    monkeypatch.setattr(UnlimitedOcrArchConfig, "initialize", classmethod(lambda cls, *args, **kwargs: arch))
    monkeypatch.setattr(UnlimitedOCRModel, "max_seq_len", 400)
    monkeypatch.setattr(UnlimitedOCRModel, "huggingface_config", SimpleNamespace())
    monkeypatch.setattr(UnlimitedOCRModel, "_prompt_len", lambda self, geometry: 277)
    monkeypatch.setattr(UnlimitedOCRModel, "_weights_are_int8", staticmethod(lambda weights: int8))
    monkeypatch.setattr(model_module, "UnlimitedOcrPipeline", _RecordingPipeline)
    return UnlimitedOCRModel(session=SimpleNamespace())


@pytest.mark.parametrize("int8", [False, True])
@pytest.mark.parametrize("max_batch_size", [1, 2, 4, 8])
def test_init_pads_and_warms_only_above_batch_one(
    monkeypatch: pytest.MonkeyPatch, max_batch_size: int, int8: bool
) -> None:
    """N > 1: ``min_decode_rows = 2`` and ONE warm-up of ``B = 2 .. N`` after the build. N == 1:
    neither. Either way N sizes the pipeline's KV page pool, which is built last, at startup, not
    at the first prefill. int8 and bf16 alike -- KON-227 lifted int8's refusal above batch 1, and
    this is the same init path either checkpoint takes."""
    model = _init_model(monkeypatch, max_batch_size, int8=int8)
    pipeline = model._pipeline
    assert isinstance(pipeline, _RecordingPipeline)
    if max_batch_size == 1:
        assert (pipeline.min_decode_rows, pipeline.warmed) == (1, [])
        assert pipeline.events == ["kv_pool"]
    else:
        assert (pipeline.min_decode_rows, pipeline.warmed) == (2, [(max_batch_size, 2)])
        assert pipeline.events == ["warm_decode_graphs", "kv_pool"]
    assert pipeline.kwargs["max_batch_size"] == max_batch_size  # it sizes the KV page pool
    assert model._served == {}


@pytest.mark.parametrize(("env", "scheduler"), [(1, 8), (8, 1), (4, 8)])
def test_init_refuses_a_scheduler_batch_size_the_env_did_not_set(
    monkeypatch: pytest.MonkeyPatch, env: int, scheduler: int
) -> None:
    """``max serve --force`` skips ``required_arguments``; the worker must not serve a batch size it did not warm."""
    with pytest.raises(ValueError, match=f"the scheduler batches up to {scheduler} requests but"):
        _init_model(monkeypatch, env, scheduler_max_batch_size=scheduler)


def test_init_refuses_a_cpu_device_with_the_cli_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain ``max serve --devices cpu`` stops here (KON-232): the refusal reads the resolved device
    -- the fake pipeline config carries no ``--devices`` flag at all -- and gives the CLI's reason."""
    with pytest.raises(ValueError) as refused:
        _init_model(monkeypatch, 1, device=DeviceRef.CPU())
    assert str(refused.value) == cli.CPU_REFUSED


# --------------------------------------------------------------------------
# on the accelerator: a padded lone row computes the bits it computes beside real rows
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def _accelerator_session():
    from max.driver import Accelerator
    from max.engine import InferenceSession

    driver = Accelerator()
    return driver, InferenceSession(devices=[driver])


@pytest.mark.slow
@gpu_only
def test_a_padded_lone_row_computes_the_bits_it_computes_beside_real_rows(_accelerator_session) -> None:
    """The load-independence invariant, asserted bitwise over three steps, logits and KV alike.

    Row 0 decoded alone with ``min_decode_rows = 2`` (so beside a padding row
    on the pool's null page, on the 2-row graph) against the same row, from
    identical starting caches in fresh pages, beside real rows: one at B = 2
    and two at B = 3, in more than one row order. The pipeline is the shipped
    accelerator configuration -- the shared device registry (bf16
    projections, fp32-resident norms and router), the page pool on the device -- on
    ``_small_decoder_config`` with random weights, as in
    ``test_batched_decode``'s slow tests. Row 0 is warming up (appending), row
    1 has a full ring (overwriting), row 2 has a short fresh prefix, so the
    rows differ in cache length, position and write mode. The padding row
    takes no pages, three steps running.
    """
    pipeline = _small_pipeline(_accelerator_session, int8=False, seed=41)
    pipeline.min_decode_rows = 2
    small = pipeline.config.decoder
    window = small.sliding_window_size
    states = [
        {"prefill_len": 5, "length": 7, "ring_pos": 0, "position": 7},
        {"prefill_len": 9, "length": 9 + window, "ring_pos": 5, "position": 40},
        {"prefill_len": 3, "length": 3, "ring_pos": 0, "position": 3},
    ]
    rng = np.random.default_rng(47)
    host_rows = [_random_rows(rng, pipeline, state["prefill_len"]) for state in states]
    tokens = [[3, 9, 27], [17, 5, 44], [60, 2, 11]]  # tokens[row][step]
    steps = len(tokens[0])
    free = pipeline.kv_pool.free_pages

    def run(order: list[int]) -> tuple[np.ndarray, list[np.ndarray], tuple[int, int, int]]:
        """Row 0's logits at every step, then its whole cache and ring state after the last."""
        caches = {row: _paged_cache(pipeline, host_rows[row], **states[row]) for row in order}
        assert pipeline.kv_pool.free_pages == free - len(order)
        logits = []
        for step in range(steps):
            out = pipeline.decode_rows(
                [caches[row] for row in order], [tokens[row][step] for row in order], [_UNREAD] * len(order)
            )
            assert out.shape == (len(order), small.vocab_size)
            logits.append(np.array(out[order.index(0)]))
        mine = caches[0]
        kv = _whole(mine)
        state = (mine.length, mine.ring_pos, mine.position)
        for cache in caches.values():
            cache.release()
        return np.stack(logits), kv, state

    alone_logits, alone_kv, alone_state = run([0])
    assert pipeline.kv_pool.free_pages == free
    assert np.all(np.isfinite(alone_logits))

    for order in ([0, 1], [1, 0], [0, 1, 2], [2, 0, 1]):
        logits, kv, state = run(order)
        assert state == alone_state == (7 + steps, 0, 7 + steps), order
        delta = max(
            float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
            for a, b in zip([logits, *kv], [alone_logits, *alone_kv], strict=True)
        )
        print(f"[uocr] padded lone row vs beside real rows {order}: max |delta| {delta:.3e}")
        assert np.array_equal(logits, alone_logits), (order, delta)
        for got, want in zip(kv, alone_kv, strict=True):
            assert np.array_equal(got, want), (order, delta)

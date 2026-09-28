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
``_execute`` on host caches, reusing ``test_batched_decode``'s (one check
needs an accelerator for its tiny device buffers, still no compile). The slow
test is the invariant the padding exists for, on an accelerator at a small
config with random weights.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from max.driver import CPU, Buffer
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib.interfaces.pipeline_model import PipelineModelWithKVCache

import unlimited_ocr_max.model as model_module
import unlimited_ocr_max.pipeline as pipeline_module
from unlimited_ocr_max.batch_processor import UnlimitedOcrBatchProcessor
from unlimited_ocr_max.graphs import DecodeGraph
from unlimited_ocr_max.kv_cache import DeviceKvCache, KvCache, to_host
from unlimited_ocr_max.model import (
    MAX_BATCH_SIZE_ENV,
    ServedRequest,
    UnlimitedOcrArchConfig,
    UnlimitedOcrInputs,
    UnlimitedOCRModel,
)
from unlimited_ocr_max.pipeline import PADDING_PREFIX_LEN, PADDING_TOKEN_ID, PrefillResult, UnlimitedOcrPipeline

from test_batched_decode import (
    HEAD_DIM,
    HEADS,
    LAYERS,
    VOCAB,
    _BatchedExecute,
    _device_cache,
    _host_cache,
    _output_names,
    _random_language_weights,
    _refuse,
)
from test_decoder_int8 import _small_decoder_config
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
    on_accelerator = True

    def __init__(self, blocker: Any = None) -> None:
        self.ngram_blocker = blocker
        self.calls: list[tuple[Any, ...]] = []

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
        return PrefillResult(logits=_row_logits(ids[0]), cache=SimpleNamespace(name=f"cache-{ids[0]}"))

    def decode_rows(self, caches: list[Any], token_ids: list[int]) -> np.ndarray:
        self.calls.append(("decode_rows", [cache.name for cache in caches], list(token_ids)))
        return np.stack([_row_logits(token) for token in token_ids])

    def decode_step(self, *_args: Any) -> np.ndarray:
        raise AssertionError("the model decodes through decode_rows")


def _model(pipeline: Any) -> UnlimitedOCRModel:
    model = object.__new__(UnlimitedOCRModel)
    model._pipeline = pipeline
    model._served = {}
    model.devices = [CPU()]
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
    """Per row, against that row's own request's sequence, the result landing in that row -- at prefill and decode."""
    blocker = _RecordingBlocker()
    model = _model(_FakePipeline(blocker))

    prefill = _logits(model.execute(_inputs(_prefill_contexts(3))))
    assert blocker.seen == [_prompt(b) for b in range(3)]
    assert [row[0] for row in prefill] == [float(sum(_prompt(b))) for b in range(3)]

    blocker.seen.clear()
    step = {"r1": 61, "r2": 62, "r0": 60}
    decode = _logits(model.execute(_inputs(_decode_contexts(step))))
    own = [[*_prompt(int(request_id[1:])), token] for request_id, token in step.items()]
    assert blocker.seen == own
    assert [row[0] for row in decode] == [float(sum(sequence)) for sequence in own]


# --------------------------------------------------------------------------
# N == 1: the batch-1 graph, as before
# --------------------------------------------------------------------------


def _stub_pipeline(device: DeviceRef | None = None, driver: Any = None) -> UnlimitedOcrPipeline:
    """A real pipeline at ``test_batched_decode``'s stub dimensions; nothing is ever compiled."""
    decoder = SimpleNamespace(
        num_hidden_layers=LAYERS, num_key_value_heads=HEADS, head_dim=HEAD_DIM, vocab_size=VOCAB, sliding_window_size=4
    )
    return UnlimitedOcrPipeline(
        SimpleNamespace(decoder=decoder, dtype=DType.bfloat16),
        vision_state_dict={},
        language_state_dict={},
        seq_len=8,
        max_new_tokens=8,
        device=device or DeviceRef.CPU(),
        driver_device=driver,
        session=SimpleNamespace(),
    )


def test_batch_size_one_decodes_on_the_batch1_graph_with_no_padding() -> None:
    """``min_decode_rows`` defaults to 1: a lone request's decode is ``decode_step``, the call today's model made."""
    pipeline = _stub_pipeline()
    assert pipeline.min_decode_rows == 1
    calls: list[tuple[KvCache, int]] = []

    def decode_step(cache: KvCache, token_id: int) -> np.ndarray:
        calls.append((cache, token_id))
        return _row_logits(token_id)

    pipeline.decode_step = decode_step
    pipeline.batched_decode_graph = _refuse
    pipeline._execute = _refuse
    model = _model(pipeline)
    cache = _host_cache(3, seed=0)
    model._served["r0"] = ServedRequest(cache=cache, sequence=[1, 2, 3])

    logits = _logits(model.execute(_inputs(_decode_contexts({"r0": 5}))))
    assert calls == [(cache, 5)]
    assert logits.shape == (1, VOCAB) and np.array_equal(logits[0], _row_logits(5))
    assert model._served["r0"].sequence == [1, 2, 3, 5]
    assert pipeline._padding_caches == []


# --------------------------------------------------------------------------
# padding: a lone row runs the 2-row graph, the dummy never grows
# --------------------------------------------------------------------------


def _padded_pipeline() -> tuple[UnlimitedOcrPipeline, _BatchedExecute, DecodeGraph]:
    pipeline = _stub_pipeline()
    pipeline.min_decode_rows = 2
    staged = DecodeGraph(graph=None, output_names=_output_names(LAYERS), num_layers=LAYERS, batch=2)
    pipeline._batched_decode[2] = (staged, "model-b2")
    pipeline.decode_step = _refuse
    execute = _BatchedExecute(2)
    pipeline._execute = execute
    return pipeline, execute, staged


def test_a_lone_row_runs_the_two_row_graph_and_gets_one_row_back() -> None:
    pipeline, execute, staged = _padded_pipeline()
    cache = _host_cache(3, seed=0)
    row = cache.write_index

    logits = pipeline.decode_rows([cache], [5])

    assert logits.shape == (1, VOCAB)
    assert np.array_equal(logits[0], execute.outputs["logits"][0])
    assert len(execute.calls) == 1
    model, names, arrays, host_outputs = execute.calls[0]
    assert (model, names, host_outputs) == ("model-b2", staged.output_names, KvCache.HOST_OUTPUTS)
    assert len(arrays) == 2 + 2 * (1 + 2 * LAYERS)
    assert arrays[0].tolist() == [5, PADDING_TOKEN_ID]
    assert arrays[1].tolist() == [3, PADDING_PREFIX_LEN]
    (dummy,) = pipeline._padding_caches
    assert type(dummy) is type(cache)  # a CPU pipeline pads with host rows, like its prefill caches
    dummy_sel = arrays[2 + (1 + 2 * LAYERS)]
    assert dummy_sel.shape == (1, 2, 1) and np.flatnonzero(dummy_sel).tolist() == [1]
    # The real row got row 0 of every output, nothing of the dummy's.
    assert (cache.length, cache.position) == (4, 4)
    for i in range(LAYERS):
        assert np.array_equal(cache.keys[i][row], execute.outputs[f"key_{i}"][0])
        assert np.array_equal(cache.values[i][row], execute.outputs[f"value_{i}"][0])


def test_the_dummy_state_does_not_grow_across_3000_steps() -> None:
    """Allocated once, reset before every padded step: position 1 and two attended rows, however long the run."""
    pipeline, execute, _ = _padded_pipeline()
    cache = _host_cache(3, seed=0)
    dummy_start = 2 + (1 + 2 * LAYERS)
    for step in range(3000):
        pipeline.decode_rows([cache], [step % VOCAB])
    assert len(execute.calls) == 3000
    assert {int(arrays[1][1]) for _, _, arrays, _ in execute.calls} == {PADDING_PREFIX_LEN}
    assert {arrays[dummy_start].shape for _, _, arrays, _ in execute.calls} == {(1, 2, 1)}
    assert {arrays[dummy_start + 1].shape[0] for _, _, arrays, _ in execute.calls} == {2}
    (dummy,) = pipeline._padding_caches
    assert (dummy.length, dummy.prefill_len, dummy.ring_pos, dummy.position) == (2, 1, 0, 2)
    assert dummy.position < pipeline.max_total_len
    assert cache.position == 3 + 3000  # the real row did advance


def test_no_padding_at_or_above_min_decode_rows() -> None:
    pipeline, execute, _ = _padded_pipeline()
    caches = [_host_cache(3, seed=0), _host_cache(5, seed=1)]
    logits = pipeline.decode_rows(caches, [5, 6])
    assert logits is execute.outputs["logits"]
    assert execute.calls[0][2][0].tolist() == [5, 6]
    assert pipeline._padding_caches == []


def test_decode_rows_refuses_the_same_cache_twice() -> None:
    pipeline, execute, _ = _padded_pipeline()
    cache, other = _host_cache(3, seed=0), _host_cache(4, seed=1)
    for caches in ([cache, cache], [cache, other, cache]):
        with pytest.raises(ValueError, match="the same cache appears more than once"):
            pipeline.decode_rows(caches, [1] * len(caches))
    assert execute.calls == []
    assert pipeline._padding_caches == []
    assert (cache.position, other.position) == (3, 4)


@gpu_only
def test_an_accelerator_pipeline_pads_with_a_device_cache() -> None:
    """Of the class its prefill allocates -- device rows -- so ``decode_rows``' one-class rule holds with padding in."""
    from max.driver import Accelerator

    pipeline = _stub_pipeline(DeviceRef.GPU(0), Accelerator())
    (dummy,) = pipeline._padding_rows(1)
    assert isinstance(dummy, DeviceKvCache)
    assert (dummy.capacity, dummy.length, dummy.prefill_len, dummy.ring_pos, dummy.position) == (2, 1, 1, 0, 1)
    assert len(dummy.device_keys) == len(dummy.device_values) == LAYERS
    assert all(np.array_equal(to_host(buf), np.zeros((2, HEADS, HEAD_DIM), np.float32)) for buf in dummy.device_keys)
    assert pipeline._padding_rows(1)[0] is dummy


# --------------------------------------------------------------------------
# eager warm-up: the pipeline's, and the model's init
# --------------------------------------------------------------------------


def test_warm_decode_graphs_loads_b2_to_n_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    pipeline = _stub_pipeline()
    built: list[int] = []

    def build(config, decoder, **kwargs):
        built.append(kwargs["batch"])
        return DecodeGraph(graph=f"graph-b{kwargs['batch']}", output_names=(), num_layers=0, batch=kwargs["batch"])

    monkeypatch.setattr(pipeline_module, "build_batched_decode_graph", build)
    pipeline._decoder = lambda: SimpleNamespace(state_dict=dict)
    pipeline.session = SimpleNamespace(load=lambda graph, *, weights_registry: f"model-{graph}")

    for below_two in (1, 0):
        pipeline.warm_decode_graphs(below_two)
    assert built == []
    pipeline.warm_decode_graphs(4)
    assert built == [2, 3, 4]
    assert list(pipeline._batched_decode) == [2, 3, 4]
    pipeline.batched_decode_graph(3)  # warmed: served from the cache, not built again
    assert built == [2, 3, 4]


class _RecordingPipeline:
    """Stands in for ``UnlimitedOcrPipeline`` in ``UnlimitedOCRModel.__init__``."""

    def __init__(self, config: Any, **kwargs: Any) -> None:
        self.min_decode_rows = 1
        self.warmed: list[tuple[int, int]] = []

    def warm_decode_graphs(self, max_batch: int) -> None:
        self.warmed.append((max_batch, self.min_decode_rows))


def _init_model(monkeypatch: pytest.MonkeyPatch, max_batch_size: int) -> UnlimitedOCRModel:
    """``UnlimitedOCRModel.__init__`` itself, over a fake base init, arch config, prompt length and pipeline."""

    def base_init(self, *args: Any, **kwargs: Any) -> None:
        self.pipeline_config = SimpleNamespace()
        self.devices = [CPU()]
        self.device_refs = [DeviceRef.GPU(0)]
        self.weights = {}
        self.adapter = lambda weights, *, config, image_size: {"vision": {}, "language_model": {}}

    arch = SimpleNamespace(model=SimpleNamespace(), decoder=SimpleNamespace(sliding_window_size=16))
    monkeypatch.setenv(MAX_BATCH_SIZE_ENV, str(max_batch_size))
    monkeypatch.setattr(PipelineModelWithKVCache, "__init__", base_init)
    monkeypatch.setattr(UnlimitedOcrArchConfig, "initialize", classmethod(lambda cls, *args, **kwargs: arch))
    monkeypatch.setattr(UnlimitedOCRModel, "max_seq_len", 400)
    monkeypatch.setattr(UnlimitedOCRModel, "huggingface_config", SimpleNamespace())
    monkeypatch.setattr(UnlimitedOCRModel, "_prompt_len", lambda self, geometry: 277)
    monkeypatch.setattr(model_module, "UnlimitedOcrPipeline", _RecordingPipeline)
    return UnlimitedOCRModel(session=SimpleNamespace())


@pytest.mark.parametrize("max_batch_size", [1, 2, 4, 8])
def test_init_pads_and_warms_only_above_batch_one(monkeypatch: pytest.MonkeyPatch, max_batch_size: int) -> None:
    """N > 1: ``min_decode_rows = 2`` and ONE warm-up of ``B = 2 .. N`` after the build. N == 1: neither."""
    model = _init_model(monkeypatch, max_batch_size)
    pipeline = model._pipeline
    assert isinstance(pipeline, _RecordingPipeline)
    if max_batch_size == 1:
        assert (pipeline.min_decode_rows, pipeline.warmed) == (1, [])
    else:
        assert (pipeline.min_decode_rows, pipeline.warmed) == (2, [(max_batch_size, 2)])
    assert model._served == {}


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

    Row 0 decoded alone with ``min_decode_rows = 2`` (so beside the dummy on
    the 2-row graph) against the same row, from identical starting caches, beside real
    rows: one at B = 2 and two at B = 3, in more than one row order. The
    pipeline is the shipped accelerator configuration -- the shared device
    registry, fp32-resident non-expert weights, device KV caches -- on
    ``_small_decoder_config`` with random weights, as in
    ``test_batched_decode``'s slow test. Row 0 is warming up (appending),
    row 1 has a full ring (overwriting), row 2 has a short fresh prefix, so the
    rows differ in cache length, position and write mode. Three steps also
    run the dummy through its ring-state reset twice.
    """
    driver, session = _accelerator_session
    small = _small_decoder_config(int8=False)
    config = SimpleNamespace(decoder=small, dtype=DType.bfloat16)
    pipeline = UnlimitedOcrPipeline(
        config,
        vision_state_dict={},
        language_state_dict=_random_language_weights(config, seed=41),
        seq_len=32,
        max_new_tokens=32,
        device=DeviceRef.GPU(0),
        driver_device=driver,
        session=session,
    )
    assert pipeline.shares_language_weights
    pipeline.min_decode_rows = 2

    window, layers = small.sliding_window_size, small.num_hidden_layers
    heads, head_dim = small.num_key_value_heads, small.head_dim
    states = [
        {"prefill_len": 5, "length": 7, "ring_pos": 0, "position": 7},
        {"prefill_len": 9, "length": 9 + window, "ring_pos": 5, "position": 40},
        {"prefill_len": 3, "length": 3, "ring_pos": 0, "position": 3},
    ]
    rng = np.random.default_rng(47)
    host_rows = []
    for state in states:
        shape = (state["prefill_len"] + window, heads, head_dim)
        host_rows.append(
            ([rng.standard_normal(shape).astype(np.float32) for _ in range(layers)],
             [rng.standard_normal(shape).astype(np.float32) for _ in range(layers)])
        )
    tokens = [[3, 9, 27], [17, 5, 44], [60, 2, 11]]  # tokens[row][step]
    steps = len(tokens[0])

    def run(order: list[int]) -> tuple[np.ndarray, list[np.ndarray], tuple[int, int, int]]:
        """Row 0's logits at every step, then its whole cache and ring state after the last."""
        caches = {row: _device_cache(driver, config, host_rows[row], **states[row]) for row in order}
        logits = []
        for step in range(steps):
            out = pipeline.decode_rows([caches[row] for row in order], [tokens[row][step] for row in order])
            assert out.shape == (len(order), small.vocab_size)
            logits.append(np.array(out[order.index(0)]))
        mine = caches[0]
        kv = [to_host(buffer) for buffer in (*mine.device_keys, *mine.device_values)]
        return np.stack(logits), kv, (mine.length, mine.ring_pos, mine.position)

    alone_logits, alone_kv, alone_state = run([0])
    (dummy,) = pipeline._padding_caches
    assert isinstance(dummy, DeviceKvCache)
    assert (dummy.length, dummy.position) == (2, 2)
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

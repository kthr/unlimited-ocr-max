"""What the GEMV kernels' vector loads share: the alignment they claim and the host-side choice of load path.

``dense_bf16_qmv``, ``moe_bf16_qmv`` and ``moe_int8_qmv`` each read a weight
and an activation ``width`` elements at a time along K. ``execute`` picks the
call's load path once, host-side (:func:`load_path`), and hands it to every
work item: ``ALIGNED`` loads claim ``LOAD_ALIGN`` bytes (:func:`load_alignment`),
``UNALIGNED`` loads claim one element, ``SCALAR`` reads one element at a time.
"""

from extensibility import InputTensor
from std.sys import size_of

#: Bytes of alignment the vector loads claim on the ``ALIGNED`` path. It is
#: the boundary the buffers MAX allocates itself start on: device tensors at
#: 256 bytes, so a served call always takes that path, and on the CPU MAX's
#: own outputs at 16 bytes, whatever the tensor's static alignment (64) says.
#: A CPU graph input or weight reaches the kernel where the caller's buffer
#: lies, which can be any address (the kernel tests use one element past a
#: 16-byte boundary). On the M4 a larger claim is no faster.
comptime LOAD_ALIGN = 16

#: The load paths of a call, chosen host-side by :func:`load_path` and handed
#: to every work item. ``ALIGNED`` and ``UNALIGNED`` run the same
#: ``width``-wide loop -- the same loads in the same order, so the same sums --
#: and differ only in the alignment the loads claim, ``LOAD_ALIGN`` or one
#: element; so a call's bits do not depend on where its buffers lie.
#: ``SCALAR`` runs the loop one element at a time.
comptime ALIGNED = 0
comptime UNALIGNED = 1
comptime SCALAR = 2


@always_inline
def load_alignment[dtype: DType, width: Int, aligned: Bool = True]() -> Int:
    """The ``element_alignment`` of a ``width``-wide load: with ``aligned``, a load at a multiple of ``width`` elements from a ``LOAD_ALIGN``-aligned base; else one element."""
    comptime if aligned:
        return min(width, LOAD_ALIGN // size_of[dtype]())
    return 1


@always_inline
def packed[
    dtype: DType, rank: Int, //
](t: InputTensor[dtype=dtype, rank=rank, static_spec=_]) -> Bool:
    """Whether ``t`` is contiguous and row-major without padding: each stride is the product of the dims after it.

    A dim of size one moves no offset, so its stride is not checked.
    """
    var expected = 1
    for axis in range(rank - 1, -1, -1):
        var size = Int(t.dim_size(axis))
        if size != 1 and Int(t.stride_length(axis)) != expected:
            return False
        expected *= size
    return True


@always_inline
def load_path[
    wtype: DType, wrank: Int, xtype: DType, xrank: Int, //
](
    run: Int,
    width: Int,
    w: InputTensor[dtype=wtype, rank=wrank, static_spec=_],
    x: InputTensor[dtype=xtype, rank=xrank, static_spec=_],
) -> Int:
    """The load path of a call whose loads read ``width`` elements at a time along runs of ``run`` elements of K.

    ``run`` is K for the bf16 kernels and the group size for int8. Vector loads
    need ``run % width == 0`` and a unit stride along K in ``w`` and ``x``,
    else ``SCALAR``. ``ALIGNED`` also needs both tensors :func:`packed` and both
    bases on ``LOAD_ALIGN``: then every load starts a multiple of ``width``
    elements from its base, which is what the claim assumes. Otherwise
    ``UNALIGNED``.
    """
    if (
        run % width != 0
        or Int(w.stride_length(wrank - 1)) != 1
        or Int(x.stride_length(xrank - 1)) != 1
    ):
        return SCALAR
    if (
        packed(w)
        and packed(x)
        and Int(w.unsafe_ptr()) % LOAD_ALIGN == 0
        and Int(x.unsafe_ptr()) % LOAD_ALIGN == 0
    ):
        return ALIGNED
    return UNALIGNED

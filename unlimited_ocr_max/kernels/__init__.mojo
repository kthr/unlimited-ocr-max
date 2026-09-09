"""Mojo custom ops for the Unlimited-OCR MAX port.

``Graph(custom_extensions=[...])`` takes a directory holding ``__init__.mojo``;
importing the struct here is what makes its ``@register`` reachable.
"""

from .moe_int8 import Int8DequantExpert, MoeInt8Qmv
from .ngram_block import NgramBlock

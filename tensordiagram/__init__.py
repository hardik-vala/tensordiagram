import sys
from typing import TYPE_CHECKING


if sys.version_info >= (3, 8):
    from importlib import metadata
else:
    import importlib_metadata as metadata

from tensordiagram.core import to_diagram
from tensordiagram.types import TensorDiagram

if not TYPE_CHECKING:
    __libname__: str = "tensordiagram"
    __version__: str = metadata.version(__libname__)
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, ClassVar, Literal, Protocol, Optional, Union, TYPE_CHECKING
from typing_extensions import Self

import chalk

if TYPE_CHECKING:
    from dataclasses import Field

Scalar = Union[int, float, str, bool]
TensorLike = Any


_m = lambda a, b: a if a is not None else b


class TensorOrder(Enum):
    R = "r"  # row
    C = "c"  # column
    D = "d"  # depth
    RC = "rc"  # row, then column
    RD = "rd"  # row, then depth
    CR = "cr"  # column, then row
    CD = "cd"  # column, then depth
    DR = "dr"  # depth, then row
    DC = "dc"  # depth, then column
    RCD = "rcd"  # row, then column, then depth
    RDC = "rdc"  # row, then depth, then column
    CRD = "crd"  # column, then row, then depth
    CDR = "cdr"  # column, then depth, then row
    DRC = "drc"  # depth, then row, then column
    DCR = "dcr"  # depth, then column, then row


class Transferable:
    """Mixin class that provides a transfer method for dataclasses.

    This should only be used with @dataclass decorated classes.
    """

    __dataclass_fields__: ClassVar[dict[str, Field[Any]]]

    def transfer(self: Self, other: Self) -> Self:
        """Transfer non-None fields from other to self, creating a new instance."""
        return type(self)(
            *(
                _m(getattr(other, f.name), getattr(self, f.name))
                for f in fields(self)  # type: ignore[arg-type]
            )
        )


@dataclass
class TensorStyle(Transferable):
    cell_size: Optional[float] = None
    color_map: Optional[TensorLike] = None
    opacity_map: Optional[TensorLike] = None
    show_values: Optional[bool] = None


@dataclass
class TensorAnnotation(Transferable):
    color: Optional[str] = None


class Renderable(Protocol):

    def render_png(
        self, path: str, height: Optional[int] = None, width: Optional[int] = None
    ) -> None: ...

    def render_svg(
        self, path: str, height: Optional[int] = None, width: Optional[int] = None
    ) -> None: ...

    def render_pdf(self, path: str, height: Optional[int] = None) -> None: ...

    def _repr_svg_(self) -> str: ...

    def _repr_html_(self) -> str: ...


class TensorStylable(Protocol):

    def fill_region(
        self,
        start_coord: Union[int, tuple[int, int], tuple[int, int, int]],
        end_coord: Union[int, tuple[int, int], tuple[int, int, int]],
        color: Optional[str],
        opacity: Optional[
            Union[
                float,
                tuple[float, float],
                tuple[float, float, TensorOrder],
            ]
        ],
    ) -> Self: ...

    def fill_color(self, color: str) -> Self: ...

    def fill_opacity(
        self,
        opacity: float,
        end: Optional[float] = None,
        order: Optional[TensorOrder] = None,
    ) -> Self: ...

    def fill_values(self) -> Self: ...


class TensorAnnotatable(Protocol):

    def annotate_dim_indices(
        self,
        dim: Literal["row", "col", "depth", "all"] = "all",
        color: Optional[str] = None,
    ) -> Self: ...

    def annotate_dim_size(
        self,
        dim: Literal["row", "col", "depth", "all"] = "all",
        color: Optional[str] = None,
    ) -> Self: ...


class TensorDiagram(Renderable, TensorAnnotatable, TensorStylable):

    _row_indices_annotation: Optional[TensorAnnotation]
    _col_indices_annotation: Optional[TensorAnnotation]
    _depth_indices_annotation: Optional[TensorAnnotation]

    _row_size_annotation: Optional[TensorAnnotation]
    _col_size_annotation: Optional[TensorAnnotation]
    _depth_size_annotation: Optional[TensorAnnotation]

    _style: TensorStyle

    @property
    def rank(self) -> int: ...

    @property
    def tensor_shape(self) -> tuple[int, ...]: ...

    def to_chalk_diagram(self) -> chalk.Diagram: ...

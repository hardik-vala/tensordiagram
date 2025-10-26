from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Callable, ClassVar, Literal, Protocol, Optional, Union, TYPE_CHECKING
from typing_extensions import Self

import chalk

if TYPE_CHECKING:
    from dataclasses import Field

Scalar = Union[int, float, str, bool]
TensorLike = Any
ColorFunction = Callable[[Union[int, tuple[int, ...]], Scalar], str]
OpacityFunction = Callable[[Union[int, tuple[int, ...]], Scalar], float]


_m = lambda a, b: a if a is not None else b


class TensorOrder(Enum):
    """Specifies the order of tensor dimensions.

    This is used to control the layout of gradients and other effects.
    """

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
    """Styling options for a tensor diagram.

    Attributes:
        cell_size: The size of each cell in the tensor diagram.
        color_map: A tensor-like object that maps colors to the cells.
        opacity_map: A tensor-like object that maps opacities to the cells.
        show_values: Whether to show the values in the cells.
    """

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
    ) -> None:
        """Renders the object to a PNG file.

        Args:
            path: The path to save the PNG file to.
            height: The height of the PNG file.
            width: The width of the PNG file.
        """
        ...

    def render_svg(
        self, path: str, height: Optional[int] = None, width: Optional[int] = None
    ) -> None:
        """Renders the object to an SVG file.

        Args:
            path: The path to save the SVG file to.
            height: The height of the SVG file.
            width: The width of the SVG file.
        """
        ...

    def render_pdf(self, path: str, height: Optional[int] = None) -> None:
        """Renders the object to a PDF file.

        Args:
            path: The path to save the PDF file to.
            height: The height of the PDF file.
        """
        ...

    def _repr_svg_(self) -> str:
        """Returns an SVG representation of the object for display in notebooks."""
        ...

    def _repr_html_(self) -> str:
        """Returns an HTML representation of the object for display in notebooks."""
        ...


class TensorStylable(Protocol):

    def fill_region(
        self,
        start_coord: Union[int, tuple[int, int], tuple[int, int, int]],
        end_coord: Union[int, tuple[int, int], tuple[int, int, int]],
        color: Optional[Union[str, ColorFunction]],
        opacity: Optional[
            Union[
                float,
                tuple[float, float],
                tuple[float, float, TensorOrder],
                OpacityFunction,
            ]
        ],
    ) -> Self:
        """Fills a region of the tensor with a color and/or opacity.

        Args:
            start_coord: The starting coordinate of the region to fill (inclusive).
            end_coord: The ending coordinate of the region to fill (inclusive).
            color: The color to fill the region with. Can be a string color
                or a callable that takes (index, value) and returns a color string.
            opacity: The opacity to fill the region with. Can be a single
                value, a tuple of (start, end) opacities for a gradient,
                a tuple of (start, end, order) for a gradient with a
                specific order, or a callable that takes (index, value) and
                returns an opacity float. Note: gradients and functions are
                mutually exclusive.

        Examples:
            Static color and opacity:
            >>> d = td.to_diagram((2, 3, 4))
            >>> d.fill_region((0, 0, 0), (0, 1, 2), color="red", opacity=0.5)

            Gradient opacity:
            >>> d = td.to_diagram((2, 3, 4))
            >>> d.fill_region((0, 0, 0), (0, 1, 2), color="red", opacity=(0.1, 0.9))

            Function-based color (based on cell value):
            >>> tensor = td.to_diagram(np.array([[1, -2], [3, -4]]))
            >>> tensor.fill_region((0, 0), (1, 1),
            ...     color=lambda idx, val: "red" if val > 0 else "blue",
            ...     opacity=None)

            Function-based opacity (based on index position):
            >>> d = td.to_diagram((3, 4))
            >>> d.fill_region((0, 0), (2, 3),
            ...     color="green",
            ...     opacity=lambda idx, val: idx[0] / 2.0)
        """
        ...

    def fill_color(self, color: Union[str, ColorFunction]) -> Self:
        """Fills the entire tensor with a color.

        Args:
            color: The color to fill the tensor with. Can be a string color
                or a callable that takes (index, value) and returns a color string.

        Examples:
            Static color:
            >>> d = td.to_diagram((3, 4))
            >>> d.fill_color("blue")

            Function-based color:
            >>> tensor = td.to_diagram(np.array([[1, 2], [3, 4]]))
            >>> tensor.fill_color(lambda idx, val: "red" if val > 2 else "blue")
        """
        ...

    def fill_opacity(
        self,
        opacity: Union[float, OpacityFunction],
        end: Optional[float] = None,
        order: Optional[TensorOrder] = None,
    ) -> Self:
        """Fills the entire tensor with an opacity or opacity gradient.

        Args:
            opacity: The starting opacity or a callable that takes (index, value)
                and returns an opacity float. When using a function, end and
                order parameters must be None (functions and gradients are
                mutually exclusive).
            end: The ending opacity for a gradient.
            order: The order of the gradient.

        Examples:
            Static opacity:
            >>> d = td.to_diagram((3, 4))
            >>> d.fill_opacity(0.5)

            Gradient opacity:
            >>> d = td.to_diagram((3, 4))
            >>> d.fill_opacity(0.2, 0.8)

            Function-based opacity:
            >>> tensor = td.to_diagram(np.array([[1, 2, 3], [4, 5, 6]]))
            >>> tensor.fill_opacity(lambda idx, val: val / 10.0)
        """
        ...

    def fill_values(self) -> Self:
        """Fills the tensor cells with their values as text."""
        ...


class TensorAnnotatable(Protocol):

    def annotate_dim_indices(
        self,
        dim: Literal["row", "col", "depth", "all"] = "all",
        color: Optional[str] = None,
    ) -> Self:
        """Annotates dimensions of the tensor with indices.

        Args:
            dim: The dimension to annotate.
            color: The color of the index text.

        Examples:
            >>> d = td.to_diagram((3, 4))
            >>> d.annotate_dim_indices("row", color="green")
            >>>
            >>> d = td.to_diagram((3, 4))
            >>> d.annotate_dim_indices("all")
        """
        ...

    def annotate_dim_size(
        self,
        dim: Literal["row", "col", "depth", "all"] = "all",
        color: Optional[str] = None,
    ) -> Self:
        """Annotates dimensions of the tensor with their size.

        Args:
            dim: The dimension to annotate.
            color: The color of the size annotations.

        Examples:
            >>> d = TensorDiagram(np.arange(12).reshape(3, 4))
            >>> d.annotate_dim_size("col", color="purple")
            >>> d.annotate_dim_size("all")
        """
        ...


class TensorDiagram(Renderable, TensorAnnotatable, TensorStylable):
    """A diagram object wrapping a tensor.

    This class provides the interface for creating, styling, annotating, and
    rendering tensor diagrams.
    """

    _row_indices_annotation: Optional[TensorAnnotation]
    _col_indices_annotation: Optional[TensorAnnotation]
    _depth_indices_annotation: Optional[TensorAnnotation]

    _row_size_annotation: Optional[TensorAnnotation]
    _col_size_annotation: Optional[TensorAnnotation]
    _depth_size_annotation: Optional[TensorAnnotation]

    _style: TensorStyle

    @property
    def rank(self) -> int:
        """The rank of the underlying tensor."""
        ...

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        """The shape of the underlying tensor."""
        ...

    def to_chalk_diagram(self) -> chalk.Diagram:
        """Converts the tensor diagram to a chalk diagram."""
        ...

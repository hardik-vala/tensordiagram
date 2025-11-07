from __future__ import annotations

from colour import Color
from dataclasses import dataclass
import functools
from typing import Any, Callable, List, Literal, Optional, Tuple, Union
import sys

import chalk
import numpy as np

from tensordiagram.types import (
    ColorFunction,
    FontSize,
    OpacityFunction,
    Scalar,
    TensorAnnotation,
    TensorDiagram,
    TensorLike,
    TensorOrder,
    TensorStyle,
)
from .utils import convert_tensor, draw_cell, draw_cube

DEFAULT_HEIGHT = 128


def set_default_height(height: int) -> None:
    "Globally set the default height for rendering diagrams."
    global DEFAULT_HEIGHT
    DEFAULT_HEIGHT = height
    chalk.set_svg_height(height)


_default_map = lambda shape, value: np.full(
    shape, value, dtype=object if isinstance(value, str) else None
)


def _get_font_size(
    cell_size: float,
    tensor: WrappedTensor,
    format_fn: Optional[Callable[[Scalar], str]],
) -> float:
    values = tensor.flatten()
    if format_fn is None:
        format_fn = lambda x: f"{x:.2f}" if isinstance(x, float) else str(x)
    value_strs = [format_fn(v) for v in values]
    max_len = max(len(s) for s in value_strs)
    font_size_multiplier = 1.5 / max_len if max_len > 2 else 0.6
    return cell_size * font_size_multiplier


def _build_diagram_1d(
    tensor: WrappedTensor,
    cell_size: float,
    color_map: TensorLike,
    opacity_map: TensorLike,
    show_values: bool,
    font_size: Optional[float],
    format_fn: Optional[Callable[[Scalar], str]],
) -> chalk.Diagram:
    rows = tensor.shape[0]
    return chalk.vcat(
        draw_cell(
            cell_size=cell_size,
            color=Color(color_map[c]),
            opacity=opacity_map[c],
            value=tensor[c] if show_values else None,
            font_size=font_size,
            format_fn=format_fn,
        )
        for c in range(rows)
    )


def _build_diagram_2d(
    tensor: WrappedTensor,
    cell_size: float,
    color_map: TensorLike,
    opacity_map: TensorLike,
    show_values: bool,
    font_size: Optional[float],
    format_fn: Optional[Callable[[Scalar], str]],
) -> chalk.Diagram:
    rows, cols = tensor.shape
    return chalk.vcat(
        chalk.hcat(
            draw_cell(
                cell_size=cell_size,
                color=Color(color_map[r, c]),
                opacity=opacity_map[r, c],
                value=tensor[r, c] if show_values else None,
                font_size=font_size,
                format_fn=format_fn,
            )
            for c in range(cols)
        )
        for r in range(rows)
    )


def _build_diagram_3d(
    tensor: WrappedTensor,
    cell_size: float,
    color_map: TensorLike,
    opacity_map: TensorLike,
    show_values: bool,
    font_size: Optional[float],
    format_fn: Optional[Callable[[Scalar], str]],
) -> chalk.Diagram:
    rows, cols, depth = tensor.shape
    hyp = (chalk.unit_y * 0.5 * cell_size).shear_x(-1)  # type: ignore

    layers = []
    for d in reversed(range(depth)):
        front = chalk.cat(
            [
                chalk.hcat(
                    draw_cube(
                        cell_size=cell_size,
                        color=Color(color_map[r, c, d]),
                        opacity=opacity_map[r, c, d],
                        value=tensor[r, c, d] if show_values else None,
                        font_size=font_size,
                        format_fn=format_fn,
                    )
                    for c in range(cols)
                )
                for r in reversed(range(rows))
            ],
            -chalk.unit_y,  # type: ignore
        ).align_t()
        layers.append(front.translate(-d * hyp.x, -d * hyp.y))

    return chalk.concat(layers)


def _build_row_size_annotation(
    rows: int, cell_size: float, color: str
) -> Tuple[chalk.Diagram, float]:
    line = chalk.vrule(cell_size * rows).line_color(Color(color))
    endpoint_top = chalk.hrule(cell_size * 0.25).line_color(Color(color))
    endpoint_bottom = chalk.hrule(cell_size * 0.25).line_color(Color(color))
    line = endpoint_top / line / endpoint_bottom
    label = (
        chalk.text(str(rows), size=cell_size * 0.5)
        .fill_color(Color(color))
        .line_color(Color(color))
        .line_width(0.0)
    )
    spacing = chalk.hstrut(cell_size * 0.5)
    return (label.center_xy() | spacing | line.center_xy(), cell_size * 1.25)


def _build_col_size_annotation(
    cols: int, cell_size: float, color: str
) -> Tuple[chalk.Diagram, float]:
    line = chalk.hrule(cell_size * cols).line_color(Color(color))
    endpoint_left = chalk.vrule(cell_size * 0.25).line_color(Color(color))
    endpoint_right = chalk.vrule(cell_size * 0.25).line_color(Color(color))
    line = endpoint_left | line | endpoint_right

    label = (
        chalk.text(str(cols), size=cell_size * 0.5)
        .fill_color(Color(color))
        .line_color(Color(color))
        .line_width(0.0)
    )
    spacing = chalk.vstrut(cell_size * 0.5)
    return (line.center_xy() / spacing / label.center_xy(), cell_size * 1.25)


def _build_depth_size_annotation(
    depth: int, cell_size: float, color: str
) -> Tuple[chalk.Diagram, float]:
    line = chalk.hrule(0.5 * cell_size * depth).shear_y(-1).line_color(Color(color))
    endpoint_left = (
        chalk.hrule(cell_size * 0.25)
        .line_color(Color(color))
        .align_bl()
        .translate(
            dx=-0.25 * cell_size * depth - 0.125 * cell_size,
            dy=0.25 * cell_size * depth,
        )
    )
    endpoint_right = (
        chalk.hrule(cell_size * 0.25)
        .line_color(Color(color))
        .align_tr()
        .translate(
            dx=0.25 * cell_size * depth + 0.125 * cell_size,
            dy=-0.25 * cell_size * depth,
        )
    )
    line = line + endpoint_left + endpoint_right

    label = (
        chalk.text(str(depth), size=cell_size * 0.5)
        .fill_color(Color(color))
        .line_color(Color(color))
        .line_width(0.0)
    )
    return (line + label.center_xy().translate(dx=cell_size * 0.5, dy=0), 0.0)


@dataclass
class WrappedTensor:
    _tensor: np.ndarray

    def __init__(self, tensor: TensorLike) -> None:
        super().__init__()
        _tensor = convert_tensor(tensor)
        if len(_tensor.shape) == 0:
            raise ValueError(
                f"Tensor must have at least 1 dimension, got shape {_tensor.shape}"
            )
        if len(_tensor.shape) > 3:
            raise ValueError(
                f"Tensor must be at most 3-dimensional, got shape {_tensor.shape} "
                f"with {len(_tensor.shape)} dimensions"
            )

        self._tensor = _tensor

    def __getitem__(self, key: Any) -> Any:
        return self._tensor[key]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._tensor.shape

    def flatten(self) -> List[Scalar]:
        return self._tensor.flatten().tolist()


def to_diagram(shape: Union[tuple[int, ...], TensorLike]) -> TensorDiagram:
    """Convert a shape tuple or tensor-like object to a TensorDiagram.

    Args:
        shape: Either a tuple of integers representing tensor dimensions (e.g., (3, 4) for a 3x4 matrix)
               or a tensor-like object (numpy array, torch tensor, jax array, tensorflow tensor, mlx array, or list).

    Returns:
        TensorDiagram: A diagram object that can be styled, annotated, combined, and exported to various formats.

    Raises:
        ValueError: If the shape tuple is empty or the tensor has more than 3 dimensions.
        TypeError: If the input is neither a shape tuple nor a tensor-like object.

    Examples:
        >>> import tensordiagram as td
        >>> import numpy as np

        >>> # Create from shape tuple
        >>> diagram = td.to_diagram((3, 4))

        >>> # Create from numpy array
        >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
        >>> diagram = td.to_diagram(arr)

        >>> # Create from torch tensor
        >>> import torch
        >>> tensor = torch.randn(2, 3, 4)
        >>> diagram = td.to_diagram(tensor)

    Note:
        - Supports tensors with 1, 2, or 3 dimensions only.
        - When passing a shape tuple, a placeholder tensor filled with zeros is created internally.
        - Tensor-like objects from torch, jax, tensorflow, and mlx are automatically converted to numpy arrays.
    """
    if isinstance(shape, tuple):
        if len(shape) == 0:
            raise ValueError(f"Shape tuple must have at least 1 element, got {shape}")
        if len(shape) > 3:
            raise ValueError(
                f"Tensor must be at most 3-dimensional, got shape {shape} "
                f"with {len(shape)} dimensions"
            )
        wt = WrappedTensor(np.full(shape, 0))  # Placeholder tensor with zeros
    else:
        wt = WrappedTensor(shape)
    return TensorDiagramImpl(
        _wrapped_tensor=wt,
        _style=TensorStyle(),
    )


@dataclass
class TensorDiagramImpl(TensorDiagram):

    _wrapped_tensor: WrappedTensor

    _style: TensorStyle

    _row_indices_annotation: Optional[TensorAnnotation] = None
    _col_indices_annotation: Optional[TensorAnnotation] = None
    _depth_indices_annotation: Optional[TensorAnnotation] = None
    _row_size_annotation: Optional[TensorAnnotation] = None
    _col_size_annotation: Optional[TensorAnnotation] = None
    _depth_size_annotation: Optional[TensorAnnotation] = None

    @property
    def tensor_shape(self) -> tuple[int, ...]:
        return self._wrapped_tensor.shape

    @property
    def rank(self) -> int:
        return len(self.tensor_shape)

    @property
    def tensor_size(self) -> int:
        return functools.reduce(lambda x, y: x * y, self.tensor_shape, 1)

    @property
    def _cell_size(self) -> float:
        return self._style.cell_size if self._style.cell_size else 1.0

    @property
    def _color_map(self) -> TensorLike:
        if self._style.color_map is None:
            return _default_map(self.tensor_shape, "white")
        return self._style.color_map

    @property
    def _opacity_map(self) -> TensorLike:
        if self._style.opacity_map is None:
            return _default_map(self.tensor_shape, 1.0)
        return self._style.opacity_map

    @property
    def _show_values(self) -> bool:
        return self._style.show_values if self._style.show_values is not None else False

    @property
    def _diagram(self) -> chalk.Diagram:
        shape = self.tensor_shape
        cell_size = self._cell_size
        color_map = self._color_map
        opacity_map = self._opacity_map
        show_values = self._show_values

        assert len(shape) > 0 and len(shape) <= 3

        tensor = self._wrapped_tensor

        format_fn = self._style.value_format_fn
        in_font_size = self._style.value_font_size
        if show_values:
            if in_font_size is not None:
                font_size = (
                    in_font_size
                    if isinstance(in_font_size, float)
                    else float(in_font_size)
                )
            else:
                font_size = _get_font_size(cell_size, tensor, format_fn)
        else:
            font_size = None

        if len(tensor.shape) == 1:
            # 1D tensor
            build_diagram_f = _build_diagram_1d
        elif len(tensor.shape) == 2:
            # 2D tensor
            build_diagram_f = _build_diagram_2d
        else:
            # 3D tensor
            build_diagram_f = _build_diagram_3d
        out_diagram = build_diagram_f(
            tensor, cell_size, color_map, opacity_map, show_values, font_size, format_fn
        )

        offset_l = 0.0
        offset_b = 0.0
        if self._row_indices_annotation:
            rows = self.tensor_shape[0]
            color = (
                self._row_indices_annotation.color
                if self._row_indices_annotation.color
                else "black"
            )
            size = (
                self._row_indices_annotation.font_size
                if self._row_indices_annotation.font_size is not None
                else cell_size * 0.5
            )

            labels = []
            for r in range(rows):
                label = (
                    chalk.text(str(r), size=size)
                    .fill_color(Color(color))
                    .line_color(Color(color))
                    .line_width(0.0)
                )
                labels.append(label.center_xy())

            hspacing = chalk.hstrut(cell_size * 0.5)
            vspacing = chalk.vstrut(cell_size * 0.5)
            out_diagram = (
                (chalk.vcat(labels, cell_size) / vspacing).align_b()
                | hspacing
                | out_diagram.align_b()
            )
            offset_l += cell_size * 0.5
        if self._row_size_annotation:
            rows = self.tensor_shape[0]
            color = (
                self._row_size_annotation.color
                if self._row_size_annotation.color
                else "black"
            )
            annotation_diagram, offset = _build_row_size_annotation(
                rows, cell_size, color
            )
            spacing = chalk.hstrut(cell_size * 0.5)

            out_diagram = annotation_diagram.align_b() | spacing | out_diagram.align_b()

            offset_l += offset
        if self._col_indices_annotation:
            cols = self.tensor_shape[1]
            color = (
                self._col_indices_annotation.color
                if self._col_indices_annotation.color
                else "black"
            )
            size = (
                self._col_indices_annotation.font_size
                if self._col_indices_annotation.font_size is not None
                else cell_size * 0.5
            )

            labels = []
            for c in range(cols):
                label = (
                    chalk.text(str(c), size=size)
                    .fill_color(Color(color))
                    .line_color(Color(color))
                    .line_width(0.0)
                )
                labels.append(label.center_xy())

            hspacing = chalk.hstrut(cell_size * 0.5 + offset_l)
            vspacing = chalk.vstrut(cell_size * 0.5)
            out_diagram = (
                out_diagram.align_l()
                / vspacing
                / (hspacing | chalk.hcat(labels, cell_size)).align_l()
            )
            offset_b += cell_size * 0.5
        if self._col_size_annotation:
            cols = self.tensor_shape[1]
            color = (
                self._col_size_annotation.color
                if self._col_size_annotation.color
                else "black"
            )

            annotation_diagram, offset = _build_col_size_annotation(
                cols, cell_size, color
            )
            spacing = chalk.vstrut(cell_size * 0.5)

            out_diagram = (
                out_diagram.align_l()
                / spacing
                / annotation_diagram.align_l().translate(dx=offset_l, dy=0)
            )

            offset_b += offset
        if self._depth_indices_annotation:
            rows, cols, depth = self.tensor_shape
            color = (
                self._depth_indices_annotation.color
                if self._depth_indices_annotation.color
                else "black"
            )
            size = (
                self._depth_indices_annotation.font_size
                if self._depth_indices_annotation.font_size is not None
                else cell_size * 0.5
            )
            hyp = (chalk.unit_y * 0.5 * cell_size).shear_x(-1)  # type: ignore

            labels = []
            for d in range(depth):
                label = (
                    chalk.text(str(d), size=size)
                    .fill_color(Color(color))
                    .line_color(Color(color))
                    .line_width(0.0)
                )
                labels.append(label.translate(-d * hyp.x, -d * hyp.y))

            out_diagram = out_diagram.align_bl() + chalk.concat(
                labels
            ).align_bl().translate(
                dx=cell_size * (cols + 1.00) + offset_l,
                dy=-0.25 - offset_b,
            )

            offset_l += cell_size
        if self._depth_size_annotation:
            _, cols, depth = self.tensor_shape
            color = (
                self._depth_size_annotation.color
                if self._depth_size_annotation.color
                else "black"
            )

            annotation_diagram, _ = _build_depth_size_annotation(
                depth, cell_size, color
            )

            out_diagram = (
                out_diagram.align_bl()
                + annotation_diagram.align_bl().translate(
                    dx=cell_size * (cols + 0.5) + offset_l, dy=-offset_b
                )
            )

        return out_diagram

    def render(
        self, path: str, height: Optional[int] = None, width: Optional[int] = None
    ) -> None:
        return self._diagram.render(path, height=height, width=width)

    def render_png(
        self, path: str, height: Optional[int] = None, width: Optional[int] = None
    ) -> None:
        try:
            import cairo  # type: ignore[import-error]
        except ImportError:
            raise ImportError(
                "pycairo is required to render png diagrams. Please install pycairo with `pip install tensordiagram[cairo]`."
            ) from None

        height = height if height is not None else DEFAULT_HEIGHT
        return self._diagram.render_png(path, height=height, width=width)

    def render_svg(
        self, path: str, height: Optional[int] = None, width: Optional[int] = None
    ) -> None:
        try:
            import cairosvg  # type: ignore[import-error]
        except ImportError:
            raise ImportError(
                "cairosvg is required to render svg diagrams. Please install cairosvg with `pip install tensordiagram[svg]`."
            ) from None
        height = height if height is not None else DEFAULT_HEIGHT
        return self._diagram.render_svg(path, height=height, width=width)

    def render_pdf(self, path: str, height: Optional[int] = None) -> None:
        try:
            import pylatex  # type: ignore[import-error]
        except ImportError:
            raise ImportError(
                "pylatex is required to render pdf diagrams. Please install pylatex and latextools with `pip install tensordiagram[tikz]`."
            ) from None
        height = height if height is not None else DEFAULT_HEIGHT
        return self._diagram.render_pdf(path, height=height)

    def _repr_svg_(self) -> str:
        return self._diagram._repr_svg_()

    def _repr_html_(self) -> str:
        return self._diagram._repr_html_()

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
    ) -> TensorDiagram:
        if color is None and opacity is None:
            raise ValueError("At least one of color or opacity must be provided")

        if self.rank == 1:
            if isinstance(start_coord, int) and isinstance(end_coord, int):
                x_start = start_coord
                x_end = end_coord
            else:
                raise ValueError(
                    "For 1D tensors, start and end coordinates must be integers"
                )
            x = slice(x_start, x_end + 1)
            y = None
            z = None

        if self.rank == 2:
            if (
                isinstance(start_coord, tuple)
                and len(start_coord) == 2
                and isinstance(end_coord, tuple)
                and len(end_coord) == 2
            ):
                x_start, y_start = start_coord
                x_end, y_end = end_coord
            else:
                raise ValueError(
                    "For 2D tensors, start and end coordinates must be tuples of (row, column)"
                )
            x = slice(x_start, x_end + 1)
            y = slice(y_start, y_end + 1)
            z = None

        if self.rank == 3:
            if (
                isinstance(start_coord, tuple)
                and len(start_coord) == 3
                and isinstance(end_coord, tuple)
                and len(end_coord) == 3
            ):
                x_start, y_start, z_start = start_coord
                x_end, y_end, z_end = end_coord
            else:
                raise ValueError(
                    "For 3D tensors, start and end coordinates must be tuples of (row, column, depth)"
                )
            x = slice(x_start, x_end + 1)
            y = slice(y_start, y_end + 1)
            z = slice(z_start, z_end + 1)

        color_map = self._color_map.copy()
        if color is not None:
            if callable(color):
                # color is a function: call it for each cell in the region
                if self.rank == 1:
                    for i in range(x.start, x.stop):
                        idx = i
                        val = self._wrapped_tensor[i]
                        color_map[i] = color(idx, val)
                elif self.rank == 2:
                    for i in range(x.start, x.stop):
                        for j in range(y.start, y.stop):  # type: ignore
                            idx = (i, j)
                            val = self._wrapped_tensor[i, j]
                            color_map[i, j] = color(idx, val)
                elif self.rank == 3:
                    for i in range(x.start, x.stop):
                        for j in range(y.start, y.stop):  # type: ignore
                            for k in range(z.start, z.stop):  # type: ignore
                                idx = (i, j, k)
                                val = self._wrapped_tensor[i, j, k]
                                color_map[i, j, k] = color(idx, val)
            else:
                # color is a static string
                if self.rank == 1:
                    color_map[x] = color
                elif self.rank == 2:
                    color_map[x, y] = color
                elif self.rank == 3:
                    color_map[x, y, z] = color

        opacity_map = self._opacity_map.copy()
        if opacity is not None:
            if callable(opacity):
                # opacity is a function: call it for each cell in the region
                if self.rank == 1:
                    for i in range(x.start, x.stop):
                        idx = i
                        val = self._wrapped_tensor[i]
                        opacity_map[i] = opacity(idx, val)
                elif self.rank == 2:
                    for i in range(x.start, x.stop):
                        for j in range(y.start, y.stop):  # type: ignore
                            idx = (i, j)
                            val = self._wrapped_tensor[i, j]
                            opacity_map[i, j] = opacity(idx, val)
                elif self.rank == 3:
                    for i in range(x.start, x.stop):
                        for j in range(y.start, y.stop):  # type: ignore
                            for k in range(z.start, z.stop):  # type: ignore
                                idx = (i, j, k)
                                val = self._wrapped_tensor[i, j, k]
                                opacity_map[i, j, k] = opacity(idx, val)
            elif isinstance(opacity, float):
                if self.rank == 1:
                    opacity_map[x] = opacity
                elif self.rank == 2:
                    opacity_map[x, y] = opacity
                elif self.rank == 3:
                    opacity_map[x, y, z] = opacity
            elif isinstance(opacity, tuple):
                if len(opacity) == 2:
                    start_o, end_o = opacity
                    order = None
                elif len(opacity) == 3:
                    start_o, end_o, order = opacity
                else:
                    raise ValueError(
                        "Opacity tuple must have the form (start, end) or (start, end, order)"
                    )

                if self.rank == 1:
                    if order is None:
                        order = TensorOrder.R

                    if order != TensorOrder.R:
                        raise ValueError(
                            "For 1D tensors, order for opacity must be 'r'"
                        )
                    size = x.stop - x.start
                    for i in range(size):
                        interp_opacity = start_o + (end_o - start_o) * (i / (size - 1))
                        opacity_map[x.start + i] = interp_opacity
                elif self.rank == 2:
                    if order is None:
                        order = TensorOrder.RC

                    if order not in (
                        TensorOrder.R,
                        TensorOrder.C,
                        TensorOrder.RC,
                        TensorOrder.CR,
                    ):
                        raise ValueError(
                            "For 2D tensors, order for opacity must be 'R', 'C', 'RC', or 'CR'"
                        )

                    y_start = y.start  # type: ignore
                    x_size = x.stop - x.start
                    y_size = y.stop - y_start  # type: ignore
                    size = y_size * x_size

                    indices = [(i, j) for j in range(y_size) for i in range(x_size)]

                    for i, j in indices:
                        if order == TensorOrder.R:
                            mul = i / (x_size - 1)
                        elif order == TensorOrder.C:
                            mul = j / (y_size - 1)
                        elif order == TensorOrder.RC:
                            mul = (i + j * x_size) / (x_size * y_size - 1)
                        elif order == TensorOrder.CR:
                            mul = (j + i * y_size) / (y_size * x_size - 1)
                        else:
                            raise ValueError("Invalid order")
                        interp_opacity = start_o + (end_o - start_o) * mul
                        opacity_map[x.start + i, y_start + j] = interp_opacity
                elif self.rank == 3:
                    y_start = y.start  # type: ignore
                    z_start = z.start  # type: ignore
                    x_size = x.stop - x.start
                    y_size = y.stop - y_start  # type: ignore
                    z_size = z.stop - z_start  # type: ignore
                    size = z_size * y_size * x_size

                    indices = [
                        (i, j, k)
                        for k in range(z_size)
                        for j in range(y_size)
                        for i in range(x_size)
                    ]

                    for i, j, k in indices:
                        if order is None:
                            order = TensorOrder.RCD

                        if order == TensorOrder.R:
                            mul = i / (x_size - 1)
                        elif order == TensorOrder.C:
                            mul = j / (y_size - 1)
                        elif order == TensorOrder.D:
                            mul = k / (z_size - 1)
                        elif order == TensorOrder.RC:
                            mul = (i + j * x_size) / (x_size * y_size - 1)
                        elif order == TensorOrder.RD:
                            mul = (i + k * x_size) / (x_size * z_size - 1)
                        elif order == TensorOrder.CR:
                            mul = (j + i * y_size) / (y_size * x_size - 1)
                        elif order == TensorOrder.CD:
                            mul = (j + k * y_size) / (y_size * z_size - 1)
                        elif order == TensorOrder.DR:
                            mul = (k + i * z_size) / (z_size * x_size - 1)
                        elif order == TensorOrder.DC:
                            mul = (k + j * z_size) / (z_size * y_size - 1)
                        elif order == TensorOrder.RCD:
                            mul = (i + j * x_size + k * x_size * y_size) / (size - 1)
                        elif order == TensorOrder.RDC:
                            mul = (i + k * x_size + j * x_size * z_size) / (size - 1)
                        elif order == TensorOrder.CRD:
                            mul = (j + i * y_size + k * y_size * x_size) / (size - 1)
                        elif order == TensorOrder.CDR:
                            mul = (j + k * y_size + i * y_size * z_size) / (size - 1)
                        elif order == TensorOrder.DRC:
                            mul = (k + i * z_size + j * z_size * x_size) / (size - 1)
                        elif order == TensorOrder.DCR:
                            mul = (k + j * z_size + i * z_size * y_size) / (size - 1)
                        else:
                            raise ValueError("Invalid direction")

                        interp_opacity = start_o + (end_o - start_o) * mul
                        opacity_map[x.start + i, y_start + j, z_start + k] = (
                            interp_opacity
                        )

        style = self._style.transfer(
            TensorStyle(color_map=color_map, opacity_map=opacity_map)
        )
        return TensorDiagramImpl(
            _wrapped_tensor=self._wrapped_tensor,
            _row_indices_annotation=self._row_indices_annotation,
            _col_indices_annotation=self._col_indices_annotation,
            _depth_indices_annotation=self._depth_indices_annotation,
            _row_size_annotation=self._row_size_annotation,
            _col_size_annotation=self._col_size_annotation,
            _depth_size_annotation=self._depth_size_annotation,
            _style=style,
        )

    def fill_color(self, color: Union[str, ColorFunction]) -> TensorDiagram:
        if self.rank == 1:
            start_coord = 0
            end_coord = self.tensor_shape[0] - 1
        elif self.rank == 2:
            start_coord = (0, 0)
            end_coord = (self.tensor_shape[0] - 1, self.tensor_shape[1] - 1)
        elif self.rank == 3:
            start_coord = (0, 0, 0)
            end_coord = (
                self.tensor_shape[0] - 1,
                self.tensor_shape[1] - 1,
                self.tensor_shape[2] - 1,
            )

        return self.fill_region(
            start_coord=start_coord,
            end_coord=end_coord,
            color=color,
            opacity=None,
        )

    def fill_opacity(
        self,
        opacity: Union[float, OpacityFunction],
        end: Optional[float] = None,
        order: Optional[TensorOrder] = None,
    ) -> TensorDiagram:
        # Check for mutually exclusive function and gradient parameters
        if callable(opacity) and (end is not None or order is not None):
            raise ValueError(
                "When using a function for opacity, 'end' and 'order' parameters must be None. "
                "Functions and gradients are mutually exclusive."
            )

        if self.rank == 1:
            start_coord = 0
            end_coord = self.tensor_shape[0] - 1
        elif self.rank == 2:
            start_coord = (0, 0)
            end_coord = (self.tensor_shape[0] - 1, self.tensor_shape[1] - 1)
        elif self.rank == 3:
            start_coord = (0, 0, 0)
            end_coord = (
                self.tensor_shape[0] - 1,
                self.tensor_shape[1] - 1,
                self.tensor_shape[2] - 1,
            )

        if callable(opacity):
            opacity_arg = opacity
        elif end is not None and order is not None:
            opacity_arg = (opacity, end, order)
        elif end is not None and order is None:
            opacity_arg = (opacity, end)
        else:
            opacity_arg = opacity

        return self.fill_region(
            start_coord=start_coord,
            end_coord=end_coord,
            color=None,
            opacity=opacity_arg,
        )

    def fill_values(
        self,
        font_size: Optional[FontSize] = None,
        format_fn: Optional[Callable[[Scalar], str]] = None,
    ) -> TensorDiagram:
        if self.rank == 3:
            raise ValueError("Showing values for 3D tensors is not supported")
        style = self._style.transfer(
            TensorStyle(
                show_values=True,
                value_font_size=font_size,
                value_format_fn=format_fn,
            )
        )
        return TensorDiagramImpl(
            _wrapped_tensor=self._wrapped_tensor,
            _row_indices_annotation=self._row_indices_annotation,
            _col_indices_annotation=self._col_indices_annotation,
            _depth_indices_annotation=self._depth_indices_annotation,
            _row_size_annotation=self._row_size_annotation,
            _col_size_annotation=self._col_size_annotation,
            _depth_size_annotation=self._depth_size_annotation,
            _style=style,
        )

    def annotate_dim_indices(
        self,
        dim: Literal["row", "col", "depth", "all"] = "all",
        color: Optional[str] = None,
        font_size: Optional[FontSize] = None,
    ) -> TensorDiagram:
        if self.rank == 1 and dim != "row" and dim != "all":
            raise ValueError("1D tensors can only annotate 'row' dimension")
        if self.rank == 2 and dim == "depth":
            raise ValueError("2D tensors cannot annotate 'depth' dimension")
        if self.rank == 3 and dim not in ["row", "col", "depth", "all"]:
            raise ValueError(
                "3D tensors can only annotate 'row', 'col', or 'depth' dimension"
            )

        row_annotation = self._row_indices_annotation
        col_annotation = self._col_indices_annotation
        depth_annotation = self._depth_indices_annotation
        if dim == "row" or dim == "all":
            if self._row_indices_annotation is None:
                row_annotation = TensorAnnotation(color, font_size)
            else:
                row_annotation = self._row_indices_annotation.transfer(
                    TensorAnnotation(color, font_size)
                )
        if (dim == "col" or dim == "all") and self.rank >= 2:
            if self._col_indices_annotation is None:
                col_annotation = TensorAnnotation(color, font_size)
            else:
                col_annotation = self._col_indices_annotation.transfer(
                    TensorAnnotation(color, font_size)
                )
        if (dim == "depth" or dim == "all") and self.rank == 3:
            if self._depth_indices_annotation is None:
                depth_annotation = TensorAnnotation(color, font_size)
            else:
                depth_annotation = self._depth_indices_annotation.transfer(
                    TensorAnnotation(color, font_size)
                )

        return TensorDiagramImpl(
            _wrapped_tensor=self._wrapped_tensor,
            _row_indices_annotation=row_annotation,
            _col_indices_annotation=col_annotation,
            _depth_indices_annotation=depth_annotation,
            _row_size_annotation=self._row_size_annotation,
            _col_size_annotation=self._col_size_annotation,
            _depth_size_annotation=self._depth_size_annotation,
            _style=self._style,
        )

    def annotate_dim_size(
        self,
        dim: Literal["row", "col", "depth", "all"] = "all",
        color: Optional[str] = None,
    ) -> TensorDiagram:
        if self.rank == 1 and dim != "row" and dim != "all":
            raise ValueError("1D tensors can only annotate 'row' dimension")
        if self.rank == 2 and dim == "depth":
            raise ValueError("2D tensors cannot annotate 'depth' dimension")
        if self.rank == 3 and dim not in ["row", "col", "depth", "all"]:
            raise ValueError(
                "3D tensors can only annotate 'row', 'col', or 'depth' dimension"
            )

        row_annotation = self._row_size_annotation
        col_annotation = self._col_size_annotation
        depth_annotation = self._depth_size_annotation
        if dim == "row" or dim == "all":
            if self._row_size_annotation is None:
                row_annotation = TensorAnnotation(color)
            else:
                row_annotation = self._row_size_annotation.transfer(
                    TensorAnnotation(color)
                )
        if (dim == "col" or dim == "all") and self.rank >= 2:
            if self._col_size_annotation is None:
                col_annotation = TensorAnnotation(color)
            else:
                col_annotation = self._col_size_annotation.transfer(
                    TensorAnnotation(color)
                )
        if (dim == "depth" or dim == "all") and self.rank == 3:
            if self._depth_size_annotation is None:
                depth_annotation = TensorAnnotation(color)
            else:
                depth_annotation = self._depth_size_annotation.transfer(
                    TensorAnnotation(color)
                )

        return TensorDiagramImpl(
            _wrapped_tensor=self._wrapped_tensor,
            _row_indices_annotation=self._row_indices_annotation,
            _col_indices_annotation=self._col_indices_annotation,
            _depth_indices_annotation=self._depth_indices_annotation,
            _row_size_annotation=row_annotation,
            _col_size_annotation=col_annotation,
            _depth_size_annotation=depth_annotation,
            _style=self._style,
        )

    def to_chalk_diagram(self) -> chalk.Diagram:
        return self._diagram

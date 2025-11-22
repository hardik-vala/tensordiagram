from colour import Color
import sys
from typing import Optional

import chalk
import numpy as np

from tensordiagram.types import FormatFunction, IndexType, Scalar, TensorLike


def convert_tensor(tensor: TensorLike) -> np.ndarray:
    if "torch" in sys.modules:
        import torch  # type: ignore[import-error]

        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
    if "jax" in sys.modules:
        import jax  # type: ignore[import-error]

        if isinstance(tensor, jax.Array):
            return np.asarray(tensor)
    if "tensorflow" in sys.modules:
        import tensorflow as tf  # type: ignore[import-error]

        if isinstance(tensor, tf.Tensor):
            return tensor.numpy()
    if "mlx.core" in sys.modules:
        import mlx.core as mx  # type: ignore[import-error]

        if isinstance(tensor, mx.array):
            return np.array(tensor)
    # numpy
    if isinstance(tensor, np.ndarray):
        return tensor
    # list
    if isinstance(tensor, list):
        return np.array(tensor)
    else:
        raise TypeError("Unsupported tensor type")


def draw_cell(
    cell_size: float,
    color: Color,
    opacity: float,
    value: Optional[Scalar] = None,
    font_size: Optional[float] = None,
    format_fn: Optional[FormatFunction] = None,
    index: Optional[IndexType] = None,
) -> chalk.Diagram:
    c = chalk.rectangle(cell_size, cell_size).fill_color(color).fill_opacity(opacity)

    if value is not None:
        if format_fn is not None:
            if index is None:
                index = 0  # default index if not provided
            value_str = format_fn(index, value)
        else:
            value_str = f"{value:.2f}" if isinstance(value, float) else str(value)

        if font_size is None:
            value_len = len(value_str)
            font_size_multiplier = 1.5 / value_len if value_len > 1 else 0.8
            font_size = cell_size * font_size_multiplier
        else:
            font_size = font_size

        txt = (
            chalk.text(value_str, font_size).fill_color(Color("black")).line_width(0.0)
        )
        c = c + txt

    return c


def draw_cube(
    cell_size: float,
    color: Color,
    opacity: float,
    # [TODO] Add value support later
    value: Optional[Scalar] = None,
    font_size: Optional[float] = None,
    format_fn: Optional[FormatFunction] = None,
    index: Optional[IndexType] = None,
) -> chalk.Diagram:
    face_f = chalk.rectangle(cell_size, cell_size)
    face_t = chalk.rectangle(cell_size, cell_size * 0.5).shear_x(-1)
    face_r = chalk.rectangle(cell_size * 0.5, cell_size).shear_y(-1)

    face_f = face_f.fill_color(color).fill_opacity(opacity)
    face_t = face_t.fill_color(color).fill_opacity(opacity)
    face_r = face_r.fill_color(color).fill_opacity(opacity)

    cube = (face_t.align_bl() + face_f.align_tl()).align_tr() + face_r.align_tr()
    return cube.align_bl().with_envelope(face_f.align_bl())

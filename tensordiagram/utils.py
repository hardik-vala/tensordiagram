from colour import Color
from typing import Optional

import chalk

from tensordiagram.types import Scalar


def draw_cell(
    cell_size: float,
    color: Color,
    opacity: float,
    value: Optional[Scalar] = None,
    font_size: Optional[float] = None,
) -> chalk.Diagram:
    c = chalk.rectangle(cell_size, cell_size).fill_color(color).fill_opacity(opacity)

    if value is not None:
        value_str = f"{value:.2f}" if isinstance(value, float) else str(value)
        value_len = len(value_str)
        font_size_multiplier = 1.5 / value_len if value_len > 1 else 0.8
        font_size = (
            font_size if font_size is not None else cell_size * font_size_multiplier
        )
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
) -> chalk.Diagram:
    face_f = chalk.rectangle(cell_size, cell_size)
    face_t = chalk.rectangle(cell_size, cell_size * 0.5).shear_x(-1)
    face_r = chalk.rectangle(cell_size * 0.5, cell_size).shear_y(-1)

    face_f = face_f.fill_color(color).fill_opacity(opacity)
    face_t = face_t.fill_color(color).fill_opacity(opacity)
    face_r = face_r.fill_color(color).fill_opacity(opacity)

    cube = (face_t.align_bl() + face_f.align_tl()).align_tr() + face_r.align_tr()
    return cube.align_bl().with_envelope(face_f.align_bl())

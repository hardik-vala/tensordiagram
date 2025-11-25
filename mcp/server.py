#!/usr/bin/env python3
"""
tensordiagram MCP Server

Provides tensor drawing tools for AI assistants via MCP.
Enables LLMs to generate visual representations of tensor shapes and operations.
"""
import argparse
import functools
import json
import os
import tempfile
from typing import Optional, Literal

import chalk
from colour import Color
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
import numpy as np
import tensordiagram as td

# initialize MCP server
mcp = FastMCP(
    name="tensordiagram",
    version="0.1.0",
)

# constants
DEFAULT_HEIGHT = 200
MIN_WIDTH = 400  # minimum width for output images (pixels)
MAX_IMAGE_SIZE = 900_000  # bytes (900KB with safety margin for 1MB MCP limit)


def add_background(diagram, rank, bg_color="white"):
    if rank == 1:
        padding = 0.25
    elif rank == 2:
        padding = 0.5
    else:
        padding = 0.75

    cd = diagram.to_chalk_diagram()
    cd = cd.frame(padding)
    env = cd.get_envelope()
    bgd = (
        chalk.rectangle(env.width, env.height)
        .fill_color(Color(bg_color))
        .line_width(0.0)
    )

    dx, dy = 0.0, 0.0
    if rank == 3:
        dy = 0.25

    return bgd + cd.center_xy().translate(dx=dx, dy=dy)


@mcp.tool()
def draw_tensor(
    shape: tuple[int, ...],
    values: Optional[list[int] | list[float] | str] = None,
    color: Optional[str] = None,
    show_values: bool = False,
    show_dim_indices: bool = False,
    show_dim_sizes: bool = False,
) -> Image:
    """
    Create a visual diagram of a tensor with the given shape.

    This tool generates a diagram showing the structure of a tensor, useful for
    understanding tensor shapes, explaining operations, and debugging dimensional
    mismatches.

    Args:
        shape: Tuple of integers representing tensor dimensions.
               Examples: (3, 4) for 3x4 matrix, (2, 3, 4) for 3D tensor.
               Supports 1D, 2D, and 3D tensors only.

        values: Optional flattened list of values to display in the tensor cells.
                Must match the total size of the shape (product of all dimensions).
                Can be a list ([1, 2, 3, 4, 5, 6]) or a JSON string ('[1, 2, 3, 4, 5, 6]').
                Example: [1, 2, 3, 4, 5, 6] or '[1, 2, 3, 4, 5, 6]' for shape (2, 3).
                Note: Not supported for 3D tensors.

        color: Optional fill color for tensor cells. Can be a color name (e.g., "red", "blue")
               or hex code (e.g., "#FF5733"). If not specified, cells will not be filled
               with a color. Default: None.

        show_values: Whether to display values inside cells. Only supported for 1D
                     and 2D tensors. Default: False.

        show_dim_indices: Whether to show dimension indices (0, 1, 2, etc.) along
                          each dimension. Default: False.

        show_dim_sizes: Whether to show dimension sizes (the actual numbers from
                        the shape tuple). Default: False.

    Returns:
        Image object containing the PNG visualization of the tensor.

    Raises:
        ValueError: If shape is invalid (empty, >3 dimensions, non-positive values).
        ValueError: If values array size doesn't match shape.
        ValueError: If 3D tensor requested with show_values=True (not supported).
        RuntimeError: If generated image exceeds 1MB MCP limit.
        ImportError: If required dependencies (pycairo) are not installed.

    Examples:
        Simple 3x4 matrix visualization:
            draw_tensor(shape=(3, 4))

        3D tensor with custom color:
            draw_tensor(shape=(2, 3, 4), color="coral")

        1D tensor with values displayed:
            draw_tensor(
                shape=(5,),
                values=[1.0, 2.0, 3.0, 4.0, 5.0],
                show_values=True
            )

        2D tensor with annotations:
            draw_tensor(
                shape=(4, 6),
                color="lightgreen",
                show_dim_sizes=True,
                show_dim_indices=True
            )

        Matrix with custom styling:
            draw_tensor(
                shape=(3, 3),
                values=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                color="purple",
                show_values=True,
                show_dim_sizes=True
            )
    """

    height = DEFAULT_HEIGHT

    # ===== input validation =====

    # parse values if provided as string
    if values is not None and isinstance(values, str):
        try:
            values = json.loads(values)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"values string is not valid JSON: {str(e)}\n"
                f"Expected format: '[1, 2, 3, ...]' or '[1.0, 2.0, 3.0, ...]'"
            ) from e

    if not shape or len(shape) == 0:
        raise ValueError("shape must have at least 1 dimension")

    if len(shape) > 3:
        raise ValueError(
            f"only 1d, 2d, and 3d tensors are supported (got {len(shape)} dimensions), "
            f"shape provided: {shape}"
        )

    if any(dim <= 0 for dim in shape):
        raise ValueError(
            f"all dimensions must be positive integers, " f"shape provided: {shape}"
        )

    if values is not None:
        expected_size = functools.reduce(lambda x, y: x * y, shape, 1)
        if len(values) != expected_size:
            raise ValueError(
                f"values length ({len(values)}) doesn't match shape size ({expected_size}), "
                f"shape {shape} requires exactly {expected_size} values"
            )

    if len(shape) == 3 and show_values:
        raise ValueError("show_values is not supported for 3d tensors.")

    if height <= 0:
        raise ValueError(f"height must be positive (got {height})")

    # ===== diagram =====

    rank = len(shape)

    # reshape 1D tensors to be horizontal (1 row, N columns) instead of vertical
    if rank == 1:
        shape = (1, shape[0])
        rank = 2

    try:
        # create base diagram from shape
        if values is not None:
            # convert list to appropriate array structure for tensordiagram
            arr = np.array(values).reshape(shape)
            diagram = td.to_diagram(arr)
        else:
            diagram = td.to_diagram(shape)

        # apply styling
        if color is not None:
            diagram = diagram.fill_color(color)

        # apply value display if requested
        if show_values and values is not None:
            diagram = diagram.fill_values()

        # apply annotations
        if show_dim_indices:
            diagram = diagram.annotate_dim_indices()

        if show_dim_sizes:
            diagram = diagram.annotate_dim_size()

        # add background
        diagram = add_background(diagram, rank, bg_color="white")

        # calculate aspect ratio and adjust height if needed to meet minimum width
        # claude app displays images with a minimum width of ~400 pixels
        env = diagram.get_envelope()
        aspect_ratio = env.width / env.height

        # calculate what the actual pixel width would be with the current height
        # render_png uses height parameter, and width is calculated based on aspect ratio
        calculated_width = height * aspect_ratio

        # if calculated width is less than minimum, adjust height to achieve minimum width
        if calculated_width < MIN_WIDTH:
            # use ceiling to ensure we meet the minimum (avoid rounding down)
            height = int(MIN_WIDTH / aspect_ratio) + 1

        # render to PNG via temporary file
        # (tensordiagram requires file paths for rendering)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # render the diagram
            diagram.render_png(tmp_path, height=height)

            # read the PNG file into bytes
            with open(tmp_path, "rb") as f:
                img_bytes = f.read()

            # validate image size against MCP limit
            if len(img_bytes) > MAX_IMAGE_SIZE:
                raise RuntimeError(
                    f"generated image ({len(img_bytes)} bytes) exceeds MCP limit (~1MB). "
                    f"try reducing the height parameter (current: {height})."
                )

            return Image(data=img_bytes, format="png")

        finally:
            # clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except ImportError as e:
        if "cairo" in str(e).lower():
            raise ImportError(
                "pycairo is required for PNG rendering, "
                "install with: pip install pycairo"
            ) from e
        raise

    except Exception as e:
        # re-raise with additional context
        raise RuntimeError(f"Failed to generate tensor diagram: {str(e)}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="tensordiagram MCP Server")
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "sse"],
        help="transport mode: stdio (default) or sse (HTTP/SSE)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="port for HTTP/SSE transport (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="host for HTTP/SSE transport (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run()

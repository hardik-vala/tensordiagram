#!/usr/bin/env python3
"""
Script to generate reference images for visual regression testing.

Run this script to create baseline reference images that will be used
in the visual regression tests.
"""
import argparse
from pathlib import Path
import sys

# Add parent directory to path to import tensor_draw
sys.path.insert(0, str(Path(__file__).parent.parent))

import tensordiagram as td
import numpy as np


# Reference generation functions
def generate_shape_3x4(fixtures_dir):
    """Generate reference_shape_3x4.svg"""
    print("Generating reference_shape_3x4.svg...")
    diagram = td.to_diagram((3, 4))
    diagram.render_svg(str(fixtures_dir / "reference_shape_3x4.svg"), height=128)


def generate_2x2_values(fixtures_dir):
    """Generate reference_2x2_values.svg"""
    print("Generating reference_2x2_values.svg...")
    tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
    diagram = td.to_diagram(tensor).fill_values()
    diagram.render_svg(str(fixtures_dir / "reference_2x2_values.svg"), height=128)


def generate_1d_values(fixtures_dir):
    """Generate reference_1d_values.svg"""
    print("Generating reference_1d_values.svg...")
    tensor = np.arange(5, dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_values()
    diagram.render_svg(str(fixtures_dir / "reference_1d_values.svg"), height=128)


def generate_5x7_no_values(fixtures_dir):
    """Generate reference_5x7_no_values.svg"""
    print("Generating reference_5x7_no_values.svg...")
    tensor = np.random.randn(5, 7)
    diagram = td.to_diagram(tensor)
    diagram.render_svg(str(fixtures_dir / "reference_5x7_no_values.svg"), height=200)


def generate_single_element(fixtures_dir):
    """Generate reference_single_element.svg"""
    print("Generating reference_single_element.svg...")
    tensor = np.array([42.0])
    diagram = td.to_diagram(tensor).fill_values()
    diagram.render_svg(str(fixtures_dir / "reference_single_element.svg"), height=128)


def generate_styled_color(fixtures_dir):
    """Generate reference_styled_color.svg"""
    print("Generating reference_styled_color.svg...")
    diagram = td.to_diagram((3, 4)).fill_color("blue")
    diagram.render_svg(str(fixtures_dir / "reference_styled_color.svg"), height=128)


def generate_styled_opacity_only(fixtures_dir):
    """Generate reference_styled_opacity_only.svg"""
    print("Generating reference_styled_opacity_only.svg...")
    diagram = td.to_diagram((3, 4)).fill_opacity(0.5)
    diagram.render_svg(
        str(fixtures_dir / "reference_styled_opacity_only.svg"), height=128
    )


def generate_styled_region(fixtures_dir):
    """Generate reference_styled_region.svg"""
    print("Generating reference_styled_region.svg...")
    tensor = np.arange(16, dtype=np.float32).reshape(4, 4)
    diagram = td.to_diagram(tensor).fill_region(
        start_coord=(0, 0), end_coord=(2, 2), color="coral", opacity=None
    )
    diagram.render_svg(str(fixtures_dir / "reference_styled_region.svg"), height=128)


def generate_styled_chained(fixtures_dir):
    """Generate reference_styled_chained.svg"""
    print("Generating reference_styled_chained.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = (
        td.to_diagram(tensor).fill_color("green").fill_opacity(0.6).fill_values()
    )
    diagram.render_svg(str(fixtures_dir / "reference_styled_chained.svg"), height=128)


def generate_styled_1d(fixtures_dir):
    """Generate reference_styled_1d.svg"""
    print("Generating reference_styled_1d.svg...")
    tensor = np.arange(5, dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_color("red").fill_opacity(0.8)
    diagram.render_svg(str(fixtures_dir / "reference_styled_1d.svg"), height=128)


def generate_styled_1d_gradient_reversed(fixtures_dir):
    """Generate reference_styled_1d_gradient_reversed.svg"""
    print("Generating reference_styled_1d_gradient_reversed.svg...")
    tensor = np.arange(5, dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_opacity(1.0, 0.3)
    diagram.render_svg(str(fixtures_dir / "reference_styled_1d_gradient_reversed.svg"), height=128)


def generate_styled_gradient(fixtures_dir):
    """Generate reference_styled_gradient.svg"""
    print("Generating reference_styled_gradient.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9)
    diagram.render_svg(
        str(fixtures_dir / "reference_styled_gradient.svg"), height=128
    )


def generate_styled_gradient_reversed(fixtures_dir):
    """Generate reference_styled_gradient_reversed.svg"""
    print("Generating reference_styled_gradient_reversed.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.9, 0.2)
    diagram.render_svg(
        str(fixtures_dir / "reference_styled_gradient_reversed.svg"), height=128
    )


def generate_styled_single_element(fixtures_dir):
    """Generate reference_styled_single_element.svg"""
    print("Generating reference_styled_single_element.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).fill_region(
        start_coord=(1, 2), end_coord=(1, 2), color="green", opacity=None
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_styled_single_element.svg"), height=128
    )


def generate_3d_tensor(fixtures_dir):
    """Generate reference_3d_tensor.svg"""
    print("Generating reference_3d_tensor.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor)
    diagram.render_svg(str(fixtures_dir / "reference_3d_tensor.svg"), height=128)


def generate_styled_3d(fixtures_dir):
    """Generate reference_styled_3d.svg"""
    print("Generating reference_styled_3d.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_color("green").fill_opacity(0.7)
    diagram.render_svg(str(fixtures_dir / "reference_styled_3d.svg"), height=128)


def generate_styled_3d_gradient_reversed(fixtures_dir):
    """Generate reference_styled_3d_gradient_reversed.svg"""
    print("Generating reference_styled_3d_gradient_reversed.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.95, 0.25)
    diagram.render_svg(str(fixtures_dir / "reference_styled_3d_gradient_reversed.svg"), height=128)


def generate_3d_fill_region(fixtures_dir):
    """Generate reference_3d_fill_region.svg"""
    print("Generating reference_3d_fill_region.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_region(
        start_coord=(0, 1, 1), end_coord=(1, 2, 3), color="green", opacity=None
    )
    diagram.render_svg(str(fixtures_dir / "reference_3d_fill_region.svg"), height=128)


def generate_3d_fill_region_single(fixtures_dir):
    """Generate reference_3d_fill_region_single.svg"""
    print("Generating reference_3d_fill_region_single.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_region(
        start_coord=(1, 1, 0), end_coord=(1, 1, 0), color="red", opacity=0.8
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_fill_region_single.svg"), height=128
    )


def generate_3d_fill_region_slice(fixtures_dir):
    """Generate reference_3d_fill_region_slice.svg"""
    print("Generating reference_3d_fill_region_slice.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_region(
        start_coord=(0, 0, 0), end_coord=(0, 2, 3), color="blue", opacity=0.6
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_fill_region_slice.svg"), height=128
    )


def generate_1d_annotate_row_default(fixtures_dir):
    """Generate reference_1d_annotate_row_default.svg"""
    print("Generating reference_1d_annotate_row_default.svg...")
    tensor = np.arange(5, dtype=np.float32)
    diagram = td.to_diagram(tensor).annotate_dim_size("row")
    diagram.render_svg(
        str(fixtures_dir / "reference_1d_annotate_row_default.svg"), height=128
    )


def generate_1d_annotate_row_custom(fixtures_dir):
    """Generate reference_1d_annotate_row_custom.svg"""
    print("Generating reference_1d_annotate_row_custom.svg...")
    tensor = np.arange(5, dtype=np.float32)
    diagram = td.to_diagram(tensor).annotate_dim_size("row", color="red")
    diagram.render_svg(
        str(fixtures_dir / "reference_1d_annotate_row_custom.svg"), height=128
    )


def generate_2d_annotate_row_only(fixtures_dir):
    """Generate reference_2d_annotate_row_only.svg"""
    print("Generating reference_2d_annotate_row_only.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).annotate_dim_size("row")
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_row_only.svg"), height=128
    )


def generate_2d_annotate_col_only(fixtures_dir):
    """Generate reference_2d_annotate_col_only.svg"""
    print("Generating reference_2d_annotate_col_only.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).annotate_dim_size("col")
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_col_only.svg"), height=128
    )


def generate_2d_annotate_all_default(fixtures_dir):
    """Generate reference_2d_annotate_all_default.svg"""
    print("Generating reference_2d_annotate_all_default.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).annotate_dim_size("all")
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_all_default.svg"), height=128
    )


def generate_2d_annotate_all_custom(fixtures_dir):
    """Generate reference_2d_annotate_all_custom.svg"""
    print("Generating reference_2d_annotate_all_custom.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).annotate_dim_size("all", color="green")
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_all_custom.svg"), height=128
    )


def generate_3d_annotate_all(fixtures_dir):
    """Generate reference_3d_annotate_all.svg"""
    print("Generating reference_3d_annotate_all.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).annotate_dim_size("all")
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_annotate_all.svg"), height=128
    )


def generate_3d_annotate_col_depth(fixtures_dir):
    """Generate reference_3d_annotate_col_depth.svg"""
    print("Generating reference_3d_annotate_col_depth.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = (
        td.to_diagram(tensor).annotate_dim_size("col").annotate_dim_size("depth")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_annotate_col_depth.svg"), height=128
    )


def generate_2d_annotate_fill_color(fixtures_dir):
    """Generate reference_2d_annotate_fill_color.svg"""
    print("Generating reference_2d_annotate_fill_color.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_size("all", color="blue")
        .fill_color("blue")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_fill_color.svg"), height=128
    )


def generate_2d_annotate_fill_region(fixtures_dir):
    """Generate reference_2d_annotate_fill_region.svg"""
    print("Generating reference_2d_annotate_fill_region.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_size("all", color="red")
        .fill_region(
            start_coord=(0, 0), end_coord=(2, 2), color="green", opacity=None
        )
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_fill_region.svg"), height=128
    )


def generate_1d_annotate_indices(fixtures_dir):
    """Generate reference_1d_annotate_indices.svg"""
    print("Generating reference_1d_annotate_indices.svg...")
    tensor = np.arange(8, dtype=np.float32)
    diagram = td.to_diagram(tensor).annotate_dim_indices("all")
    diagram.render_svg(
        str(fixtures_dir / "reference_1d_annotate_indices.svg"), height=128
    )


def generate_1d_annotate_indices_color(fixtures_dir):
    """Generate reference_1d_annotate_indices_color.svg"""
    print("Generating reference_1d_annotate_indices_color.svg...")
    tensor = np.arange(8, dtype=np.float32)
    diagram = td.to_diagram(tensor).annotate_dim_indices("all", color="red")
    diagram.render_svg(
        str(fixtures_dir / "reference_1d_annotate_indices_color.svg"), height=128
    )


def generate_1d_annotate_size_indices(fixtures_dir):
    """Generate reference_1d_annotate_size_indices.svg"""
    print("Generating reference_1d_annotate_size_indices.svg...")
    tensor = np.arange(8, dtype=np.float32)
    diagram = (
        td.to_diagram(tensor).annotate_dim_size("all").annotate_dim_indices("all")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_1d_annotate_size_indices.svg"), height=128
    )


def generate_2d_annotate_indices_row(fixtures_dir):
    """Generate reference_2d_annotate_indices_row.svg"""
    print("Generating reference_2d_annotate_indices_row.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).annotate_dim_indices("row")
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_indices_row.svg"), height=128
    )


def generate_2d_annotate_indices_col(fixtures_dir):
    """Generate reference_2d_annotate_indices_col.svg"""
    print("Generating reference_2d_annotate_indices_col.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).annotate_dim_indices("col")
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_indices_col.svg"), height=128
    )


def generate_2d_annotate_indices_all(fixtures_dir):
    """Generate reference_2d_annotate_indices_all.svg"""
    print("Generating reference_2d_annotate_indices_all.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).annotate_dim_indices("all")
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_indices_all.svg"), height=128
    )


def generate_2d_annotate_indices_all_color(fixtures_dir):
    """Generate reference_2d_annotate_indices_all_color.svg"""
    print("Generating reference_2d_annotate_indices_all_color.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).annotate_dim_indices("all", color="blue")
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_indices_all_color.svg"), height=128
    )


def generate_2d_annotate_size_indices(fixtures_dir):
    """Generate reference_2d_annotate_size_indices.svg"""
    print("Generating reference_2d_annotate_size_indices.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_size("all", color="red")
        .annotate_dim_indices("all", color="blue")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_size_indices.svg"), height=128
    )


def generate_3d_annotate_indices_row(fixtures_dir):
    """Generate reference_3d_annotate_indices_row.svg"""
    print("Generating reference_3d_annotate_indices_row.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).annotate_dim_indices("row")
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_annotate_indices_row.svg"), height=128
    )


def generate_3d_annotate_indices_all(fixtures_dir):
    """Generate reference_3d_annotate_indices_all.svg"""
    print("Generating reference_3d_annotate_indices_all.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).annotate_dim_indices("all")
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_annotate_indices_all.svg"), height=128
    )


def generate_3d_annotate_indices_col_depth(fixtures_dir):
    """Generate reference_3d_annotate_indices_col_depth.svg"""
    print("Generating reference_3d_annotate_indices_col_depth.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_indices("col", color="red")
        .annotate_dim_indices("depth", color="blue")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_annotate_indices_col_depth.svg"), height=128
    )


def generate_3d_annotate_size_indices(fixtures_dir):
    """Generate reference_3d_annotate_size_indices.svg"""
    print("Generating reference_3d_annotate_size_indices.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_size("all", color="green")
        .annotate_dim_indices("all", color="gray")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_annotate_size_indices.svg"), height=128
    )


def generate_2d_annotate_indices_fill_color(fixtures_dir):
    """Generate reference_2d_annotate_indices_fill_color.svg"""
    print("Generating reference_2d_annotate_indices_fill_color.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_indices("all", color="red")
        .fill_color("blue")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_annotate_indices_fill_color.svg"), height=128
    )


def generate_2d_row_size_col_indices(fixtures_dir):
    """Generate reference_2d_row_size_col_indices.svg"""
    print("Generating reference_2d_row_size_col_indices.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_size("row", color="red")
        .annotate_dim_indices("col", color="blue")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_row_size_col_indices.svg"), height=128
    )


def generate_2d_row_indices_col_size(fixtures_dir):
    """Generate reference_2d_row_indices_col_size.svg"""
    print("Generating reference_2d_row_indices_col_size.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_indices("row", color="blue")
        .annotate_dim_size("col", color="red")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_row_indices_col_size.svg"), height=128
    )


def generate_3d_row_size_col_indices_depth_size(fixtures_dir):
    """Generate reference_3d_row_size_col_indices_depth_size.svg"""
    print("Generating reference_3d_row_size_col_indices_depth_size.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_size("row", color="red")
        .annotate_dim_indices("col", color="blue")
        .annotate_dim_size("depth", color="green")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_row_size_col_indices_depth_size.svg"), height=128
    )


def generate_3d_row_indices_col_size_depth_indices(fixtures_dir):
    """Generate reference_3d_row_indices_col_size_depth_indices.svg"""
    print("Generating reference_3d_row_indices_col_size_depth_indices.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_indices("row", color="blue")
        .annotate_dim_size("col", color="red")
        .annotate_dim_indices("depth", color="gray")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_row_indices_col_size_depth_indices.svg"), height=128
    )


def generate_3d_row_size_col_size_depth_indices(fixtures_dir):
    """Generate reference_3d_row_size_col_size_depth_indices.svg"""
    print("Generating reference_3d_row_size_col_size_depth_indices.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_size("row", color="red")
        .annotate_dim_size("col", color="green")
        .annotate_dim_indices("depth", color="blue")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_row_size_col_size_depth_indices.svg"), height=128
    )


def generate_3d_row_indices_col_indices_depth_size(fixtures_dir):
    """Generate reference_3d_row_indices_col_indices_depth_size.svg"""
    print("Generating reference_3d_row_indices_col_indices_depth_size.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_indices("row", color="blue")
        .annotate_dim_indices("col", color="gray")
        .annotate_dim_size("depth", color="red")
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_row_indices_col_indices_depth_size.svg"), height=128
    )


def generate_2d_mixed_annotate_fill_region(fixtures_dir):
    """Generate reference_2d_mixed_annotate_fill_region.svg"""
    print("Generating reference_2d_mixed_annotate_fill_region.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = (
        td.to_diagram(tensor)
        .annotate_dim_size("row", color="red")
        .annotate_dim_indices("col", color="blue")
        .fill_region(start_coord=(0, 0), end_coord=(2, 2), color="green", opacity=None)
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_mixed_annotate_fill_region.svg"), height=128
    )


def generate_2d_function_color_value(fixtures_dir):
    """Generate reference_2d_function_color_value.svg"""
    print("Generating reference_2d_function_color_value.svg...")
    tensor = np.array([[1, -2, 3, -4], [5, -6, 7, -8], [9, -10, 11, -12]], dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_color(
        lambda idx, val: "red" if val > 0 else "blue" # type: ignore[arg-type]
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_function_color_value.svg"), height=128
    )


def generate_2d_function_color_index(fixtures_dir):
    """Generate reference_2d_function_color_index.svg"""
    print("Generating reference_2d_function_color_index.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).fill_color(
        lambda idx, val: "green" if idx[0] % 2 == 0 else "purple" # type: ignore[arg-type]
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_function_color_index.svg"), height=128
    )


def generate_2d_function_opacity_value(fixtures_dir):
    """Generate reference_2d_function_opacity_value.svg"""
    print("Generating reference_2d_function_opacity_value.svg...")
    tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_opacity(
        lambda idx, val: val / 15.0 # type: ignore[arg-type]
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_function_opacity_value.svg"), height=128
    )


def generate_2d_function_opacity_index(fixtures_dir):
    """Generate reference_2d_function_opacity_index.svg"""
    print("Generating reference_2d_function_opacity_index.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(
        lambda idx, val: (idx[0] + idx[1]) / 6.0 # type: ignore[arg-type]
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_function_opacity_index.svg"), height=128
    )


def generate_2d_function_both(fixtures_dir):
    """Generate reference_2d_function_both.svg"""
    print("Generating reference_2d_function_both.svg...")
    tensor = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_region(
        start_coord=(0, 0),
        end_coord=(1, 2),
        color=lambda idx, val: "orange" if val > 0 else "cyan", # type: ignore[arg-type]
        opacity=lambda idx, val: abs(val) / 10.0 # type: ignore[arg-type]
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_function_both.svg"), height=128
    )


def generate_1d_function_color(fixtures_dir):
    """Generate reference_1d_function_color.svg"""
    print("Generating reference_1d_function_color.svg...")
    tensor = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_color(
        lambda idx, val: "red" if idx % 2 == 0 else "blue" # type: ignore[arg-type]
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_1d_function_color.svg"), height=128
    )


def generate_3d_function_color(fixtures_dir):
    """Generate reference_3d_function_color.svg"""
    print("Generating reference_3d_function_color.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_color(
        lambda idx, val: "red" if idx[2] < 2 else "blue" # type: ignore[arg-type]
    )
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_function_color.svg"), height=128
    )


# Gradient order reference generators - 2D
def generate_2d_gradient_order_R(fixtures_dir):
    """Generate reference_2d_gradient_order_R.svg"""
    from tensordiagram.types import TensorOrder
    print("Generating reference_2d_gradient_order_R.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.R)
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_gradient_order_R.svg"), height=128
    )


def generate_2d_gradient_order_C(fixtures_dir):
    """Generate reference_2d_gradient_order_C.svg"""
    from tensordiagram.types import TensorOrder
    print("Generating reference_2d_gradient_order_C.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.C)
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_gradient_order_C.svg"), height=128
    )


def generate_2d_gradient_order_CR(fixtures_dir):
    """Generate reference_2d_gradient_order_CR.svg"""
    from tensordiagram.types import TensorOrder
    print("Generating reference_2d_gradient_order_CR.svg...")
    tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.CR)
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_gradient_order_CR.svg"), height=128
    )


# Gradient order reference generators - 3D
def generate_3d_gradient_order_R(fixtures_dir):
    """Generate reference_3d_gradient_order_R.svg"""
    from tensordiagram.types import TensorOrder
    print("Generating reference_3d_gradient_order_R.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.R)
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_gradient_order_R.svg"), height=128
    )


def generate_3d_gradient_order_C(fixtures_dir):
    """Generate reference_3d_gradient_order_C.svg"""
    from tensordiagram.types import TensorOrder
    print("Generating reference_3d_gradient_order_C.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.C)
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_gradient_order_C.svg"), height=128
    )


def generate_3d_gradient_order_D(fixtures_dir):
    """Generate reference_3d_gradient_order_D.svg"""
    from tensordiagram.types import TensorOrder
    print("Generating reference_3d_gradient_order_D.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.D)
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_gradient_order_D.svg"), height=128
    )


def generate_3d_gradient_order_RD(fixtures_dir):
    """Generate reference_3d_gradient_order_RD.svg"""
    from tensordiagram.types import TensorOrder
    print("Generating reference_3d_gradient_order_RD.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.RD)
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_gradient_order_RD.svg"), height=128
    )


def generate_3d_gradient_order_RDC(fixtures_dir):
    """Generate reference_3d_gradient_order_RDC.svg"""
    from tensordiagram.types import TensorOrder
    print("Generating reference_3d_gradient_order_RDC.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.RDC)
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_gradient_order_RDC.svg"), height=128
    )


def generate_3d_gradient_order_DCR(fixtures_dir):
    """Generate reference_3d_gradient_order_DCR.svg"""
    from tensordiagram.types import TensorOrder
    print("Generating reference_3d_gradient_order_DCR.svg...")
    tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.DCR)
    diagram.render_svg(
        str(fixtures_dir / "reference_3d_gradient_order_DCR.svg"), height=128
    )


# fill_values with custom formatting reference generators
def generate_2d_fill_values_custom_font_size(fixtures_dir):
    """Generate reference_2d_fill_values_custom_font_size.svg"""
    print("Generating reference_2d_fill_values_custom_font_size.svg...")
    tensor = np.array([[1.5, 2.3], [3.7, 4.1]], dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_values(font_size=0.8)
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_fill_values_custom_font_size.svg"), height=128
    )


def generate_2d_fill_values_percentage(fixtures_dir):
    """Generate reference_2d_fill_values_percentage.svg"""
    print("Generating reference_2d_fill_values_percentage.svg...")
    tensor = np.array([[0.123, 0.456], [0.789, 0.234]], dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_values(format_fn=lambda x: f"{x*100:.1f}%")
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_fill_values_percentage.svg"), height=128
    )


def generate_2d_fill_values_scientific(fixtures_dir):
    """Generate reference_2d_fill_values_scientific.svg"""
    print("Generating reference_2d_fill_values_scientific.svg...")
    tensor = np.array([[1000, 2000], [3000, 4000]], dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_values(format_fn=lambda x: f"{x:.1e}")
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_fill_values_scientific.svg"), height=128
    )


def generate_2d_fill_values_integer(fixtures_dir):
    """Generate reference_2d_fill_values_integer.svg"""
    print("Generating reference_2d_fill_values_integer.svg...")
    tensor = np.array([[1.5, 2.7], [3.2, 4.9]], dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_values(format_fn=lambda x: str(int(x)))
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_fill_values_integer.svg"), height=128
    )


def generate_2d_fill_values_size_and_format(fixtures_dir):
    """Generate reference_2d_fill_values_size_and_format.svg"""
    print("Generating reference_2d_fill_values_size_and_format.svg...")
    tensor = np.array([[1.234, 5.678], [9.012, 3.456]], dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_values(font_size=0.5, format_fn=lambda x: f"{x:.1f}")
    diagram.render_svg(
        str(fixtures_dir / "reference_2d_fill_values_size_and_format.svg"), height=128
    )


def generate_1d_fill_values_custom_font_size(fixtures_dir):
    """Generate reference_1d_fill_values_custom_font_size.svg"""
    print("Generating reference_1d_fill_values_custom_font_size.svg...")
    tensor = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_values(font_size=1.0)
    diagram.render_svg(
        str(fixtures_dir / "reference_1d_fill_values_custom_font_size.svg"), height=128
    )


def generate_1d_fill_values_custom_format(fixtures_dir):
    """Generate reference_1d_fill_values_custom_format.svg"""
    print("Generating reference_1d_fill_values_custom_format.svg...")
    tensor = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)
    diagram = td.to_diagram(tensor).fill_values(format_fn=lambda x: f"{x:.0f}")
    diagram.render_svg(
        str(fixtures_dir / "reference_1d_fill_values_custom_format.svg"), height=128
    )


# Registry of all generators
GENERATORS = {
    "shape_3x4": generate_shape_3x4,
    "2x2_values": generate_2x2_values,
    "1d_values": generate_1d_values,
    "5x7_no_values": generate_5x7_no_values,
    "single_element": generate_single_element,
    "styled_color": generate_styled_color,
    "styled_opacity_only": generate_styled_opacity_only,
    "styled_region": generate_styled_region,
    "styled_chained": generate_styled_chained,
    "styled_1d": generate_styled_1d,
    "styled_1d_gradient_reversed": generate_styled_1d_gradient_reversed,
    "styled_gradient": generate_styled_gradient,
    "styled_gradient_reversed": generate_styled_gradient_reversed,
    "styled_single_element": generate_styled_single_element,
    "3d_tensor": generate_3d_tensor,
    "styled_3d": generate_styled_3d,
    "styled_3d_gradient_reversed": generate_styled_3d_gradient_reversed,
    "3d_fill_region": generate_3d_fill_region,
    "3d_fill_region_single": generate_3d_fill_region_single,
    "3d_fill_region_slice": generate_3d_fill_region_slice,
    "1d_annotate_row_default": generate_1d_annotate_row_default,
    "1d_annotate_row_custom": generate_1d_annotate_row_custom,
    "2d_annotate_row_only": generate_2d_annotate_row_only,
    "2d_annotate_col_only": generate_2d_annotate_col_only,
    "2d_annotate_all_default": generate_2d_annotate_all_default,
    "2d_annotate_all_custom": generate_2d_annotate_all_custom,
    "3d_annotate_all": generate_3d_annotate_all,
    "3d_annotate_col_depth": generate_3d_annotate_col_depth,
    "2d_annotate_fill_color": generate_2d_annotate_fill_color,
    "2d_annotate_fill_region": generate_2d_annotate_fill_region,
    "1d_annotate_indices": generate_1d_annotate_indices,
    "1d_annotate_indices_color": generate_1d_annotate_indices_color,
    "1d_annotate_size_indices": generate_1d_annotate_size_indices,
    "2d_annotate_indices_row": generate_2d_annotate_indices_row,
    "2d_annotate_indices_col": generate_2d_annotate_indices_col,
    "2d_annotate_indices_all": generate_2d_annotate_indices_all,
    "2d_annotate_indices_all_color": generate_2d_annotate_indices_all_color,
    "2d_annotate_size_indices": generate_2d_annotate_size_indices,
    "3d_annotate_indices_row": generate_3d_annotate_indices_row,
    "3d_annotate_indices_all": generate_3d_annotate_indices_all,
    "3d_annotate_indices_col_depth": generate_3d_annotate_indices_col_depth,
    "3d_annotate_size_indices": generate_3d_annotate_size_indices,
    "2d_annotate_indices_fill_color": generate_2d_annotate_indices_fill_color,
    "2d_row_size_col_indices": generate_2d_row_size_col_indices,
    "2d_row_indices_col_size": generate_2d_row_indices_col_size,
    "3d_row_size_col_indices_depth_size": generate_3d_row_size_col_indices_depth_size,
    "3d_row_indices_col_size_depth_indices": generate_3d_row_indices_col_size_depth_indices,
    "3d_row_size_col_size_depth_indices": generate_3d_row_size_col_size_depth_indices,
    "3d_row_indices_col_indices_depth_size": generate_3d_row_indices_col_indices_depth_size,
    "2d_mixed_annotate_fill_region": generate_2d_mixed_annotate_fill_region,
    "2d_function_color_value": generate_2d_function_color_value,
    "2d_function_color_index": generate_2d_function_color_index,
    "2d_function_opacity_value": generate_2d_function_opacity_value,
    "2d_function_opacity_index": generate_2d_function_opacity_index,
    "2d_function_both": generate_2d_function_both,
    "1d_function_color": generate_1d_function_color,
    "3d_function_color": generate_3d_function_color,
    # Gradient order references - 2D
    "2d_gradient_order_R": generate_2d_gradient_order_R,
    "2d_gradient_order_C": generate_2d_gradient_order_C,
    "2d_gradient_order_CR": generate_2d_gradient_order_CR,
    # Gradient order references - 3D
    "3d_gradient_order_R": generate_3d_gradient_order_R,
    "3d_gradient_order_C": generate_3d_gradient_order_C,
    "3d_gradient_order_D": generate_3d_gradient_order_D,
    "3d_gradient_order_RD": generate_3d_gradient_order_RD,
    "3d_gradient_order_RDC": generate_3d_gradient_order_RDC,
    "3d_gradient_order_DCR": generate_3d_gradient_order_DCR,
    # fill_values with custom formatting
    "2d_fill_values_custom_font_size": generate_2d_fill_values_custom_font_size,
    "2d_fill_values_percentage": generate_2d_fill_values_percentage,
    "2d_fill_values_scientific": generate_2d_fill_values_scientific,
    "2d_fill_values_integer": generate_2d_fill_values_integer,
    "2d_fill_values_size_and_format": generate_2d_fill_values_size_and_format,
    "1d_fill_values_custom_font_size": generate_1d_fill_values_custom_font_size,
    "1d_fill_values_custom_format": generate_1d_fill_values_custom_format,
}


def main():
    """Generate reference images based on command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate reference images for visual regression testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available references
  %(prog)s --list

  # Generate all references
  %(prog)s --all

  # Generate specific references
  %(prog)s styled_gradient_reversed styled_1d_gradient_reversed

  # Generate all gradient-related references (use shell expansion)
  %(prog)s styled_gradient styled_gradient_reversed styled_1d_gradient_reversed styled_3d_gradient_reversed
        """
    )

    parser.add_argument(
        "references",
        nargs="*",
        help="List of reference names to generate (omit 'reference_' prefix and '.svg' extension)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate all reference images"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all available reference names"
    )

    args = parser.parse_args()

    # If no arguments provided, show usage
    if not args.references and not args.all and not args.list:
        parser.print_help()
        return

    fixtures_dir = Path(__file__).parent / "fixtures"
    fixtures_dir.mkdir(exist_ok=True)

    # Handle --list flag
    if args.list:
        print("Available reference names:")
        for name in sorted(GENERATORS.keys()):
            print(f"  - {name}")
        return

    # Handle --all flag
    if args.all:
        print(f"Generating all reference images in {fixtures_dir}\n")
        for name, generator in GENERATORS.items():
            generator(fixtures_dir)

        print(f"\n✓ All reference images generated successfully in {fixtures_dir}")
        print("\nGenerated files:")
        for ref_file in sorted(fixtures_dir.glob("reference_*.svg")):
            print(f"  - {ref_file.name}")
        return

    # Generate specific references
    print(f"Generating selected reference images in {fixtures_dir}\n")
    generated = []
    not_found = []

    for ref_name in args.references:
        if ref_name in GENERATORS:
            GENERATORS[ref_name](fixtures_dir)
            generated.append(ref_name)
        else:
            not_found.append(ref_name)

    # Print summary
    if generated:
        print(f"\n✓ Generated {len(generated)} reference(s) successfully:")
        for name in generated:
            print(f"  - {name}")

    if not_found:
        print(f"\n✗ Reference(s) not found:")
        for name in not_found:
            print(f"  - {name}")
        print("\nUse --list to see all available reference names.")
        sys.exit(1)


if __name__ == "__main__":
    main()

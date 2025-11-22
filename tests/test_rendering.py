"""Tests for TensorDiagram rendering functionality."""
import pytest
import numpy as np
import tensordiagram as td


@pytest.mark.rendering
class TestSVGRendering:
    """Tests for SVG rendering."""

    def test_render_svg_basic(self, shape_2d, temp_output_dir):
        """Test basic SVG rendering."""
        diagram = td.to_diagram(shape_2d)
        output_path = temp_output_dir / "test_basic.svg"
        diagram.render_svg(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_svg_with_height(self, shape_2d, temp_output_dir):
        """Test SVG rendering with custom height."""
        diagram = td.to_diagram(shape_2d)
        output_path = temp_output_dir / "test_height.svg"
        diagram.render_svg(str(output_path), height=256)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_svg_with_width(self, shape_2d, temp_output_dir):
        """Test SVG rendering with custom width."""
        diagram = td.to_diagram(shape_2d)
        output_path = temp_output_dir / "test_width.svg"
        diagram.render_svg(str(output_path), width=256)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_svg_with_values(self, numpy_array_2d_square, temp_output_dir):
        """Test SVG rendering with values displayed."""
        diagram = td.to_diagram(numpy_array_2d_square).fill_values()
        output_path = temp_output_dir / "test_values.svg"
        diagram.render_svg(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

        # Verify the SVG contains text elements (values)
        content = output_path.read_text()
        assert "text" in content.lower()

    def test_render_svg_1d(self, shape_1d, temp_output_dir):
        """Test SVG rendering for 1D tensor."""
        diagram = td.to_diagram(shape_1d)
        output_path = temp_output_dir / "test_1d.svg"
        diagram.render_svg(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_svg_3d(self, temp_output_dir):
        """Test SVG rendering for 3D tensor."""
        tensor = np.random.randn(2, 3, 4)
        diagram = td.to_diagram(tensor)
        output_path = temp_output_dir / "test_3d.svg"
        diagram.render_svg(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0


@pytest.mark.rendering
class TestPNGRendering:
    """Tests for PNG rendering (requires pycairo)."""

    def test_render_png_basic(self, shape_2d, temp_output_dir):
        """Test basic PNG rendering."""
        pytest.importorskip("cairo", reason="pycairo not installed")

        diagram = td.to_diagram(shape_2d)
        output_path = temp_output_dir / "test_basic.png"
        diagram.render_png(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_png_with_height(self, shape_2d, temp_output_dir):
        """Test PNG rendering with custom height."""
        pytest.importorskip("cairo", reason="pycairo not installed")

        diagram = td.to_diagram(shape_2d)
        output_path = temp_output_dir / "test_height.png"
        diagram.render_png(str(output_path), height=256)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_png_with_values(self, numpy_array_2d_square, temp_output_dir):
        """Test PNG rendering with values displayed."""
        pytest.importorskip("cairo", reason="pycairo not installed")

        diagram = td.to_diagram(numpy_array_2d_square).fill_values()
        output_path = temp_output_dir / "test_values.png"
        diagram.render_png(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_png_3d(self, temp_output_dir):
        """Test PNG rendering for 3D tensor."""
        pytest.importorskip("cairo", reason="pycairo not installed")
        tensor = np.random.randn(2, 3, 4)
        diagram = td.to_diagram(tensor)
        output_path = temp_output_dir / "test_3d.png"
        diagram.render_png(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0


@pytest.mark.skip
@pytest.mark.rendering
class TestPDFRendering:
    """Tests for PDF rendering (requires pycairo)."""

    def test_render_pdf_basic(self, shape_2d, temp_output_dir):
        """Test basic PDF rendering."""
        pytest.importorskip("cairo", reason="pycairo not installed")

        diagram = td.to_diagram(shape_2d)
        output_path = temp_output_dir / "test_basic.pdf"
        diagram.render_pdf(str(output_path))

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_render_pdf_with_height(self, shape_2d, temp_output_dir):
        """Test PDF rendering with custom height."""
        pytest.importorskip("cairo", reason="pycairo not installed")

        diagram = td.to_diagram(shape_2d)
        output_path = temp_output_dir / "test_height.pdf"
        diagram.render_pdf(str(output_path), height=256)

        assert output_path.exists()
        assert output_path.stat().st_size > 0


@pytest.mark.rendering
@pytest.mark.visual
class TestVisualRegression:
    """Visual regression tests comparing rendered outputs."""

    def test_consistent_svg_output(self, numpy_array_2d_square, temp_output_dir):
        """Test that rendering the same diagram produces consistent SVG."""
        diagram = td.to_diagram(numpy_array_2d_square).fill_values()

        output1 = temp_output_dir / "output1.svg"
        output2 = temp_output_dir / "output2.svg"

        diagram.render_svg(str(output1))
        diagram.render_svg(str(output2))

        # SVG outputs should be identical
        content1 = output1.read_text()
        content2 = output2.read_text()
        assert content1 == content2

    def test_reference_comparison_2x2_tensor(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering matches reference for 2x2 tensor."""
        # Create a known tensor
        tensor = np.array([[1.0, 2.0], [3.0, 4.0]])
        diagram = td.to_diagram(tensor).fill_values()

        output_path = temp_output_dir / "2x2_values.svg"
        diagram.render_svg(str(output_path), height=128)

        # Compare with reference if it exists
        reference_path = fixtures_dir / "reference_2x2_values.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            # If reference doesn't exist, this test passes but warns
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_comparison_1d_tensor(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering matches reference for 1D tensor."""
        tensor = np.arange(5, dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_values()

        output_path = temp_output_dir / "1d_values.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_1d_values.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_comparison_shape_only(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering matches reference for shape-only diagram."""
        diagram = td.to_diagram((3, 4))

        output_path = temp_output_dir / "shape_3x4.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_shape_3x4.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    # Styled diagram tests
    def test_reference_styled_fill_color(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering with fill_color matches reference."""
        diagram = td.to_diagram((3, 4)).fill_color("blue")
        output_path = temp_output_dir / "styled_color.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_styled_color.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_styled_fill_opacity(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering with fill_opacity matches reference."""
        diagram = td.to_diagram((3, 4)).fill_opacity(0.5)
        output_path = temp_output_dir / "styled_opacity.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_styled_opacity_only.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_styled_fill_region(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering with fill_region matches reference."""
        tensor = np.arange(16, dtype=np.float32).reshape(4, 4)
        diagram = td.to_diagram(tensor).fill_region(
            start_coord=(0, 0), end_coord=(2, 2), color="coral", opacity=None
        )
        output_path = temp_output_dir / "styled_region.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_styled_region.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_styled_chained(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering with chained styles matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = (
            td.to_diagram(tensor)
            .fill_color("green")
            .fill_opacity(0.6)
            .fill_values()
        )
        output_path = temp_output_dir / "styled_chained.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_styled_chained.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_styled_1d(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering styled 1D tensor matches reference."""
        tensor = np.arange(5, dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_color("red").fill_opacity(0.8)
        output_path = temp_output_dir / "styled_1d.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_styled_1d.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_styled_1d_gradient_reversed(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering 1D tensor with reversed opacity gradient matches reference."""
        tensor = np.arange(5, dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_opacity(1.0, 0.3)
        output_path = temp_output_dir / "styled_1d_gradient_reversed.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_styled_1d_gradient_reversed.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_styled_gradient_opacity(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering with opacity gradient matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9)
        output_path = temp_output_dir / "styled_gradient.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_styled_gradient.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_styled_gradient_opacity_reversed(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering with reversed opacity gradient (high to low) matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.9, 0.2)
        output_path = temp_output_dir / "styled_gradient_reversed.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_styled_gradient_reversed.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")
    
    def test_reference_styled_single_element(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering with single element fill_region matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).fill_region(
            start_coord=(1, 2), end_coord=(1, 2), color="green", opacity=None
        )
        output_path = temp_output_dir / "styled_single_element.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_styled_single_element.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_comparison_3d_tensor(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering matches reference for 3D tensor."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor)

        output_path = temp_output_dir / "3d_tensor.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_tensor.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_styled_3d(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering styled 3D tensor matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_color("green").fill_opacity(0.7)
        output_path = temp_output_dir / "styled_3d.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_styled_3d.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_styled_3d_gradient_reversed(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering 3D tensor with reversed opacity gradient matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.95, 0.25)
        output_path = temp_output_dir / "styled_3d_gradient_reversed.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_styled_3d_gradient_reversed.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_fill_region(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering 3D tensor with fill_region matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_region(
            start_coord=(0, 1, 1), end_coord=(1, 2, 3), color="green", opacity=None
        )
        output_path = temp_output_dir / "3d_fill_region.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_fill_region.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_fill_region_single_element(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering 3D tensor with single element fill_region matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_region(
            start_coord=(1, 1, 0), end_coord=(1, 1, 0), color="red", opacity=0.8
        )
        output_path = temp_output_dir / "3d_fill_region_single.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_fill_region_single.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_fill_region_full_slice(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test rendering 3D tensor with full slice fill_region matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_region(
            start_coord=(0, 0, 0), end_coord=(0, 2, 3), color="blue", opacity=0.6
        )
        output_path = temp_output_dir / "3d_fill_region_slice.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_fill_region_slice.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    # Dimension size annotation tests
    def test_reference_1d_annotate_row_default_color(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 1D tensor with row size annotation (default color) matches reference."""
        tensor = np.arange(5, dtype=np.float32)
        diagram = td.to_diagram(tensor).annotate_dim_size(0)
        output_path = temp_output_dir / "1d_annotate_row_default.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_1d_annotate_row_default.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_1d_annotate_row_custom_color(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 1D tensor with row size annotation (custom color) matches reference."""
        tensor = np.arange(5, dtype=np.float32)
        diagram = td.to_diagram(tensor).annotate_dim_size(0, color="red")
        output_path = temp_output_dir / "1d_annotate_row_custom.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_1d_annotate_row_custom.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_row_only(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with row size annotation only matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_size(0)
        output_path = temp_output_dir / "2d_annotate_row_only.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_row_only.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_col_only(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with column size annotation only matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_size(1)
        output_path = temp_output_dir / "2d_annotate_col_only.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_col_only.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_all_default_color(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with all dimensions annotated (default color) matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_size()
        output_path = temp_output_dir / "2d_annotate_all_default.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_all_default.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_all_custom_color(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with all dimensions annotated (custom color) matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_size(color="green")
        output_path = temp_output_dir / "2d_annotate_all_custom.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_all_custom.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_annotate_all(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with all dimensions annotated matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_size()
        output_path = temp_output_dir / "3d_annotate_all.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_annotate_all.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_annotate_col_depth_only(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with col and depth annotations only matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_size(1)
            .annotate_dim_size(2)
        )
        output_path = temp_output_dir / "3d_annotate_col_depth.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_annotate_col_depth.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_with_fill_color(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D annotated diagram with fill_color matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_size(color="blue")
            .fill_color("blue")
        )
        output_path = temp_output_dir / "2d_annotate_fill_color.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_fill_color.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_with_fill_region(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D annotated diagram with fill_region matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_size(color="red")
            .fill_region(start_coord=(0, 0), end_coord=(2, 2), color="green", opacity=None)
        )
        output_path = temp_output_dir / "2d_annotate_fill_region.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_fill_region.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    # Dimension indices annotation tests
    def test_reference_1d_annotate_indices(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 1D tensor with indices annotated matches reference."""
        tensor = np.arange(8, dtype=np.float32)
        diagram = td.to_diagram(tensor).annotate_dim_indices()
        output_path = temp_output_dir / "1d_annotate_indices.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_1d_annotate_indices.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_1d_annotate_indices_color(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 1D tensor with indices annotated (custom color) matches reference."""
        tensor = np.arange(8, dtype=np.float32)
        diagram = td.to_diagram(tensor).annotate_dim_indices(color="red")
        output_path = temp_output_dir / "1d_annotate_indices_color.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_1d_annotate_indices_color.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_1d_annotate_size_indices(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 1D tensor with size and indices annotated matches reference."""
        tensor = np.arange(8, dtype=np.float32)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_size()
            .annotate_dim_indices()
        )
        output_path = temp_output_dir / "1d_annotate_size_indices.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_1d_annotate_size_indices.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_indices_row(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with row indices annotated matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_indices(0)
        output_path = temp_output_dir / "2d_annotate_indices_row.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_indices_row.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_indices_col(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with column indices annotated matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_indices(1)
        output_path = temp_output_dir / "2d_annotate_indices_col.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_indices_col.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_indices_all(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with all indices annotated matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_indices()
        output_path = temp_output_dir / "2d_annotate_indices_all.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_indices_all.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_indices_all_color(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with all indices annotated (custom color) matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_indices(color="blue")
        output_path = temp_output_dir / "2d_annotate_indices_all_color.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_indices_all_color.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_size_indices(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with size and indices annotated matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_size(color="red")
            .annotate_dim_indices(color="blue")
        )
        output_path = temp_output_dir / "2d_annotate_size_indices.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_size_indices.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_annotate_indices_row(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with row indices annotated matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_indices(0)
        output_path = temp_output_dir / "3d_annotate_indices_row.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_annotate_indices_row.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_annotate_indices_all(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with all indices annotated matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_indices()
        output_path = temp_output_dir / "3d_annotate_indices_all.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_annotate_indices_all.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_annotate_indices_col_depth(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with col and depth indices annotated matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_indices(1, color="red")
            .annotate_dim_indices(2, color="blue")
        )
        output_path = temp_output_dir / "3d_annotate_indices_col_depth.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_annotate_indices_col_depth.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_annotate_size_indices(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with size and indices annotated matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_size(color="green")
            .annotate_dim_indices(color="gray")
        )
        output_path = temp_output_dir / "3d_annotate_size_indices.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_annotate_size_indices.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_indices_fill_color(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with indices and fill_color matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_indices(color="red")
            .fill_color("blue")
        )
        output_path = temp_output_dir / "2d_annotate_indices_fill_color.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_indices_fill_color.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    # Mixed size/indices annotation combination tests
    def test_reference_2d_row_size_col_indices(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with row size + column indices matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_size(0, color="red")
            .annotate_dim_indices(1, color="blue")
        )
        output_path = temp_output_dir / "2d_row_size_col_indices.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_row_size_col_indices.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_row_indices_col_size(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with row indices + column size matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_indices(0, color="blue")
            .annotate_dim_size(1, color="red")
        )
        output_path = temp_output_dir / "2d_row_indices_col_size.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_row_indices_col_size.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_row_size_col_indices_depth_size(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with row size + col indices + depth size matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_size(0, color="red")
            .annotate_dim_indices(1, color="blue")
            .annotate_dim_size(2, color="green")
        )
        output_path = temp_output_dir / "3d_row_size_col_indices_depth_size.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_row_size_col_indices_depth_size.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_row_indices_col_size_depth_indices(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with row indices + col size + depth indices matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_indices(0, color="blue")
            .annotate_dim_size(1, color="red")
            .annotate_dim_indices(2, color="gray")
        )
        output_path = temp_output_dir / "3d_row_indices_col_size_depth_indices.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_row_indices_col_size_depth_indices.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_row_size_col_size_depth_indices(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with row size + col size + depth indices matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_size(0, color="red")
            .annotate_dim_size(1, color="green")
            .annotate_dim_indices(2, color="blue")
        )
        output_path = temp_output_dir / "3d_row_size_col_size_depth_indices.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_row_size_col_size_depth_indices.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_row_indices_col_indices_depth_size(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with row indices + col indices + depth size matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_indices(0, color="blue")
            .annotate_dim_indices(1, color="gray")
            .annotate_dim_size(2, color="red")
        )
        output_path = temp_output_dir / "3d_row_indices_col_indices_depth_size.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_row_indices_col_indices_depth_size.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_mixed_annotate_fill_region(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with mixed annotations + fill_region matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = (
            td.to_diagram(tensor)
            .annotate_dim_size(0, color="red")
            .annotate_dim_indices(1, color="blue")
            .fill_region(start_coord=(0, 0), end_coord=(2, 2), color="green", opacity=None)
        )
        output_path = temp_output_dir / "2d_mixed_annotate_fill_region.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_mixed_annotate_fill_region.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    # Function-based color/opacity tests
    def test_reference_2d_function_color_value_based(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with function-based color (value-based) matches reference."""
        tensor = np.array([[1, -2, 3, -4], [5, -6, 7, -8], [9, -10, 11, -12]], dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_color(
            lambda idx, val: "red" if val > 0 else "blue"  # type: ignore[arg-type]
        )
        output_path = temp_output_dir / "2d_function_color_value.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_function_color_value.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_function_color_index_based(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with function-based color (index-based) matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).fill_color(
            lambda idx, val: "green" if idx[0] % 2 == 0 else "purple"  # type: ignore[arg-type]
        )
        output_path = temp_output_dir / "2d_function_color_index.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_function_color_index.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_function_opacity_value_based(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with function-based opacity (value-based) matches reference."""
        tensor = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_opacity(
            lambda idx, val: val / 15.0  # type: ignore[arg-type]
        )
        output_path = temp_output_dir / "2d_function_opacity_value.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_function_opacity_value.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_function_opacity_index_based(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with function-based opacity (index-based) matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(
            lambda idx, val: (idx[0] + idx[1]) / 6.0  # type: ignore[arg-type]
        )
        output_path = temp_output_dir / "2d_function_opacity_index.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_function_opacity_index.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_function_both_color_opacity(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with both function-based color and opacity matches reference."""
        tensor = np.array([[1, -2, 3], [4, -5, 6]], dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_region(
            start_coord=(0, 0),
            end_coord=(1, 2),
            color=lambda idx, val: "orange" if val > 0 else "cyan",  # type: ignore[arg-type]
            opacity=lambda idx, val: abs(val) / 10.0  # type: ignore[arg-type]
        )
        output_path = temp_output_dir / "2d_function_both.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_function_both.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_1d_function_color(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 1D tensor with function-based color matches reference."""
        tensor = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_color(
            lambda idx, val: "red" if idx % 2 == 0 else "blue"  # type: ignore[arg-type]
        )
        output_path = temp_output_dir / "1d_function_color.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_1d_function_color.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_function_color(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with function-based color matches reference."""
        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_color(
            lambda idx, val: "red" if idx[2] < 2 else "blue"  # type: ignore[arg-type]
        )
        output_path = temp_output_dir / "3d_function_color.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_function_color.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    # Gradient order visual tests for 2D tensors
    def test_reference_2d_gradient_order_R(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with gradient order R (row-wise) matches reference."""
        from tensordiagram.types import TensorOrder

        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.R)
        output_path = temp_output_dir / "2d_gradient_order_R.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_gradient_order_R.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_gradient_order_C(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with gradient order C (column-wise) matches reference."""
        from tensordiagram.types import TensorOrder

        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.C)
        output_path = temp_output_dir / "2d_gradient_order_C.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_gradient_order_C.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_gradient_order_CR(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with gradient order CR (column-then-row) matches reference."""
        from tensordiagram.types import TensorOrder

        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.CR)
        output_path = temp_output_dir / "2d_gradient_order_CR.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_gradient_order_CR.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    # Gradient order visual tests for 3D tensors
    def test_reference_3d_gradient_order_R(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with gradient order R (row) matches reference."""
        from tensordiagram.types import TensorOrder

        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.R)
        output_path = temp_output_dir / "3d_gradient_order_R.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_gradient_order_R.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_gradient_order_C(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with gradient order C (column) matches reference."""
        from tensordiagram.types import TensorOrder

        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.C)
        output_path = temp_output_dir / "3d_gradient_order_C.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_gradient_order_C.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_gradient_order_D(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with gradient order D (depth) matches reference."""
        from tensordiagram.types import TensorOrder

        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.D)
        output_path = temp_output_dir / "3d_gradient_order_D.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_gradient_order_D.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_gradient_order_RD(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with gradient order RD (row-then-depth) matches reference."""
        from tensordiagram.types import TensorOrder

        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.RD)
        output_path = temp_output_dir / "3d_gradient_order_RD.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_gradient_order_RD.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_gradient_order_RDC(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with gradient order RDC (row-depth-column) matches reference."""
        from tensordiagram.types import TensorOrder

        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.RDC)
        output_path = temp_output_dir / "3d_gradient_order_RDC.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_gradient_order_RDC.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_3d_gradient_order_DCR(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 3D tensor with gradient order DCR (depth-column-row) matches reference."""
        from tensordiagram.types import TensorOrder

        tensor = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        diagram = td.to_diagram(tensor).fill_opacity(0.2, 0.9, order=TensorOrder.DCR)
        output_path = temp_output_dir / "3d_gradient_order_DCR.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_3d_gradient_order_DCR.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    # fill_values with custom formatting tests
    def test_reference_2d_fill_values_custom_font_size(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with fill_values and custom font size matches reference."""
        tensor = np.array([[1.5, 2.3], [3.7, 4.1]], dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_values(font_size=0.8)
        output_path = temp_output_dir / "2d_fill_values_custom_font_size.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_fill_values_custom_font_size.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_fill_values_percentage_format(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with fill_values and percentage formatting matches reference."""
        tensor = np.array([[0.123, 0.456], [0.789, 0.234]], dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_values(format_fn=lambda x: f"{x*100:.1f}%")
        output_path = temp_output_dir / "2d_fill_values_percentage.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_fill_values_percentage.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_fill_values_scientific_notation(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with fill_values and scientific notation matches reference."""
        tensor = np.array([[1000, 2000], [3000, 4000]], dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_values(format_fn=lambda x: f"{x:.1e}")
        output_path = temp_output_dir / "2d_fill_values_scientific.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_fill_values_scientific.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_fill_values_integer_format(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with fill_values and integer formatting matches reference."""
        tensor = np.array([[1.5, 2.7], [3.2, 4.9]], dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_values(format_fn=lambda x: str(int(x)))
        output_path = temp_output_dir / "2d_fill_values_integer.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_fill_values_integer.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_fill_values_custom_size_and_format(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with fill_values using both custom font size and format matches reference."""
        tensor = np.array([[1.234, 5.678], [9.012, 3.456]], dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_values(font_size=0.5, format_fn=lambda x: f"{x:.1f}")
        output_path = temp_output_dir / "2d_fill_values_size_and_format.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_fill_values_size_and_format.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_1d_fill_values_custom_font_size(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 1D tensor with fill_values and custom font size matches reference."""
        tensor = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_values(font_size=1.0)
        output_path = temp_output_dir / "1d_fill_values_custom_font_size.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_1d_fill_values_custom_font_size.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_1d_fill_values_custom_format(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 1D tensor with fill_values and custom formatting matches reference."""
        tensor = np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float32)
        diagram = td.to_diagram(tensor).fill_values(format_fn=lambda x: f"{x:.0f}")
        output_path = temp_output_dir / "1d_fill_values_custom_format.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_1d_fill_values_custom_format.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    # annotate_dim_indices with custom font size tests
    def test_reference_1d_annotate_indices_custom_font_size(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 1D tensor with indices annotated (custom font size) matches reference."""
        tensor = np.arange(8, dtype=np.float32)
        diagram = td.to_diagram(tensor).annotate_dim_indices(font_size=0.5)
        output_path = temp_output_dir / "1d_annotate_indices_custom_font_size.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_1d_annotate_indices_custom_font_size.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_indices_custom_font_size(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with indices annotated (custom font size) matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_indices(font_size=0.6)
        output_path = temp_output_dir / "2d_annotate_indices_custom_font_size.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_indices_custom_font_size.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_indices_large_font_size(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with indices annotated (large font size) matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_indices(font_size=1.2)
        output_path = temp_output_dir / "2d_annotate_indices_large_font_size.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_indices_large_font_size.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")

    def test_reference_2d_annotate_indices_small_font_size(
        self, fixtures_dir, temp_output_dir, svg_comparator
    ):
        """Test 2D tensor with indices annotated (small font size) matches reference."""
        tensor = np.arange(12, dtype=np.float32).reshape(3, 4)
        diagram = td.to_diagram(tensor).annotate_dim_indices(font_size=0.3)
        output_path = temp_output_dir / "2d_annotate_indices_small_font_size.svg"
        diagram.render_svg(str(output_path), height=128)

        reference_path = fixtures_dir / "reference_2d_annotate_indices_small_font_size.svg"
        if reference_path.exists():
            assert svg_comparator(output_path, reference_path), \
                "Rendered output does not match reference image"
        else:
            pytest.skip(f"Reference image not found: {reference_path}")


@pytest.mark.rendering
class TestRenderingDimensions:
    """Tests for rendering with different dimensions."""

    def test_default_height(self, shape_2d, temp_output_dir):
        """Test that default height is applied."""
        diagram = td.to_diagram(shape_2d)
        output_path = temp_output_dir / "default_height.svg"
        diagram.render_svg(str(output_path))

        assert output_path.exists()
        # Verify SVG contains expected dimensions
        content = output_path.read_text()
        assert "height" in content

    def test_multiple_sizes(self, shape_2d, temp_output_dir):
        """Test rendering at multiple sizes."""
        diagram = td.to_diagram(shape_2d)

        sizes = [64, 128, 256, 512]
        for size in sizes:
            output_path = temp_output_dir / f"size_{size}.svg"
            diagram.render_svg(str(output_path), height=size)
            assert output_path.exists()


@pytest.mark.rendering
@pytest.mark.skip(reason="Edge cases not needed for regular testing")
class TestRenderingEdgeCases:
    """Tests for rendering edge cases."""

    def test_render_large_tensor(self, temp_output_dir):
        """Test rendering a larger tensor."""
        tensor = np.random.randn(10, 10)
        diagram = td.to_diagram(tensor).fill_values()

        output_path = temp_output_dir / "large_tensor.svg"
        diagram.render_svg(str(output_path))
        assert output_path.exists()

    def test_render_with_scientific_notation(self, temp_output_dir):
        """Test rendering values that might use scientific notation."""
        tensor = np.array([[0.00001, 1000000.0]])
        diagram = td.to_diagram(tensor).fill_values()

        output_path = temp_output_dir / "scientific.svg"
        diagram.render_svg(str(output_path))
        assert output_path.exists()

    def test_render_to_nested_directory(self, temp_output_dir):
        """Test rendering to a nested directory path."""
        diagram = td.to_diagram((2, 2))

        nested_dir = temp_output_dir / "nested" / "path"
        nested_dir.mkdir(parents=True, exist_ok=True)

        output_path = nested_dir / "nested_output.svg"
        diagram.render_svg(str(output_path))
        assert output_path.exists()

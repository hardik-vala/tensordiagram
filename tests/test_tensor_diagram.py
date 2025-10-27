"""Unit tests for TensorDiagram core functionality."""

import numpy as np
import pytest
from tensordiagram.core import to_diagram


class TestToDiagram:
    """Tests for the to_diagram function (main public API)."""

    # Tests with shape tuples
    def test_to_diagram_with_shape_1d(self, shape_1d):
        """Test creating diagram from 1D shape."""
        diagram = to_diagram(shape_1d)
        assert diagram.rank == 1

    def test_to_diagram_with_shape_2d(self, shape_2d):
        """Test creating diagram from 2D shape."""
        diagram = to_diagram(shape_2d)
        assert diagram.rank == 2

    def test_to_diagram_with_shape_2d_square(self, shape_2d_square):
        """Test creating diagram from square 2D shape."""
        diagram = to_diagram(shape_2d_square)
        assert diagram.rank == 2

    def test_to_diagram_with_shape_3d(self, shape_3d):
        """Test creating diagram from 3D shape."""
        diagram = to_diagram(shape_3d)
        assert diagram.rank == 3
        assert diagram.tensor_shape == (2, 3, 4)

    def test_to_diagram_with_shape_invalid_empty(self):
        """Test that empty shape raises ValueError."""
        with pytest.raises(ValueError, match="at least 1 element"):
            to_diagram(())

    def test_to_diagram_with_shape_invalid_too_many_dims(self):
        """Test that >3D shape raises ValueError."""
        with pytest.raises(ValueError, match="at most 3-dimensional"):
            to_diagram((2, 3, 4, 5))

    # Tests with PyTorch tensors
    @pytest.mark.torch
    def test_to_diagram_with_torch_tensor_1d(self, torch_tensor_1d):
        """Test creating diagram from 1D PyTorch tensor."""
        diagram = to_diagram(torch_tensor_1d)
        assert diagram.rank == 1

    @pytest.mark.torch
    def test_to_diagram_with_torch_tensor_2d(self, torch_tensor_2d):
        """Test creating diagram from 2D PyTorch tensor."""
        diagram = to_diagram(torch_tensor_2d)
        assert diagram.rank == 2

    @pytest.mark.torch
    def test_to_diagram_with_torch_tensor_3d(self, torch_tensor_3d):
        """Test creating diagram from 3D PyTorch tensor."""
        diagram = to_diagram(torch_tensor_3d)
        assert diagram.rank == 3
        assert diagram.tensor_shape == (2, 3, 4)

    @pytest.mark.torch
    def test_to_diagram_with_torch_tensor_invalid_too_many_dims(self):
        """Test that >3D tensor raises ValueError."""
        import torch

        tensor = torch.zeros(2, 3, 4, 5)
        with pytest.raises(ValueError, match="at most 3-dimensional"):
            to_diagram(tensor)

    # Tests with NumPy arrays
    def test_to_diagram_with_numpy_array_1d(self, numpy_array_1d):
        """Test creating diagram from 1D NumPy array."""
        diagram = to_diagram(numpy_array_1d)
        assert diagram.rank == 1

    def test_to_diagram_with_numpy_array_2d(self, numpy_array_2d):
        """Test creating diagram from 2D NumPy array."""
        diagram = to_diagram(numpy_array_2d)
        assert diagram.rank == 2

    def test_to_diagram_with_numpy_array_3d(self, numpy_array_3d):
        """Test creating diagram from 3D NumPy array."""
        diagram = to_diagram(numpy_array_3d)
        assert diagram.rank == 3
        assert diagram.tensor_shape == (2, 3, 4)

    # Edge cases
    def test_to_diagram_single_element_tensor(self):
        """Test diagram creation from single element array."""
        tensor = np.array([42.0])
        diagram = to_diagram(tensor)
        assert diagram.rank == 1

    def test_to_diagram_with_large_values(self):
        """Test that large values are handled correctly."""
        tensor = np.array([[1000.5, 2000.75]])
        diagram = to_diagram(tensor).fill_values()
        assert diagram.rank == 2

    def test_to_diagram_with_negative_values(self):
        """Test array with negative values."""
        tensor = np.array([[-1.0, -2.5], [3.0, -4.5]])
        diagram = to_diagram(tensor).fill_values()
        assert diagram.rank == 2

    def test_to_diagram_with_zero_values(self):
        """Test array with zero values."""
        tensor = np.zeros((2, 3))
        diagram = to_diagram(tensor).fill_values()
        assert diagram.rank == 2

    def test_to_diagram_with_integer_values(self):
        """Test array with integer values."""
        tensor = np.array([[1, 2], [3, 4]])
        diagram = to_diagram(tensor).fill_values()
        assert diagram.rank == 2


class TestTensorDiagramProperties:
    """Tests for TensorDiagram properties."""

    def test_tensor_shape_1d(self, shape_1d):
        """Test tensor_shape property for 1D tensor."""
        diagram = to_diagram(shape_1d)
        assert diagram.tensor_shape == shape_1d
        assert diagram.tensor_shape == (5,)

    def test_tensor_shape_2d(self, shape_2d):
        """Test tensor_shape property for 2D tensor."""
        diagram = to_diagram(shape_2d)
        assert diagram.tensor_shape == shape_2d
        assert diagram.tensor_shape == (3, 4)

    def test_tensor_shape_from_numpy(self, numpy_array_2d):
        """Test tensor_shape property from NumPy array."""
        diagram = to_diagram(numpy_array_2d)
        assert diagram.tensor_shape == (3, 4)

    def test_rank_1d(self, shape_1d):
        """Test rank property for 1D tensor."""
        diagram = to_diagram(shape_1d)
        assert diagram.rank == 1

    def test_rank_2d(self, shape_2d):
        """Test rank property for 2D tensor."""
        diagram = to_diagram(shape_2d)
        assert diagram.rank == 2

    def test_rank_from_numpy(self, numpy_array_2d_square):
        """Test rank property from NumPy array."""
        diagram = to_diagram(numpy_array_2d_square)
        assert diagram.rank == 2


class TestFillValues:
    """Tests for the fill_values method."""

    def test_fill_values_with_tensor(self, numpy_array_2d_square):
        """Test fill_values with array data."""
        diagram = to_diagram(numpy_array_2d_square)
        diagram_with_values = diagram.fill_values()
        assert diagram_with_values.rank == diagram.rank

    def test_fill_values_without_tensor(self, shape_2d):
        """Test fill_values when created from shape."""
        diagram = to_diagram(shape_2d)
        diagram_with_values = diagram.fill_values()
        assert diagram_with_values.rank == diagram.rank

    def test_fill_values_immutability(self, numpy_array_2d_square):
        """Test that fill_values returns new instance."""
        diagram = to_diagram(numpy_array_2d_square)
        diagram_with_values = diagram.fill_values()
        # Both should still be valid
        assert diagram.rank == 2
        assert diagram_with_values.rank == 2


class TestStyling:
    """Tests for TensorDiagram styling methods."""

    def test_fill_color_basic(self, shape_2d):
        """Test fill_color() applies color to entire tensor."""
        diagram = to_diagram(shape_2d)
        colored_diagram = diagram.fill_color("blue")
        assert colored_diagram.rank == diagram.rank
        assert colored_diagram.tensor_shape == diagram.tensor_shape

    def test_fill_color_with_numpy(self, numpy_array_2d):
        """Test fill_color() with NumPy array."""
        diagram = to_diagram(numpy_array_2d)
        colored_diagram = diagram.fill_color("red")
        assert colored_diagram.rank == 2

    def test_fill_opacity_float(self, shape_2d):
        """Test fill_opacity() with single float value."""
        diagram = to_diagram(shape_2d)
        opaque_diagram = diagram.fill_opacity(0.5)
        assert opaque_diagram.rank == diagram.rank
        assert opaque_diagram.tensor_shape == diagram.tensor_shape

    def test_fill_opacity_gradient(self, shape_2d):
        """Test fill_opacity() with gradient tuple."""
        diagram = to_diagram(shape_2d)
        gradient_diagram = diagram.fill_opacity(0.2, 0.8)
        assert gradient_diagram.rank == diagram.rank
        assert gradient_diagram.tensor_shape == diagram.tensor_shape

    def test_fill_region_1d_color(self, shape_1d):
        """Test fill_region() with color on 1D tensor."""
        diagram = to_diagram(shape_1d)
        filled_diagram = diagram.fill_region(
            start_coord=1, end_coord=4, color="green", opacity=None
        )
        assert filled_diagram.rank == 1
        assert filled_diagram.tensor_shape == shape_1d

    def test_fill_region_2d_color(self, shape_2d):
        """Test fill_region() with color on 2D tensor."""
        diagram = to_diagram(shape_2d)
        filled_diagram = diagram.fill_region(
            start_coord=(0, 0), end_coord=(2, 2), color="yellow", opacity=None
        )
        assert filled_diagram.rank == 2
        assert filled_diagram.tensor_shape == shape_2d

    def test_fill_region_2d_opacity(self, shape_2d):
        """Test fill_region() with opacity on 2D tensor."""
        diagram = to_diagram(shape_2d)
        filled_diagram = diagram.fill_region(
            start_coord=(0, 0), end_coord=(2, 3), color=None, opacity=0.7
        )
        assert filled_diagram.rank == 2
        assert filled_diagram.tensor_shape == shape_2d

    def test_fill_region_both_params(self, shape_2d):
        """Test fill_region() with both color and opacity."""
        diagram = to_diagram(shape_2d)
        filled_diagram = diagram.fill_region(
            start_coord=(1, 1), end_coord=(3, 3), color="purple", opacity=0.6
        )
        assert filled_diagram.rank == 2
        assert filled_diagram.tensor_shape == shape_2d

    def test_fill_region_invalid_no_params(self, shape_2d):
        """Test fill_region() raises error when neither color nor opacity provided."""
        diagram = to_diagram(shape_2d)
        with pytest.raises(ValueError, match="At least one of color or opacity"):
            diagram.fill_region(
                start_coord=(0, 0), end_coord=(2, 2), color=None, opacity=None
            )

    def test_fill_region_with_gradient_opacity(self, numpy_array_2d_square):
        """Test fill_region() with opacity gradient on NumPy array."""
        diagram = to_diagram(numpy_array_2d_square)
        filled_diagram = diagram.fill_region(
            start_coord=(0, 0), end_coord=(2, 2), color=None, opacity=(0.1, 0.9)
        )
        assert filled_diagram.rank == 2

    def test_fill_region_1d_invalid_coords(self, shape_1d):
        """Test fill_region() with invalid coordinate types for 1D tensor."""
        diagram = to_diagram(shape_1d)
        with pytest.raises(ValueError, match="must be integers"):
            diagram.fill_region(
                start_coord=(0, 0), end_coord=(2, 2), color="red", opacity=None
            )

    def test_fill_region_2d_invalid_coords(self, shape_2d):
        """Test fill_region() with invalid coordinate types for 2D tensor."""
        diagram = to_diagram(shape_2d)
        with pytest.raises(ValueError, match="must be tuples"):
            diagram.fill_region(start_coord=0, end_coord=2, color="red", opacity=None)

    def test_fill_color_3d(self, shape_3d):
        """Test fill_color() on 3D tensor."""
        diagram = to_diagram(shape_3d)
        colored_diagram = diagram.fill_color("blue")
        assert colored_diagram.rank == 3
        assert colored_diagram.tensor_shape == shape_3d

    def test_fill_opacity_3d_float(self, shape_3d):
        """Test fill_opacity() with single value on 3D tensor."""
        diagram = to_diagram(shape_3d)
        opaque_diagram = diagram.fill_opacity(0.5)
        assert opaque_diagram.rank == 3
        assert opaque_diagram.tensor_shape == shape_3d

    def test_fill_opacity_3d_gradient(self, shape_3d):
        """Test fill_opacity() with gradient on 3D tensor."""
        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.2, 0.9)
        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_fill_region_3d_color(self, shape_3d):
        """Test fill_region() with color on 3D tensor."""
        diagram = to_diagram(shape_3d)
        filled_diagram = diagram.fill_region(
            start_coord=(0, 0, 0), end_coord=(1, 2, 3), color="red", opacity=None
        )
        assert filled_diagram.rank == 3
        assert filled_diagram.tensor_shape == shape_3d

    def test_fill_region_3d_opacity(self, shape_3d):
        """Test fill_region() with opacity on 3D tensor."""
        diagram = to_diagram(shape_3d)
        filled_diagram = diagram.fill_region(
            start_coord=(0, 0, 0), end_coord=(1, 1, 2), color=None, opacity=0.7
        )
        assert filled_diagram.rank == 3
        assert filled_diagram.tensor_shape == shape_3d

    def test_fill_region_3d_both_params(self, shape_3d):
        """Test fill_region() with both color and opacity on 3D tensor."""
        diagram = to_diagram(shape_3d)
        filled_diagram = diagram.fill_region(
            start_coord=(0, 1, 0), end_coord=(1, 2, 3), color="green", opacity=0.8
        )
        assert filled_diagram.rank == 3
        assert filled_diagram.tensor_shape == shape_3d

    def test_fill_region_3d_gradient_opacity(self, shape_3d):
        """Test fill_region() with opacity gradient on 3D tensor."""
        diagram = to_diagram(shape_3d)
        filled_diagram = diagram.fill_region(
            start_coord=(0, 0, 0), end_coord=(1, 2, 3), color=None, opacity=(0.1, 0.9)
        )
        assert filled_diagram.rank == 3
        assert filled_diagram.tensor_shape == shape_3d

    def test_fill_region_3d_invalid_coords(self, shape_3d):
        """Test fill_region() with invalid coordinate types for 3D tensor."""
        diagram = to_diagram(shape_3d)
        with pytest.raises(ValueError, match="must be tuples"):
            diagram.fill_region(
                start_coord=(0, 0), end_coord=(1, 1), color="red", opacity=None
            )

    def test_fill_values_3d(self, numpy_array_3d):
        """Test fill_values() on 3D tensor raises NotImplementedError."""
        diagram = to_diagram(numpy_array_3d)
        with pytest.raises(NotImplementedError, match="Showing values for 3D tensors is not supported"):
            diagram.fill_values()

    # Function-based color tests
    def test_fill_color_with_function_2d(self):
        """Test fill_color with function on 2D tensor."""
        tensor = np.array([[1, 2], [3, 4]])
        diagram = to_diagram(tensor)
        colored = diagram.fill_color(lambda idx, val: "red" if val > 2 else "blue") # type: ignore[arg-type]
        assert colored.rank == 2
        assert colored.tensor_shape == (2, 2)

    def test_fill_color_with_function_1d(self):
        """Test fill_color with function on 1D tensor."""
        tensor = np.array([1, 2, 3, 4, 5])
        diagram = to_diagram(tensor)
        colored = diagram.fill_color(lambda idx, val: "red" if idx % 2 == 0 else "blue") # type: ignore[arg-type]
        assert colored.rank == 1
        assert colored.tensor_shape == (5,)

    def test_fill_color_with_function_3d(self):
        """Test fill_color with function on 3D tensor."""
        tensor = np.arange(24).reshape(2, 3, 4)
        diagram = to_diagram(tensor)
        colored = diagram.fill_color(lambda idx, val: "red" if idx[0] == 0 else "blue") # type: ignore[arg-type]
        assert colored.rank == 3
        assert colored.tensor_shape == (2, 3, 4)

    def test_fill_region_with_color_function_2d(self):
        """Test fill_region with color function on 2D tensor."""
        tensor = np.array([[1, -2, 3], [-4, 5, -6]])
        diagram = to_diagram(tensor)
        filled = diagram.fill_region(
            start_coord=(0, 0),
            end_coord=(1, 2),
            color=lambda idx, val: "green" if val > 0 else "yellow", # type: ignore[arg-type]
            opacity=None
        )
        assert filled.rank == 2
        assert filled.tensor_shape == (2, 3)

    # Function-based opacity tests
    def test_fill_opacity_with_function_2d(self):
        """Test fill_opacity with function on 2D tensor."""
        tensor = np.array([[1, 2, 3], [4, 5, 6]])
        diagram = to_diagram(tensor)
        opaque = diagram.fill_opacity(lambda idx, val: val / 10.0) # type: ignore[arg-type]
        assert opaque.rank == 2
        assert opaque.tensor_shape == (2, 3)

    def test_fill_opacity_with_function_1d(self):
        """Test fill_opacity with function on 1D tensor."""
        tensor = np.array([1, 2, 3, 4, 5])
        diagram = to_diagram(tensor)
        opaque = diagram.fill_opacity(lambda idx, val: idx / 4.0) # type: ignore[arg-type]
        assert opaque.rank == 1
        assert opaque.tensor_shape == (5,)

    def test_fill_region_with_opacity_function_2d(self):
        """Test fill_region with opacity function on 2D tensor."""
        tensor = np.array([[10, 20, 30], [40, 50, 60]])
        diagram = to_diagram(tensor)
        filled = diagram.fill_region(
            start_coord=(0, 0),
            end_coord=(1, 2),
            color=None,
            opacity=lambda idx, val: val / 100.0 # type: ignore[arg-type]
        )
        assert filled.rank == 2
        assert filled.tensor_shape == (2, 3)

    def test_fill_opacity_function_with_gradient_raises_error(self):
        """Test that using function with gradient parameters raises error."""
        tensor = np.array([[1, 2], [3, 4]])
        diagram = to_diagram(tensor)
        with pytest.raises(ValueError, match="mutually exclusive"):
            diagram.fill_opacity(lambda idx, val: 0.5, end=0.9)

    # Combined function-based tests
    def test_fill_region_with_both_color_and_opacity_functions(self):
        """Test fill_region with both color and opacity functions."""
        tensor = np.array([[-5, 3], [2, -1]])
        diagram = to_diagram(tensor)
        filled = diagram.fill_region(
            start_coord=(0, 0),
            end_coord=(1, 1),
            color=lambda idx, val: "red" if val > 0 else "blue", # type: ignore[arg-type]
            opacity=lambda idx, val: abs(val) / 10.0 # type: ignore[arg-type]
        )
        assert filled.rank == 2
        assert filled.tensor_shape == (2, 2)

    def test_mixing_static_and_function_values(self):
        """Test using static color with function opacity."""
        tensor = np.array([[1, 2, 3], [4, 5, 6]])
        diagram = to_diagram(tensor)
        styled = diagram.fill_region(
            start_coord=(0, 0),
            end_coord=(1, 2),
            color="green",
            opacity=lambda idx, val: val / 10.0 # type: ignore[arg-type]
        )
        assert styled.rank == 2

    def test_function_receives_correct_index_and_value_1d(self):
        """Test that function receives correct index and value for 1D."""
        tensor = np.array([10, 20, 30])
        diagram = to_diagram(tensor)
        received_calls = []

        def color_fn(idx, val):
            received_calls.append((idx, val))
            return "red"

        diagram.fill_color(color_fn)
        assert len(received_calls) == 3
        assert (0, 10) in received_calls
        assert (1, 20) in received_calls
        assert (2, 30) in received_calls

    def test_function_receives_correct_index_and_value_2d(self):
        """Test that function receives correct index and value for 2D."""
        tensor = np.array([[1, 2], [3, 4]])
        diagram = to_diagram(tensor)
        received_calls = []

        def opacity_fn(idx, val):
            received_calls.append((idx, val))
            return 0.5

        diagram.fill_opacity(opacity_fn)
        assert len(received_calls) == 4
        assert ((0, 0), 1) in received_calls
        assert ((0, 1), 2) in received_calls
        assert ((1, 0), 3) in received_calls
        assert ((1, 1), 4) in received_calls

    def test_fill_color_function_immutability(self):
        """Test that fill_color with function returns new instance."""
        tensor = np.array([[1, 2], [3, 4]])
        original = to_diagram(tensor)
        modified = original.fill_color(lambda idx, val: "red")
        assert original is not modified
        assert original.rank == modified.rank

    def test_fill_opacity_function_immutability(self):
        """Test that fill_opacity with function returns new instance."""
        tensor = np.array([[1, 2], [3, 4]])
        original = to_diagram(tensor)
        modified = original.fill_opacity(lambda idx, val: 0.5)
        assert original is not modified
        assert original.rank == modified.rank


class TestMethodChaining:
    """Tests for method chaining and immutability."""

    def test_chaining_immutability(self, shape_2d):
        """Test that styling methods return new instances without modifying original."""
        original = to_diagram(shape_2d)
        modified = original.fill_color("blue")

        # Both should be valid TensorDiagram instances
        assert original.rank == 2
        assert modified.rank == 2
        assert original.tensor_shape == modified.tensor_shape

        # They should be different instances
        assert original is not modified

    def test_chain_multiple_operations(self, numpy_array_2d_square):
        """Test chaining multiple styling operations."""
        diagram = to_diagram(numpy_array_2d_square)
        styled_diagram = (
            diagram.fill_color("red").fill_opacity(0.5).fill_values()
        )

        # Verify the final diagram is valid
        assert styled_diagram.rank == 2
        assert styled_diagram.tensor_shape == (4, 4)

        # Original should be unchanged
        assert diagram.rank == 2

    def test_fill_region_returns_new_instance(self, shape_2d):
        """Test that fill_region() returns a new instance."""
        original = to_diagram(shape_2d)
        modified = original.fill_region(
            start_coord=(0, 0), end_coord=(2, 2), color="green", opacity=None
        )

        assert original is not modified
        assert original.rank == modified.rank
        assert original.tensor_shape == modified.tensor_shape

    def test_chain_fill_region_and_values(self, numpy_array_2d):
        """Test chaining fill_region with fill_values."""
        diagram = to_diagram(numpy_array_2d)
        styled_diagram = diagram.fill_region(
            start_coord=(0, 0), end_coord=(2, 2), color="blue", opacity=0.7
        ).fill_values()

        assert styled_diagram.rank == 2
        assert styled_diagram.tensor_shape == (3, 4)

    def test_multiple_fill_region_calls(self, shape_2d_square):
        """Test chaining multiple fill_region calls."""
        diagram = to_diagram(shape_2d_square)
        styled_diagram = (
            diagram.fill_region(
                start_coord=(0, 0), end_coord=(2, 2), color="red", opacity=None
            )
            .fill_region(
                start_coord=(2, 2), end_coord=(4, 4), color="blue", opacity=None
            )
        )

        assert styled_diagram.rank == 2
        assert styled_diagram.tensor_shape == (4, 4)

    def test_immutability_with_fill_opacity(self, numpy_array_2d):
        """Test that fill_opacity preserves original diagram."""
        original = to_diagram(numpy_array_2d)
        with_opacity = original.fill_opacity(0.3)
        with_gradient = original.fill_opacity(0.1, 0.9)

        # All should be separate instances
        assert original is not with_opacity
        assert original is not with_gradient
        assert with_opacity is not with_gradient

        # All should have same shape and rank
        assert original.rank == with_opacity.rank == with_gradient.rank == 2
        assert (
            original.tensor_shape
            == with_opacity.tensor_shape
            == with_gradient.tensor_shape
        )

    def test_chain_multiple_operations_3d(self, numpy_array_3d):
        """Test chaining multiple styling operations on 3D tensor raises NotImplementedError for fill_values."""
        diagram = to_diagram(numpy_array_3d)
        with pytest.raises(NotImplementedError, match="Showing values for 3D tensors is not supported"):
            styled_diagram = (
                diagram.fill_color("purple")
                .fill_opacity(0.6)
                .fill_region(
                    start_coord=(0, 0, 0), end_coord=(1, 1, 1), color="yellow", opacity=None
                )
                .fill_values()
            )

    def test_multiple_fill_region_calls_3d(self, shape_3d):
        """Test chaining multiple fill_region calls on 3D tensor."""
        diagram = to_diagram(shape_3d)
        styled_diagram = (
            diagram.fill_region(
                start_coord=(0, 0, 0), end_coord=(0, 1, 1), color="red", opacity=None
            ).fill_region(
                start_coord=(1, 1, 2), end_coord=(1, 2, 3), color="blue", opacity=None
            )
        )
        assert styled_diagram.rank == 3
        assert styled_diagram.tensor_shape == shape_3d

    def test_immutability_3d(self, numpy_array_3d):
        """Test that styling methods return new instances for 3D tensors."""
        original = to_diagram(numpy_array_3d)
        colored = original.fill_color("red")
        with_opacity = original.fill_opacity(0.5)
        with_region = original.fill_region(
            start_coord=(0, 0, 0), end_coord=(1, 1, 1), color="blue", opacity=None
        )

        # All should be separate instances
        assert original is not colored
        assert original is not with_opacity
        assert original is not with_region
        assert colored is not with_opacity
        assert colored is not with_region
        assert with_opacity is not with_region

        # All should have same shape and rank
        assert (
            original.rank
            == colored.rank
            == with_opacity.rank
            == with_region.rank
            == 3
        )
        assert (
            original.tensor_shape
            == colored.tensor_shape
            == with_opacity.tensor_shape
            == with_region.tensor_shape
        )


class TestAnnotateDimSize:
    """Tests for annotate_dim_size method."""

    def test_annotate_dim_size_1d_row(self, shape_1d):
        """Test annotating row dimension on 1D tensor."""
        diagram = to_diagram(shape_1d)
        annotated = diagram.annotate_dim_size("row")
        assert annotated.rank == 1
        assert annotated.tensor_shape == shape_1d

    def test_annotate_dim_size_1d_all(self, shape_1d):
        """Test annotating all dimensions on 1D tensor."""
        diagram = to_diagram(shape_1d)
        annotated = diagram.annotate_dim_size("all")
        assert annotated.rank == 1
        assert annotated.tensor_shape == shape_1d

    def test_annotate_dim_size_1d_with_color(self, shape_1d):
        """Test annotating 1D tensor with custom color."""
        diagram = to_diagram(shape_1d)
        annotated = diagram.annotate_dim_size("row", color="red")
        assert annotated.rank == 1
        assert annotated.tensor_shape == shape_1d

    def test_annotate_dim_size_1d_invalid_col(self, shape_1d):
        """Test that 1D tensor rejects column dimension."""
        diagram = to_diagram(shape_1d)
        with pytest.raises(ValueError, match="1D tensors can only annotate 'row' dimension"):
            diagram.annotate_dim_size("col")

    def test_annotate_dim_size_1d_invalid_depth(self, shape_1d):
        """Test that 1D tensor rejects depth dimension."""
        diagram = to_diagram(shape_1d)
        with pytest.raises(ValueError, match="1D tensors can only annotate 'row' dimension"):
            diagram.annotate_dim_size("depth")

    def test_annotate_dim_size_2d_row(self, shape_2d):
        """Test annotating row dimension on 2D tensor."""
        diagram = to_diagram(shape_2d)
        annotated = diagram.annotate_dim_size("row")
        assert annotated.rank == 2
        assert annotated.tensor_shape == shape_2d

    def test_annotate_dim_size_2d_col(self, shape_2d):
        """Test annotating column dimension on 2D tensor."""
        diagram = to_diagram(shape_2d)
        annotated = diagram.annotate_dim_size("col")
        assert annotated.rank == 2
        assert annotated.tensor_shape == shape_2d

    def test_annotate_dim_size_2d_all(self, shape_2d):
        """Test annotating all dimensions on 2D tensor."""
        diagram = to_diagram(shape_2d)
        annotated = diagram.annotate_dim_size("all")
        assert annotated.rank == 2
        assert annotated.tensor_shape == shape_2d

    def test_annotate_dim_size_2d_with_color(self, shape_2d):
        """Test annotating 2D tensor with custom color."""
        diagram = to_diagram(shape_2d)
        annotated = diagram.annotate_dim_size("all", color="green")
        assert annotated.rank == 2
        assert annotated.tensor_shape == shape_2d

    def test_annotate_dim_size_2d_invalid_depth(self, shape_2d):
        """Test that 2D tensor rejects depth dimension."""
        diagram = to_diagram(shape_2d)
        with pytest.raises(ValueError, match="2D tensors cannot annotate 'depth' dimension"):
            diagram.annotate_dim_size("depth")

    def test_annotate_dim_size_3d_row(self, shape_3d):
        """Test annotating row dimension on 3D tensor."""
        diagram = to_diagram(shape_3d)
        annotated = diagram.annotate_dim_size("row")
        assert annotated.rank == 3
        assert annotated.tensor_shape == shape_3d

    def test_annotate_dim_size_3d_col(self, shape_3d):
        """Test annotating column dimension on 3D tensor."""
        diagram = to_diagram(shape_3d)
        annotated = diagram.annotate_dim_size("col")
        assert annotated.rank == 3
        assert annotated.tensor_shape == shape_3d

    def test_annotate_dim_size_3d_depth(self, shape_3d):
        """Test annotating depth dimension on 3D tensor."""
        diagram = to_diagram(shape_3d)
        annotated = diagram.annotate_dim_size("depth")
        assert annotated.rank == 3
        assert annotated.tensor_shape == shape_3d

    def test_annotate_dim_size_3d_all(self, shape_3d):
        """Test annotating all dimensions on 3D tensor."""
        diagram = to_diagram(shape_3d)
        annotated = diagram.annotate_dim_size("all")
        assert annotated.rank == 3
        assert annotated.tensor_shape == shape_3d

    def test_annotate_dim_size_3d_with_color(self, shape_3d):
        """Test annotating 3D tensor with custom color."""
        diagram = to_diagram(shape_3d)
        annotated = diagram.annotate_dim_size("all", color="blue")
        assert annotated.rank == 3
        assert annotated.tensor_shape == shape_3d

    def test_annotate_dim_size_immutability(self, numpy_array_2d):
        """Test that annotate_dim_size returns new instance."""
        original = to_diagram(numpy_array_2d)
        annotated = original.annotate_dim_size("all")

        # Should be different instances
        assert original is not annotated

        # Both should still be valid
        assert original.rank == 2
        assert annotated.rank == 2
        assert original.tensor_shape == annotated.tensor_shape

    def test_annotate_dim_size_chain_with_fill_color(self, numpy_array_2d):
        """Test chaining annotate_dim_size with fill_color."""
        diagram = to_diagram(numpy_array_2d)
        styled = diagram.annotate_dim_size("all", color="red").fill_color("blue")

        assert styled.rank == 2
        assert styled.tensor_shape == (3, 4)

    def test_annotate_dim_size_chain_with_fill_region(self, numpy_array_2d):
        """Test chaining annotate_dim_size with fill_region."""
        diagram = to_diagram(numpy_array_2d)
        styled = diagram.annotate_dim_size("all").fill_region(
            start_coord=(0, 0), end_coord=(2, 2), color="green", opacity=None
        )

        assert styled.rank == 2
        assert styled.tensor_shape == (3, 4)

    def test_annotate_dim_size_multiple_calls(self, numpy_array_2d):
        """Test multiple annotate_dim_size calls."""
        diagram = to_diagram(numpy_array_2d)
        # First call annotates row
        annotated_row = diagram.annotate_dim_size("row", color="red")
        # Second call annotates col (should preserve row annotation)
        annotated_both = annotated_row.annotate_dim_size("col", color="blue")

        assert annotated_both.rank == 2
        assert annotated_both.tensor_shape == (3, 4)

    def test_annotate_dim_size_persists_through_styling(self, shape_2d):
        """Test that annotations persist when applying styling methods."""
        diagram = to_diagram(shape_2d)
        styled = (
            diagram
            .annotate_dim_size("all", color="green")
            .fill_color("red")
            .fill_opacity(0.7)
        )

        assert styled.rank == 2
        assert styled.tensor_shape == shape_2d

    def test_annotate_dim_size_3d_multiple_dimensions(self, numpy_array_3d):
        """Test annotating multiple individual dimensions on 3D tensor."""
        diagram = to_diagram(numpy_array_3d)
        annotated = (
            diagram
            .annotate_dim_size("col", color="red")
            .annotate_dim_size("depth", color="blue")
        )

        assert annotated.rank == 3
        assert annotated.tensor_shape == (2, 3, 4)


class TestAnnotateDimIndices:
    """Tests for annotate_dim_indices method."""

    def test_annotate_dim_indices_1d_row(self, shape_1d):
        """Test annotating row indices on 1D tensor."""
        diagram = to_diagram(shape_1d)
        annotated = diagram.annotate_dim_indices("row")
        assert annotated.rank == 1
        assert annotated.tensor_shape == shape_1d

    def test_annotate_dim_indices_1d_all(self, shape_1d):
        """Test annotating all indices on 1D tensor."""
        diagram = to_diagram(shape_1d)
        annotated = diagram.annotate_dim_indices("all")
        assert annotated.rank == 1
        assert annotated.tensor_shape == shape_1d

    def test_annotate_dim_indices_1d_with_color(self, shape_1d):
        """Test annotating 1D tensor indices with custom color."""
        diagram = to_diagram(shape_1d)
        annotated = diagram.annotate_dim_indices("row", color="red")
        assert annotated.rank == 1
        assert annotated.tensor_shape == shape_1d

    def test_annotate_dim_indices_1d_invalid_col(self, shape_1d):
        """Test that 1D tensor rejects column indices."""
        diagram = to_diagram(shape_1d)
        with pytest.raises(ValueError, match="1D tensors can only annotate 'row' dimension"):
            diagram.annotate_dim_indices("col")

    def test_annotate_dim_indices_1d_invalid_depth(self, shape_1d):
        """Test that 1D tensor rejects depth indices."""
        diagram = to_diagram(shape_1d)
        with pytest.raises(ValueError, match="1D tensors can only annotate 'row' dimension"):
            diagram.annotate_dim_indices("depth")

    def test_annotate_dim_indices_2d_row(self, shape_2d):
        """Test annotating row indices on 2D tensor."""
        diagram = to_diagram(shape_2d)
        annotated = diagram.annotate_dim_indices("row")
        assert annotated.rank == 2
        assert annotated.tensor_shape == shape_2d

    def test_annotate_dim_indices_2d_col(self, shape_2d):
        """Test annotating column indices on 2D tensor."""
        diagram = to_diagram(shape_2d)
        annotated = diagram.annotate_dim_indices("col")
        assert annotated.rank == 2
        assert annotated.tensor_shape == shape_2d

    def test_annotate_dim_indices_2d_all(self, shape_2d):
        """Test annotating all indices on 2D tensor."""
        diagram = to_diagram(shape_2d)
        annotated = diagram.annotate_dim_indices("all")
        assert annotated.rank == 2
        assert annotated.tensor_shape == shape_2d

    def test_annotate_dim_indices_2d_with_color(self, shape_2d):
        """Test annotating 2D tensor indices with custom color."""
        diagram = to_diagram(shape_2d)
        annotated = diagram.annotate_dim_indices("all", color="blue")
        assert annotated.rank == 2
        assert annotated.tensor_shape == shape_2d

    def test_annotate_dim_indices_2d_invalid_depth(self, shape_2d):
        """Test that 2D tensor rejects depth indices."""
        diagram = to_diagram(shape_2d)
        with pytest.raises(ValueError, match="2D tensors cannot annotate 'depth' dimension"):
            diagram.annotate_dim_indices("depth")

    def test_annotate_dim_indices_3d_row(self, shape_3d):
        """Test annotating row indices on 3D tensor."""
        diagram = to_diagram(shape_3d)
        annotated = diagram.annotate_dim_indices("row")
        assert annotated.rank == 3
        assert annotated.tensor_shape == shape_3d

    def test_annotate_dim_indices_3d_col(self, shape_3d):
        """Test annotating column indices on 3D tensor."""
        diagram = to_diagram(shape_3d)
        annotated = diagram.annotate_dim_indices("col")
        assert annotated.rank == 3
        assert annotated.tensor_shape == shape_3d

    def test_annotate_dim_indices_3d_depth(self, shape_3d):
        """Test annotating depth indices on 3D tensor."""
        diagram = to_diagram(shape_3d)
        annotated = diagram.annotate_dim_indices("depth")
        assert annotated.rank == 3
        assert annotated.tensor_shape == shape_3d

    def test_annotate_dim_indices_3d_all(self, shape_3d):
        """Test annotating all indices on 3D tensor."""
        diagram = to_diagram(shape_3d)
        annotated = diagram.annotate_dim_indices("all")
        assert annotated.rank == 3
        assert annotated.tensor_shape == shape_3d

    def test_annotate_dim_indices_3d_with_color(self, shape_3d):
        """Test annotating 3D tensor indices with custom color."""
        diagram = to_diagram(shape_3d)
        annotated = diagram.annotate_dim_indices("all", color="green")
        assert annotated.rank == 3
        assert annotated.tensor_shape == shape_3d

    def test_annotate_dim_indices_immutability(self, numpy_array_2d):
        """Test that annotate_dim_indices returns new instance."""
        original = to_diagram(numpy_array_2d)
        annotated = original.annotate_dim_indices("all")

        # Should be different instances
        assert original is not annotated

        # Both should still be valid
        assert original.rank == 2
        assert annotated.rank == 2
        assert original.tensor_shape == annotated.tensor_shape

    def test_annotate_dim_indices_chain_with_fill_color(self, numpy_array_2d):
        """Test chaining annotate_dim_indices with fill_color."""
        diagram = to_diagram(numpy_array_2d)
        styled = diagram.annotate_dim_indices("all", color="red").fill_color("blue")

        assert styled.rank == 2
        assert styled.tensor_shape == (3, 4)

    def test_annotate_dim_indices_chain_with_fill_region(self, numpy_array_2d):
        """Test chaining annotate_dim_indices with fill_region."""
        diagram = to_diagram(numpy_array_2d)
        styled = diagram.annotate_dim_indices("all").fill_region(
            start_coord=(0, 0), end_coord=(2, 2), color="green", opacity=None
        )

        assert styled.rank == 2
        assert styled.tensor_shape == (3, 4)

    def test_annotate_dim_indices_multiple_calls(self, numpy_array_2d):
        """Test multiple annotate_dim_indices calls."""
        diagram = to_diagram(numpy_array_2d)
        # First call annotates row
        annotated_row = diagram.annotate_dim_indices("row", color="red")
        # Second call annotates col (should preserve row annotation)
        annotated_both = annotated_row.annotate_dim_indices("col", color="blue")

        assert annotated_both.rank == 2
        assert annotated_both.tensor_shape == (3, 4)

    def test_annotate_dim_indices_persists_through_styling(self, shape_2d):
        """Test that index annotations persist when applying styling methods."""
        diagram = to_diagram(shape_2d)
        styled = (
            diagram
            .annotate_dim_indices("all", color="gray")
            .fill_color("red")
            .fill_opacity(0.6)
        )

        assert styled.rank == 2
        assert styled.tensor_shape == shape_2d

    def test_annotate_dim_indices_3d_multiple_dimensions(self, numpy_array_3d):
        """Test annotating multiple individual index dimensions on 3D tensor."""
        diagram = to_diagram(numpy_array_3d)
        annotated = (
            diagram
            .annotate_dim_indices("col", color="red")
            .annotate_dim_indices("depth", color="blue")
        )

        assert annotated.rank == 3
        assert annotated.tensor_shape == (2, 3, 4)

    def test_annotate_dim_indices_and_size_together(self, numpy_array_2d):
        """Test combining index and size annotations."""
        diagram = to_diagram(numpy_array_2d)
        annotated = (
            diagram
            .annotate_dim_indices("all", color="blue")
            .annotate_dim_size("all", color="red")
        )

        assert annotated.rank == 2
        assert annotated.tensor_shape == (3, 4)

    def test_annotate_indices_then_size_then_style(self, numpy_array_2d):
        """Test chaining index annotations, size annotations, and styling."""
        diagram = to_diagram(numpy_array_2d)
        styled = (
            diagram
            .annotate_dim_indices("row", color="gray")
            .annotate_dim_size("col", color="green")
            .fill_color("blue")
            .fill_opacity(0.5)
        )

        assert styled.rank == 2
        assert styled.tensor_shape == (3, 4)

    def test_annotate_dim_indices_default_color(self, shape_2d):
        """Test annotating indices with default color (None)."""
        diagram = to_diagram(shape_2d)
        annotated = diagram.annotate_dim_indices("all")
        assert annotated.rank == 2
        assert annotated.tensor_shape == shape_2d


class TestGradientOrders:
    """Tests for gradient orders with fill_opacity."""

    # 1D tensor gradient order tests
    def test_1d_gradient_order_R(self, shape_1d):
        """Test 1D tensor with TensorOrder.R (only valid order)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_1d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.R)

        assert gradient_diagram.rank == 1
        assert gradient_diagram.tensor_shape == shape_1d

    def test_1d_gradient_default_order(self, shape_1d):
        """Test 1D tensor defaults to TensorOrder.R when order not specified."""
        diagram = to_diagram(shape_1d)
        gradient_diagram = diagram.fill_opacity(0.2, 0.8)

        assert gradient_diagram.rank == 1
        assert gradient_diagram.tensor_shape == shape_1d

    def test_1d_gradient_invalid_order_raises_error(self, shape_1d):
        """Test that invalid orders for 1D tensor raise ValueError."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_1d)

        with pytest.raises(ValueError, match="For 1D tensors, order for opacity must be 'r'"):
            diagram.fill_opacity(0.1, 0.9, order=TensorOrder.C)

    # 2D tensor gradient order tests
    def test_2d_gradient_order_R(self, shape_2d):
        """Test 2D tensor with TensorOrder.R (row-wise gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_2d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.R)

        assert gradient_diagram.rank == 2
        assert gradient_diagram.tensor_shape == shape_2d

    def test_2d_gradient_order_C(self, shape_2d):
        """Test 2D tensor with TensorOrder.C (column-wise gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_2d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.C)

        assert gradient_diagram.rank == 2
        assert gradient_diagram.tensor_shape == shape_2d

    def test_2d_gradient_order_RC(self, shape_2d):
        """Test 2D tensor with TensorOrder.RC (row-then-column gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_2d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.RC)

        assert gradient_diagram.rank == 2
        assert gradient_diagram.tensor_shape == shape_2d

    def test_2d_gradient_order_CR(self, shape_2d):
        """Test 2D tensor with TensorOrder.CR (column-then-row gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_2d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.CR)

        assert gradient_diagram.rank == 2
        assert gradient_diagram.tensor_shape == shape_2d

    def test_2d_gradient_default_order_is_RC(self, shape_2d):
        """Test 2D tensor defaults to TensorOrder.RC when order not specified."""
        diagram = to_diagram(shape_2d)
        gradient_diagram = diagram.fill_opacity(0.2, 0.8)

        assert gradient_diagram.rank == 2
        assert gradient_diagram.tensor_shape == shape_2d

    def test_2d_gradient_invalid_order_raises_error(self, shape_2d):
        """Test that invalid orders for 2D tensor raise ValueError."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_2d)

        with pytest.raises(ValueError, match="For 2D tensors, order for opacity must be"):
            diagram.fill_opacity(0.1, 0.9, order=TensorOrder.D)

    def test_2d_fill_region_with_gradient_order_R(self, numpy_array_2d):
        """Test fill_region with gradient order R on 2D tensor."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(numpy_array_2d)
        filled = diagram.fill_region(
            start_coord=(0, 0),
            end_coord=(2, 3),
            color=None,
            opacity=(0.2, 0.9, TensorOrder.R)
        )

        assert filled.rank == 2
        assert filled.tensor_shape == (3, 4)

    def test_2d_fill_region_with_gradient_order_C(self, numpy_array_2d):
        """Test fill_region with gradient order C on 2D tensor."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(numpy_array_2d)
        filled = diagram.fill_region(
            start_coord=(0, 0),
            end_coord=(2, 3),
            color=None,
            opacity=(0.2, 0.9, TensorOrder.C)
        )

        assert filled.rank == 2
        assert filled.tensor_shape == (3, 4)

    def test_2d_fill_region_with_gradient_order_CR(self, numpy_array_2d):
        """Test fill_region with gradient order CR on 2D tensor."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(numpy_array_2d)
        filled = diagram.fill_region(
            start_coord=(0, 0),
            end_coord=(2, 3),
            color=None,
            opacity=(0.2, 0.9, TensorOrder.CR)
        )

        assert filled.rank == 2
        assert filled.tensor_shape == (3, 4)

    # 3D tensor gradient order tests - single dimension orders
    def test_3d_gradient_order_R(self, shape_3d):
        """Test 3D tensor with TensorOrder.R (row gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.R)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_C(self, shape_3d):
        """Test 3D tensor with TensorOrder.C (column gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.C)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_D(self, shape_3d):
        """Test 3D tensor with TensorOrder.D (depth gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.D)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    # 3D tensor gradient order tests - two dimension orders
    def test_3d_gradient_order_RC(self, shape_3d):
        """Test 3D tensor with TensorOrder.RC (row-column gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.RC)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_RD(self, shape_3d):
        """Test 3D tensor with TensorOrder.RD (row-depth gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.RD)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_CR(self, shape_3d):
        """Test 3D tensor with TensorOrder.CR (column-row gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.CR)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_CD(self, shape_3d):
        """Test 3D tensor with TensorOrder.CD (column-depth gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.CD)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_DR(self, shape_3d):
        """Test 3D tensor with TensorOrder.DR (depth-row gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.DR)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_DC(self, shape_3d):
        """Test 3D tensor with TensorOrder.DC (depth-column gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.DC)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    # 3D tensor gradient order tests - three dimension orders
    def test_3d_gradient_order_RCD(self, shape_3d):
        """Test 3D tensor with TensorOrder.RCD (row-column-depth gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.RCD)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_RDC(self, shape_3d):
        """Test 3D tensor with TensorOrder.RDC (row-depth-column gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.RDC)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_CRD(self, shape_3d):
        """Test 3D tensor with TensorOrder.CRD (column-row-depth gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.CRD)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_CDR(self, shape_3d):
        """Test 3D tensor with TensorOrder.CDR (column-depth-row gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.CDR)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_DRC(self, shape_3d):
        """Test 3D tensor with TensorOrder.DRC (depth-row-column gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.DRC)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_order_DCR(self, shape_3d):
        """Test 3D tensor with TensorOrder.DCR (depth-column-row gradient)."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.1, 0.9, order=TensorOrder.DCR)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_gradient_default_order_is_RCD(self, shape_3d):
        """Test 3D tensor defaults to TensorOrder.RCD when order not specified."""
        diagram = to_diagram(shape_3d)
        gradient_diagram = diagram.fill_opacity(0.2, 0.8)

        assert gradient_diagram.rank == 3
        assert gradient_diagram.tensor_shape == shape_3d

    def test_3d_fill_region_with_gradient_order_D(self, numpy_array_3d):
        """Test fill_region with gradient order D on 3D tensor."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(numpy_array_3d)
        filled = diagram.fill_region(
            start_coord=(0, 0, 0),
            end_coord=(1, 2, 3),
            color=None,
            opacity=(0.2, 0.9, TensorOrder.D)
        )

        assert filled.rank == 3
        assert filled.tensor_shape == (2, 3, 4)

    def test_3d_fill_region_with_gradient_order_RDC(self, numpy_array_3d):
        """Test fill_region with gradient order RDC on 3D tensor."""
        from tensordiagram.types import TensorOrder

        diagram = to_diagram(numpy_array_3d)
        filled = diagram.fill_region(
            start_coord=(0, 0, 0),
            end_coord=(1, 2, 3),
            color=None,
            opacity=(0.2, 0.9, TensorOrder.RDC)
        )

        assert filled.rank == 3
        assert filled.tensor_shape == (2, 3, 4)

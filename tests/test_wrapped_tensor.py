"""Unit tests for WrappedTensor wrapper class."""

import pytest
from tensordiagram.core import WrappedTensor


class TestWrappedTensor:
    """Tests for the WrappedTensor wrapper class."""

    @pytest.mark.torch
    def test_wrapped_tensor_shape(self, numpy_array_2d):
        """Test that WrappedTensor exposes shape correctly."""
        wrapped = WrappedTensor(numpy_array_2d)
        assert wrapped.shape == (3, 4)

    @pytest.mark.torch
    def test_wrapped_tensor_flatten(self, numpy_array_2d):
        """Test that WrappedTensor can flatten."""
        wrapped = WrappedTensor(numpy_array_2d)
        flattened = wrapped.flatten()
        assert len(flattened) == 12
        assert isinstance(flattened, list)

    @pytest.mark.torch
    def test_wrapped_tensor_indexing(self, numpy_array_2d_square):
        """Test that WrappedTensor supports indexing."""
        wrapped = WrappedTensor(numpy_array_2d_square)
        # Test single element access
        value = wrapped[0, 0]
        assert value == 0.0
        value = wrapped[1, 1]
        assert value == 5.0

    # PyTorch tests
    @pytest.mark.torch
    def test_wrapped_tensor_pytorch(self, torch_tensor_2d):
        """Test WrappedTensor creation with PyTorch tensor."""
        wrapped = WrappedTensor(torch_tensor_2d)
        assert wrapped.shape == (3, 4)

    # JAX tests
    @pytest.mark.jax
    def test_wrapped_tensor_jax(self, jax_array_2d):
        """Test WrappedTensor creation with JAX array."""
        wrapped = WrappedTensor(jax_array_2d)
        assert wrapped.shape == (3, 4)

    # TensorFlow tests
    @pytest.mark.tensorflow
    def test_wrapped_tensor_tensorflow(self, tf_tensor_2d):
        """Test WrappedTensor creation with TensorFlow tensor."""
        wrapped = WrappedTensor(tf_tensor_2d)
        assert wrapped.shape == (3, 4)

    # MLX tests
    @pytest.mark.mlx
    def test_wrapped_tensor_mlx(self, mlx_array_2d):
        """Test WrappedTensor creation with MLX array."""
        wrapped = WrappedTensor(mlx_array_2d)
        assert wrapped.shape == (3, 4)

    # List-based tensor tests
    def test_wrapped_tensor_list(self):
        """Test WrappedTensor creation with list."""
        tensor_list = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        wrapped = WrappedTensor(tensor_list)
        assert wrapped.shape == (2, 3)

    def test_wrapped_tensor_not_a_tensor(self):
        """Test that WrappedTensor raises error for objects without shape."""
        with pytest.raises(TypeError, match="Unsupported tensor type"):
            _ = WrappedTensor("not a tensor")
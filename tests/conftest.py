"""Pytest configuration and shared fixtures for tensordiagram tests."""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    import torch # type: ignore[import-error]

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import jax # type: ignore[import-error]
    import jax.numpy as jnp # type: ignore[import-error]

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

try:
    import tensorflow as tf # type: ignore[import-error]

    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

try:
    import mlx.core as mx # type: ignore[import-error]

    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

try:
    from PIL import Image

    PILLOW_AVAILABLE = True
except ImportError:
    print("WARNING: Pillow not installed; image comparison will use hash comparison.")
    PILLOW_AVAILABLE = False


# Fixtures for temporary directories
@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    # For debugging, uncomment following
    # tmpdir = Path("tests/tmp")
    # tmpdir.mkdir(exist_ok=True)
    # yield tmpdir

    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def fixtures_dir():
    """Return the path to the test fixtures directory."""
    return Path(__file__).parent / "fixtures"


# Fixtures for common tensor shapes
@pytest.fixture
def shape_1d():
    """Return a 1D tensor shape."""
    return (5,)


@pytest.fixture
def shape_2d():
    """Return a 2D tensor shape."""
    return (3, 4)


@pytest.fixture
def shape_2d_square():
    """Return a square 2D tensor shape."""
    return (4, 4)


@pytest.fixture
def shape_3d():
    """Return a 3D tensor shape."""
    return (2, 3, 4)


# PyTorch tensor fixtures
if TORCH_AVAILABLE:

    @pytest.fixture
    def torch_tensor_1d():
        """Create a 1D PyTorch tensor."""
        return torch.arange(5, dtype=torch.float32)

    @pytest.fixture
    def torch_tensor_2d():
        """Create a 2D PyTorch tensor."""
        return torch.arange(12, dtype=torch.float32).reshape(3, 4)

    @pytest.fixture
    def torch_tensor_2d_square():
        """Create a square 2D PyTorch tensor."""
        return torch.arange(16, dtype=torch.float32).reshape(4, 4)

    @pytest.fixture
    def torch_tensor_3d():
        """Create a 3D PyTorch tensor."""
        return torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)


# NumPy array fixtures
@pytest.fixture
def numpy_array_1d():
    """Create a 1D NumPy array."""
    return np.arange(5, dtype=np.float32)


@pytest.fixture
def numpy_array_2d():
    """Create a 2D NumPy array."""
    return np.arange(12, dtype=np.float32).reshape(3, 4)


@pytest.fixture
def numpy_array_2d_square():
    """Create a square 2D NumPy array."""
    return np.arange(16, dtype=np.float32).reshape(4, 4)


@pytest.fixture
def numpy_array_3d():
    """Create a 3D NumPy array."""
    return np.arange(24, dtype=np.float32).reshape(2, 3, 4)


# JAX array fixtures
if JAX_AVAILABLE:

    @pytest.fixture
    def jax_array_1d():
        """Create a 1D JAX array."""
        return jnp.arange(5, dtype=jnp.float32)

    @pytest.fixture
    def jax_array_2d():
        """Create a 2D JAX array."""
        return jnp.arange(12, dtype=jnp.float32).reshape(3, 4)

    @pytest.fixture
    def jax_array_2d_square():
        """Create a square 2D JAX array."""
        return jnp.arange(16, dtype=jnp.float32).reshape(4, 4)

    @pytest.fixture
    def jax_array_3d():
        """Create a 3D JAX array."""
        return jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4)


# TensorFlow tensor fixtures
if TENSORFLOW_AVAILABLE:

    @pytest.fixture
    def tf_tensor_1d():
        """Create a 1D TensorFlow tensor."""
        return tf.range(5, dtype=tf.float32)

    @pytest.fixture
    def tf_tensor_2d():
        """Create a 2D TensorFlow tensor."""
        return tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))

    @pytest.fixture
    def tf_tensor_2d_square():
        """Create a square 2D TensorFlow tensor."""
        return tf.reshape(tf.range(16, dtype=tf.float32), (4, 4))

    @pytest.fixture
    def tf_tensor_3d():
        """Create a 3D TensorFlow tensor."""
        return tf.reshape(tf.range(24, dtype=tf.float32), (2, 3, 4))


# MLX array fixtures
if MLX_AVAILABLE:

    @pytest.fixture
    def mlx_array_1d():
        """Create a 1D MLX array."""
        return mx.arange(5, dtype=mx.float32)

    @pytest.fixture
    def mlx_array_2d():
        """Create a 2D MLX array."""
        return mx.arange(12, dtype=mx.float32).reshape(3, 4)

    @pytest.fixture
    def mlx_array_2d_square():
        """Create a square 2D MLX array."""
        return mx.arange(16, dtype=mx.float32).reshape(4, 4)

    @pytest.fixture
    def mlx_array_3d():
        """Create a 3D MLX array."""
        return mx.arange(24, dtype=mx.float32).reshape(2, 3, 4)


# Utility functions
def compute_image_hash(image_path: Path) -> str:
    """
    Compute a hash of an image file for comparison.

    Args:
        image_path: Path to the image file

    Returns:
        MD5 hash of the image content
    """
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def compare_svg_files(file1: Path, file2: Path) -> bool:
    content1 = file1.read_text()
    content2 = file2.read_text()
    return content1 == content2


def compare_images(
    image1_path: Path, image2_path: Path, tolerance: float = 0.01
) -> bool:
    if not PILLOW_AVAILABLE:
        # Fallback to hash comparison if PIL not available
        return compute_image_hash(image1_path) == compute_image_hash(image2_path)

    img1 = Image.open(image1_path)
    img2 = Image.open(image2_path)

    # Ensure same size
    if img1.size != img2.size:
        return False

    # Convert to RGB if needed
    if img1.mode != "RGB":
        img1 = img1.convert("RGB")
    if img2.mode != "RGB":
        img2 = img2.convert("RGB")

    # Compare pixel by pixel
    pixels1 = list(img1.getdata()) # type: ignore[attr-defined]
    pixels2 = list(img2.getdata()) # type: ignore[attr-defined]

    if len(pixels1) != len(pixels2):
        return False

    differences = 0
    for p1, p2 in zip(pixels1, pixels2):
        if p1 != p2:
            differences += 1

    diff_ratio = differences / len(pixels1)
    return diff_ratio <= tolerance


@pytest.fixture
def image_comparator():
    """Return a function for comparing images."""
    return compare_images


@pytest.fixture
def svg_comparator():
    """Return a function for comparing SVG files."""
    return compare_svg_files


# Auto-skip tests based on available dependencies
def pytest_collection_modifyitems(config, items):
    """Skip tests based on available dependencies."""
    skip_torch = pytest.mark.skip(reason="PyTorch not installed")
    skip_jax = pytest.mark.skip(reason="JAX not installed")
    skip_tensorflow = pytest.mark.skip(reason="TensorFlow not installed")
    skip_mlx = pytest.mark.skip(reason="MLX not installed")

    for item in items:
        if "torch" in item.keywords and not TORCH_AVAILABLE:
            item.add_marker(skip_torch)
        if "jax" in item.keywords and not JAX_AVAILABLE:
            item.add_marker(skip_jax)
        if "tensorflow" in item.keywords and not TENSORFLOW_AVAILABLE:
            item.add_marker(skip_tensorflow)
        if "mlx" in item.keywords and not MLX_AVAILABLE:
            item.add_marker(skip_mlx)

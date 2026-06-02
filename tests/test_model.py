"""Unit tests for src/model.py — skipped when TensorFlow is not installed."""

import pytest

from src.config import INPUT_SHAPE, NUM_CLASSES


# All tests in this module require TensorFlow.
tf = pytest.importorskip("tensorflow")
keras = tf.keras


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build():
    from src.model import build_cnn
    return build_cnn()


# ---------------------------------------------------------------------------
# build_cnn — construction and compilation
# ---------------------------------------------------------------------------


def test_build_cnn_callable_with_defaults():
    """build_cnn() must succeed using only the config defaults."""
    model = _build()
    assert model is not None


def test_build_cnn_is_keras_model():
    model = _build()
    assert isinstance(model, keras.Model)


def test_build_cnn_is_compiled():
    """Model must be compiled: loss and optimizer must be set."""
    model = _build()
    assert model.loss is not None, "model.loss should be set after compile()"
    assert model.optimizer is not None, "model.optimizer should be set after compile()"


def test_build_cnn_optimizer_is_adam():
    model = _build()
    assert isinstance(model.optimizer, keras.optimizers.Adam)


def test_build_cnn_optimizer_learning_rate():
    model = _build()
    lr = float(model.optimizer.learning_rate)
    assert abs(lr - 1e-3) < 1e-7, f"expected lr=1e-3, got {lr}"


# ---------------------------------------------------------------------------
# build_cnn — architecture shape
# ---------------------------------------------------------------------------


def test_build_cnn_input_shape():
    model = _build()
    # model.input_shape is (None, *INPUT_SHAPE) — batch dim is None
    assert model.input_shape == (None,) + INPUT_SHAPE


def test_build_cnn_output_shape():
    model = _build()
    assert model.output_shape == (None, NUM_CLASSES)


def test_build_cnn_output_activation_is_softmax():
    model = _build()
    last_layer = model.layers[-1]
    # Dense with softmax activation — config name contains "softmax"
    assert "softmax" in last_layer.get_config().get("activation", ""), (
        "final layer must use softmax activation"
    )


# ---------------------------------------------------------------------------
# build_cnn — forward pass
# ---------------------------------------------------------------------------


def test_build_cnn_forward_pass_shape():
    import numpy as np
    model = _build()
    batch = np.random.randn(4, *INPUT_SHAPE).astype("float32")
    preds = model.predict(batch, verbose=0)
    assert preds.shape == (4, NUM_CLASSES)


def test_build_cnn_output_is_probability_distribution():
    """Softmax outputs must sum to ~1.0 per sample."""
    import numpy as np
    model = _build()
    batch = np.random.randn(2, *INPUT_SHAPE).astype("float32")
    preds = model.predict(batch, verbose=0)
    row_sums = preds.sum(axis=1)
    np.testing.assert_allclose(row_sums, np.ones(2), atol=1e-5)


# ---------------------------------------------------------------------------
# build_cnn — layer structure
# ---------------------------------------------------------------------------


def test_build_cnn_has_three_conv_blocks():
    model = _build()
    conv_layers = [l for l in model.layers if isinstance(l, keras.layers.Conv2D)]
    assert len(conv_layers) == 3, f"expected 3 Conv2D layers, got {len(conv_layers)}"


def test_build_cnn_conv_filters_are_32_64_128():
    model = _build()
    filters = [l.filters for l in model.layers if isinstance(l, keras.layers.Conv2D)]
    assert filters == [32, 64, 128], f"unexpected filter counts {filters}"


def test_build_cnn_has_two_maxpool_layers():
    model = _build()
    pool_layers = [l for l in model.layers if isinstance(l, keras.layers.MaxPooling2D)]
    assert len(pool_layers) == 2


def test_build_cnn_has_global_average_pooling():
    model = _build()
    gap_layers = [l for l in model.layers if isinstance(l, keras.layers.GlobalAveragePooling2D)]
    assert len(gap_layers) == 1


def test_build_cnn_has_dropout():
    model = _build()
    dropout_layers = [l for l in model.layers if isinstance(l, keras.layers.Dropout)]
    assert len(dropout_layers) == 1


def test_build_cnn_dropout_rate_default():
    model = _build()
    dropout_layer = next(l for l in model.layers if isinstance(l, keras.layers.Dropout))
    assert dropout_layer.rate == pytest.approx(0.4)


def test_build_cnn_custom_dropout_rate():
    from src.model import build_cnn
    model = build_cnn(dropout_rate=0.5)
    dropout_layer = next(l for l in model.layers if isinstance(l, keras.layers.Dropout))
    assert dropout_layer.rate == pytest.approx(0.5)


def test_build_cnn_has_batch_norm_after_each_conv():
    model = _build()
    bn_layers = [l for l in model.layers if isinstance(l, keras.layers.BatchNormalization)]
    assert len(bn_layers) == 3, f"expected 3 BatchNorm layers, got {len(bn_layers)}"

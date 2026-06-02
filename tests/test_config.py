import math

from src.config import (
    CLASS_LABELS,
    DURATION,
    HOP_LENGTH,
    INPUT_SHAPE,
    N_FFT,
    N_MELS,
    NUM_CLASSES,
    SAMPLE_RATE,
)


def test_all_constants_importable():
    assert SAMPLE_RATE is not None
    assert DURATION is not None
    assert N_FFT is not None
    assert HOP_LENGTH is not None
    assert N_MELS is not None
    assert NUM_CLASSES is not None
    assert INPUT_SHAPE is not None
    assert CLASS_LABELS is not None


def test_constant_values():
    assert SAMPLE_RATE == 22050
    assert DURATION == 4.0
    assert N_FFT == 2048
    assert HOP_LENGTH == 512
    assert N_MELS == 128
    assert NUM_CLASSES == 5
    assert INPUT_SHAPE == (128, 173, 1)


def test_input_shape_time_frames_derived():
    # T = ceil(DURATION * SAMPLE_RATE / HOP_LENGTH)
    expected_frames = math.ceil(DURATION * SAMPLE_RATE / HOP_LENGTH)
    assert INPUT_SHAPE[1] == expected_frames


def test_class_labels_structure():
    assert isinstance(CLASS_LABELS, dict)
    assert len(CLASS_LABELS) == NUM_CLASSES
    assert set(CLASS_LABELS.keys()) == {0, 1, 2, 3, 4}
    assert all(isinstance(v, str) for v in CLASS_LABELS.values())


def test_class_labels_values():
    assert CLASS_LABELS[0] == "normal_operation"
    assert CLASS_LABELS[1] == "metallic_impact"
    assert CLASS_LABELS[2] == "friction_squeal"
    assert CLASS_LABELS[3] == "alarm_tone"
    assert CLASS_LABELS[4] == "silence_ambient"

SAMPLE_RATE = 22050
DURATION = 4.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
NUM_CLASSES = 5
INPUT_SHAPE = (128, 173, 1)
CLASS_LABELS = {
    0: "normal_operation",
    1: "metallic_impact",
    2: "friction_squeal",
    3: "alarm_tone",
    4: "silence_ambient",
}
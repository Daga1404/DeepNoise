SAMPLE_RATE = 22050
DURATION = 4.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
NUM_CLASSES = 4
INPUT_SHAPE = (128, 173, 1)
CLASS_LABELS = {
    0: "normal_operation",
    1: "metallic_impact",
    2: "alarm_tone",
    3: "silence_ambient",
}
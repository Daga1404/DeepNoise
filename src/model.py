"""CNN architecture for acoustic event classification."""

from tensorflow import keras

from src.config import INPUT_SHAPE, NUM_CLASSES


def build_cnn(
    input_shape: tuple = INPUT_SHAPE,
    num_classes: int = NUM_CLASSES,
    dropout_rate: float = 0.4,
) -> keras.Model:
    """
    Build and compile the acoustic event classification CNN.

    Architecture (from DESIGN.md §CNN Architecture):
        Input → Conv2D(32)+BN+ReLU+MaxPool → Conv2D(64)+BN+ReLU+MaxPool
               → Conv2D(128)+BN+ReLU → GlobalAvgPool
               → Dense(128)+ReLU+Dropout → Dense(num_classes)+Softmax

    Compiled with Adam(lr=1e-3) and categorical_crossentropy loss.

    Args:
        input_shape: Spectrogram tensor shape ``(n_mels, time_frames, channels)``.
        num_classes: Number of output classes.
        dropout_rate: Dropout probability applied before the final Dense layer.

    Returns:
        Compiled ``keras.Model``.
    """
    inputs = keras.Input(shape=input_shape)

    x = keras.layers.Conv2D(32, (3, 3), padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="acoustic_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

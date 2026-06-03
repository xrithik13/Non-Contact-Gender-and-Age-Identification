import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout
)

# -----------------------------------
# LOAD PRETRAINED MOBILENETV2
# -----------------------------------

base_model = MobileNetV2(

    weights='imagenet',

    include_top=False,

    input_shape=(128,128,3)
)

# -----------------------------------
# FREEZE BASE MODEL
# -----------------------------------

base_model.trainable = False

# -----------------------------------
# BUILD GENDER CLASSIFIER
# -----------------------------------

model = Sequential([

    base_model,

    GlobalAveragePooling2D(),

    Dense(
        128,
        activation='relu'
    ),

    Dropout(0.3),

    Dense(
        1,
        activation='sigmoid'
    )
])

# -----------------------------------
# COMPILE MODEL
# -----------------------------------

model.compile(

    optimizer='adam',

    loss='binary_crossentropy',

    metrics=['accuracy']
)

# -----------------------------------
# SHOW MODEL SUMMARY
# -----------------------------------

model.summary()
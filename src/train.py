import tensorflow as tf
import os
from model import build_model

DATA_DIR = "data"
MODEL_PATH = "model/my_face_model.h5"
IMG_SIZE = (100,100)
BATCH_SIZE = 32

ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

ds = ds.map(lambda x, y: (x/255.0, y))

model = build_model(input_shape=(100,100,3), num_classes=2)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(ds, epochs=10)
model.save(MODEL_PATH)

print("Model trained & saved")
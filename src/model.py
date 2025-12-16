from tensorflow import keras 
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential

def build_model(input_shape=(100,100,3), num_classes=2):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D(2,2),

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),

        Dropout(0.2),
        Flatten(),

        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    return model
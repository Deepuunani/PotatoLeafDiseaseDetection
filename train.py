import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(256,256),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

val_data = train_datagen.flow_from_directory(
    "dataset",
    target_size=(256,256),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

model = Sequential([
    Conv2D(16,(3,3),activation='relu',input_shape=(256,256,3)),
    MaxPooling2D(),

    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(),

    Flatten(),

    Dense(128,activation='relu'),

    Dense(3,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

model.save("potato_model.h5")

print("Model Saved Successfully")
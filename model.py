from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import os

# ---------------------------
# Data Preparation
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1/255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1/255)

batch_size = 4

train_generator = train_datagen.flow_from_directory(
    "E:/open cv/data/train",
    target_size=(48,48),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale"
)

val_generator = val_datagen.flow_from_directory(
    "E:/open cv/data/test",
    target_size=(48,48),
    batch_size=batch_size,
    class_mode="categorical",
    color_mode="grayscale"
)

class_name = list(train_generator.class_indices.keys())
print("Classes:", class_name)

# ---------------------------
# Model
# ---------------------------
model = Sequential()

model.add(Conv2D(32, (3,3), activation="relu", input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(128, (3,3), activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(9, activation="softmax"))   # Only 2 classes: A and B

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ---------------------------
# Training
# ---------------------------
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=60,   # start with 20, increase later if needed
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size,
    callbacks=[early_stop]
)

# ---------------------------
# Save Model
# ---------------------------
model.save("signLanguageModel_AB.keras")

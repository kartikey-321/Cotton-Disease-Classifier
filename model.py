from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

train_dir=('./Cotton Disease/train')
validation_dir=('./Cotton Disease/val')
test_dir=('./Cotton Disease/test')

train_data=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
test_data=ImageDataGenerator(rescale=1./255)

xtrain=train_data.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode="categorical"
)
xval=test_data.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode="categorical"
)
from tensorflow.keras.applications import ResNet50
conv_base=ResNet50(weights='imagenet',include_top=False,input_shape=(150,150,3))

model=keras.models.Sequential([
    conv_base,
    keras.layers.Flatten(),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(4,activation='softmax')
])
callback=keras.callbacks.ModelCheckpoint("cotton_disease.h5",save_best_only=True)
model.compile(loss="categorical_crossentropy",
             optimizer=keras.optimizers.RMSprop(learning_rate=2e-5),metrics=['acc'])

model_history=model.fit_generator(
    xtrain,
    steps_per_epoch=len(xtrain),
    epochs=20,
    validation_data=xval,
    validation_steps=len(xval),
    callbacks=[callback]
)
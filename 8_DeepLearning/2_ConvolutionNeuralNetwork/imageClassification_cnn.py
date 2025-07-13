import tensorflow as tf
import keras.preprocessing.image
import keras
from keras.src.legacy.preprocessing.image import ImageDataGenerator

##### Data Preprocessing #####

# Preprocessing Training set

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
training_set = train_datagen.flow_from_directory(
    '/Users/adityaagrawal/PycharmProjects/PythonProject/8_DeepLearning/2_ConvolutionNeuralNetwork/dataset_classification_dog_cat/training_set',
    target_size = (64, 64),
    batch_size=32,
    class_mode='binary'
)

# Preprocessing Training set

test_dataGen = ImageDataGenerator(rescale=1./255)

testing_set = test_dataGen.flow_from_directory(
    '/Users/adityaagrawal/PycharmProjects/PythonProject/8_DeepLearning/2_ConvolutionNeuralNetwork/dataset_classification_dog_cat/test_set',
    target_size = (64, 64),
    batch_size=32,
    class_mode='binary'
)

##### Building CNN #####

# Initialisation
cnn = keras.models.Sequential()

# Step1: Convolution
cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step2: Pooling
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding 2nd Convolution Layer
cnn.add(keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step3: Flattening
cnn.add(keras.layers.Flatten())

# Step4: Full Connection
cnn.add(keras.layers.Dense(units=128, activation='relu'))

# Step5: Output Layer
cnn.add(keras.layers.Dense(units=1, activation='sigmoid'))

##### Training CNN #####

# Compiling
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# training and testing
cnn.fit(x=training_set, validation_data=testing_set, epochs=25)

##### Single Prediction #####
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('/Users/adityaagrawal/PycharmProjects/PythonProject/8_DeepLearning/2_ConvolutionNeuralNetwork/dataset_classification_dog_cat/single_prediction/cat.5000.jpg',
                            target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)


test_image = image.load_img('/Users/adityaagrawal/PycharmProjects/PythonProject/8_DeepLearning/2_ConvolutionNeuralNetwork/dataset_classification_dog_cat/single_prediction/dog.5000.jpg',
                            target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = cnn.predict(test_image)
print(training_set.class_indices)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)
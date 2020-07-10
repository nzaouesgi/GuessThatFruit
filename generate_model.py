import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from os import listdir
import numpy as np
import tensorflow.keras as keras
from PIL import Image, ImageFile
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

ImageFile.LOAD_TRUNCATED_IMAGES = True

target_size = (64, 64)

def load_dataset() :
    train_path = "./Dataset/Train/"
    test_path = "./Dataset/Test/"

    x_train_list = []
    y_train_list = []
    x_test_list = []
    y_test_list = []

    load_set_from_directory(train_path, x_train_list, y_train_list)
    load_set_from_directory(test_path, x_test_list, y_test_list)

    return (np.array(x_train_list), np.array(y_train_list)), (np.array(x_test_list), np.array(y_test_list))


def load_set_from_directory(train_path, x_train_list, y_train_list):
    load_images_from_directory(np.array([1, 0, 0]), f'{train_path}Orange/', x_train_list, y_train_list)
    load_images_from_directory(np.array([0, 1, 0]), f'{train_path}Banane/', x_train_list, y_train_list)
    load_images_from_directory(np.array([0, 0, 1]), f'{train_path}Kiwi/', x_train_list, y_train_list)


def load_images_from_directory(label, path, x_train_list, y_train_list):
    for img_name in listdir(path):
        # x_train_list.append(np.array(Image.open(f'{path}{img_name}').convert('L').resize(target_size)) / 255.0) # Grayscale
        x_train_list.append(
            np.array(Image.open(f'{path}{img_name}').convert('RGB').resize(target_size)) / 255.0)  # color
        y_train_list.append(label)


def create_linear_model():
    m = keras.models.Sequential()
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(3, activation=keras.activations.tanh))
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m

def create_mlp_model():
    m = keras.models.Sequential()
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(32, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(32, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(32, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(3, activation=keras.activations.sigmoid))
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m

def create_mlp_model_2():
    m = keras.models.Sequential()
    m.add(keras.layers.Flatten())
    m.add(keras.layers.Dense(32, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(3, activation=keras.activations.sigmoid))
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m


def create_conv_nn_model():
    m = keras.models.Sequential()

    for i in range(1, 4):
        m.add(keras.layers.Conv2D(4 * i, kernel_size=(2, 2), activation=keras.activations.relu, padding='same'))
        if i < 3:
            m.add(keras.layers.MaxPool2D((2, 2)))
        else:
            m.add(keras.layers.AvgPool2D((2,2)))

    m.add(keras.layers.Flatten())

    m.add(keras.layers.Dense(64, activation=keras.activations.tanh))
    m.add(keras.layers.Dense(3, activation=keras.activations.sigmoid))
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m


def create_residual_nn_model():
    input_tensor = keras.layers.Input((target_size[0], target_size[1], 3))

    previous_tensor = input_tensor

    next_tensor = keras.layers.Conv2D(16, kernel_size=(2, 2), activation=keras.activations.relu, padding='same')(input_tensor)
    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])
    previous_tensor = keras.layers.MaxPool2D((2, 2))(previous_tensor)

    next_tensor = keras.layers.Conv2D(32, kernel_size=(2, 2), activation=keras.activations.relu, padding='same')(previous_tensor)
    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])
    previous_tensor = keras.layers.MaxPool2D((2, 2))(previous_tensor)

    next_tensor = keras.layers.Conv2D(64, kernel_size=(2, 2), activation=keras.activations.relu, padding='same')(previous_tensor)
    previous_tensor = keras.layers.Concatenate()([previous_tensor, next_tensor])
    previous_tensor = keras.layers.MaxPool2D((2, 2))(previous_tensor)
    
    previous_tensor = keras.layers.Flatten()(previous_tensor)
    next_tensor = keras.layers.Dense(64, activation=keras.activations.tanh)(previous_tensor)
    next_tensor = keras.layers.Dense(3, activation=keras.activations.sigmoid)(next_tensor)

    m = keras.models.Model(input_tensor, next_tensor)
    m.compile(optimizer=keras.optimizers.Adam(lr=0.0001),
              loss=keras.losses.mean_squared_error,
              metrics=['accuracy'])
    return m

def show_confusion_matrix(m, x, y, show_errors: bool = False):
    predicted_values = m.predict(x)
    predicted_labels = np.argmax(predicted_values, axis=1)
    true_labels = np.argmax(y, axis=1)

    print(confusion_matrix(true_labels, predicted_labels))

    if show_errors:
        for i in range(len(predicted_labels)):
            if predicted_labels[i] != true_labels[i]:
                plt.imshow(x[i])
                plt.show()


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_dataset()
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # model = create_linear_model()
    # model = create_mlp_model()
    # model = create_conv_nn_model()
    model = create_residual_nn_model()

    # show_confusion_matrix(model, x_test, y_test)

    logs = model.fit(x_train, y_train, epochs=50,
                     batch_size=4,
                     validation_data=(x_test, y_test))

    model.summary()

    # Affichage ddes courbes de loss et d'accuracy de l'apprentissage
    plt.plot(logs.history['loss'])
    plt.plot(logs.history['val_loss'])
    plt.show()

    # Affichage ddes courbes de loss et d'accuracy de l'apprentissage
    plt.plot(logs.history['accuracy'])
    plt.plot(logs.history['val_accuracy'])
    plt.show()
<
    # show_confusion_matrix(model, x_test, y_test, show_errors=True)

    model.save("Models/resnet1.keras")

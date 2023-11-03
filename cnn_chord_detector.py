import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from tensorflow.keras import layers, Model, regularizers
import matplotlib.pyplot as plt

DATA_PATH = "chords_data.json"


def load_data(data_path):
    """Loads training dataset from json file.

        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X_cqt = np.array(data["cqt"])
    y1 = np.array(data["chord_root_labels"])
    y2 = np.array(data["chord_type_labels"])
    y3 = np.array(data["chord_inversion_labels"])

    return X_cqt, y1, y2, y3


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
    """

    # load data
    X_cqt, y1, y2, y3 = load_data(DATA_PATH)

    # create train, validation and test split
    X_cqt_train, X_cqt_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test = train_test_split(X_cqt, y1, y2, y3, test_size=test_size)
    X_cqt_train, X_cqt_validation, y1_train, y1_validation, y2_train, y2_validation, y3_train, y3_validation = train_test_split(X_cqt_train, y1_train, y2_train, y3_train, test_size=validation_size)


    # add an axis to input sets
    X_cqt_train = X_cqt_train[..., np.newaxis]
    X_cqt_validation = X_cqt_validation[..., np.newaxis]
    X_cqt_test = X_cqt_test[..., np.newaxis]

    return X_cqt_train, X_cqt_validation, X_cqt_test, y1_train, y1_validation, y1_test, y2_train, y2_validation, y2_test, y3_train, y3_validation, y3_test


def build_model(image_input_shape):
    """Generates CNN model with separate layers for output1 and shared layers for output2 and output3.

    :param input_shape (tuple): Shape of input set
    :return model: CNN model
    """

    # Input layer
    image_input = layers.Input(shape=image_input_shape, name="image_input")

    # Shared convolutional layers

    # conv 1
    x = layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(image_input)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # conv 2
    x = layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Flatten and shared dense layers
    shared_layers = layers.Flatten()(x)
    shared_layers = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(shared_layers)
    shared_layers = layers.Dropout(0.3)(shared_layers)

    # Output branch for chord_type
    # private_chord_type = layers.Dense(20, activation='relu', kernel_regularizer=regularizers.l2(0.001))(shared_layers)
    # private_chord_type = layers.Dropout(0.3)(private_chord_type)
    chord_type_output = layers.Dense(10, activation='softmax', name='chord_type_output')(shared_layers)

    # Output branch for chord_inversion
    # private_chord_inversion = layers.Dense(12, activation='relu', kernel_regularizer=regularizers.l2(0.001))(shared_layers)
    # private_chord_inversion = layers.Dropout(0.3)(private_chord_inversion)
    chord_inversion_output = layers.Dense(4, activation='softmax', name='chord_inversion_output')(shared_layers)

    # Output branch for output1
    # chord_root_input = layers.Concatenate()([shared_layers, chord_type_output, chord_inversion_output])
    # private_chord_root = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001))(shared_layers)
    # private_chord_root = layers.Dropout(0.3)(private_chord_root)
    chord_root_output = layers.Dense(21, activation='softmax', name='chord_root_output')(shared_layers)


    # Create the model
    model = Model(inputs=[image_input], outputs=[chord_root_output, chord_type_output, chord_inversion_output])

    return model


def predict(model, X, y):
    """Predict a single sample using the trained model

    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = X[np.newaxis, ...] # array shape (1, 130, 13, 1)

    # perform prediction
    prediction = model.predict(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


if __name__ == "__main__":

    # get train, validation, test splits
    X_cqt_train, X_cqt_validation, X_cqt_test, y1_train, y1_validation, y1_test, y2_train, y2_validation, y2_test, y3_train, y3_validation, y3_test = prepare_datasets(0.25, 0.2)

    # create network
    image_shape = (X_cqt_train.shape[1], X_cqt_train.shape[2], 1)
    model = build_model(image_shape)

    # compile model
    optimiser = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimiser,
                  loss={'chord_root_output': 'sparse_categorical_crossentropy', 'chord_type_output': 'sparse_categorical_crossentropy', 'chord_inversion_output': 'sparse_categorical_crossentropy'},
                  metrics={'chord_root_output': 'accuracy', 'chord_type_output': 'accuracy', 'chord_inversion_output': 'accuracy'})

    model.summary()

    # train model
    history = model.fit(x={'image_input': X_cqt_train}, y={'chord_root_output': y1_train, "chord_type_output": y2_train, "chord_inversion_output": y3_train},
                        validation_data=[{'image_input': X_cqt_validation},
                                         {'chord_root_output': y1_validation, "chord_type_output": y2_validation, "chord_inversion_output": y3_validation}], batch_size=32, epochs=10)

    # plot accuracy/error for training and validation
    # plot_history(history)

    # evaluate model on test set
    #test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    loss, output1_loss, output2_loss, output3_loss, output1_accuracy, output2_accuracy, output3_accuracy = model.evaluate(
        {'image_input': X_cqt_test}, {'chord_root_output': y1_test, "chord_type_output": y2_test, "chord_inversion_output": y3_test},
        verbose=2
    )
    print('\nTest 1 accuracy:', output1_accuracy)
    print('\nTest 2 accuracy:', output2_accuracy)
    print('\nTest 3 accuracy:', output3_accuracy)

    # pick a sample to predict from the test set
    # X_to_predict = X_test[100]
    # y1_to_predict = y1_test[100]
    # y2_to_predict = y2_test[100]
    # y3_to_predict = y3_test[100]

    # predict sample
    # predict(model, {'conv2d_input': X_to_predict}, {'output1': y1_to_predict, 'output2': y2_to_predict, 'output3': y3_to_predict})

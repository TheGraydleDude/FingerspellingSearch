import numpy as np
import numpy.core.defchararray
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt


def plot_curve(x_axis, y_axis, plotting_metric):
    # creating a new graph, and labelling axis
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Value")

    # draw the graph, giving a list for each coordinate, and labelling the line
    plt.plot(x_axis, y_axis[plotting_metric], label=plotting_metric)

    # show the labels and the graph
    plt.legend()
    plt.show()


def comparing_losses_graph(x_axis, y_axis, plotting_metric_1, plotting_metric_2):
    # creating new graph with labelled axis
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # plotting both types of loss on the graph
    plt.plot(x_axis, y_axis[plotting_metric_1], label="Training Loss")
    plt.plot(x_axis, y_axis[plotting_metric_2], label="Validation Loss")

    # showing the labels and the graph
    plt.legend()
    plt.show()

# giving each column of the csv a name
columns = ["Char", "LeftHandStart", "LeftWristX", "LeftWristY",
           "LeftThumbCMCX", "LeftThumbCMCY", "LeftThumbMCPX",
           "LeftThumbMCPY", "LeftThumbIPX", "LeftThumbIPY",
           "LeftThumbTIPX", "LeftThumbTIPY", "LeftIndexFingerMCPX",
           "LeftIndexFingerMCPY", "LeftIndexFingerPIPX", "LeftIndexFingerPIPY",
           "LeftIndexFingerDIPX", "LeftIndexFingerDIPY", "LeftIndexFingerTIPX",
           "LeftIndexFingerTIPY", "LeftMiddleFingerMCPX",
           "LeftMiddleFingerMCPY",
           "LeftMiddleFingerPIPX", "LeftMiddleFingerPIPY",
           "LeftMiddleFingerDIPX",
           "LeftMiddleFingerDIPY", "LeftMiddleFingerTIPX",
           "LeftMiddleFingerTIPY",
           "LeftRingFingerMCPX", "LeftRingFingerMCPY", "LeftRingFingerPIPX",
           "LeftRingFingerPIPY", "LeftRingFingerDIPX", "LeftRingFingerDIPY",
           "LeftRingFingerTIPX", "LeftRingFingerTIPY", "LeftPinkyMCPX",
           "LeftPinkyMCPY", "LeftPinkyPIPX", "LeftPinkyPIPY", "LeftPinkyDIPX",
           "LeftPinkyDIPY", "LeftPinkyTIPX", "LeftPinkyTIPY", "RightHandStart",
           "RightWristX", "RightWristY", "RightThumbCMCX", "RightThumbCMCY",
           "RightThumbMCPX", "RightThumbMCPY", "RightThumbIPX", "RightThumbIPY",
           "RightThumbTIPX", "RightThumbTIPY", "RightIndexFingerMCPX",
           "RightIndexFingerMCPY", "RightIndexFingerPIPX",
           "RightIndexFingerPIPY", "RightIndexFingerDIPX",
           "RightIndexFingerDIPY", "RightIndexFingerTIPX",
           "RightIndexFingerTIPY", "RightMiddleFingerMCPX",
           "RightMiddleFingerMCPY", "RightMiddleFingerPIPX",
           "RightMiddleFingerPIPY", "RightMiddleFingerDIPX",
           "RightMiddleFingerDIPY", "RightMiddleFingerTIPX",
           "RightMiddleFingerTIPY", "RightRingFingerMCPX",
           "RightRingFingerMCPY", "RightRingFingerPIPX",
           "RightRingFingerPIPY", "RightRingFingerDIPX",
           "RightRingFingerDIPY", "RightRingFingerTIPX",
           "RightRingFingerTIPY", "RightPinkyMCPX",
           "RightPinkyMCPY", "RightPinkyPIPX", "RightPinkyPIPY",
           "RightPinkyDIPX", "RightPinkyDIPY", "RightPinkyTIPX",
           "RightPinkyTIPY"]

# reading in the training and testing data
fingerspelling_training_data = pd.read_csv("C:\\Users\\ausaf\\Documents\\Computing NEA\\Python\\fingerspelling.csv",
                                           names=columns)

fingerspelling_testing_data = pd.read_csv("C:\\Users\\ausaf\\Documents\\Computing NEA\\Python\\testing.csv",
                                          names=columns)

# shuffling the training and testing data, and separating features and labels
fingerspelling_training_data = fingerspelling_training_data.reindex(
    np.random.permutation(fingerspelling_training_data.index))
fingerspelling_training_features = fingerspelling_training_data.copy()
fingerspelling_training_labels = fingerspelling_training_features.pop("Char")
fingerspelling_training_features.pop("LeftHandStart")
fingerspelling_training_features.pop("RightHandStart")

fingerspelling_testing_data = fingerspelling_testing_data.reindex(
    np.random.permutation(fingerspelling_testing_data.index))
fingerspelling_testing_features = fingerspelling_testing_data.copy()
fingerspelling_testing_labels = fingerspelling_testing_features.pop("Char")
fingerspelling_testing_features.pop("LeftHandStart")
fingerspelling_testing_features.pop("RightHandStart")

#new columns for features now that some have been removed
columns = ["LeftWristX", "LeftWristY",
           "LeftThumbCMCX", "LeftThumbCMCY", "LeftThumbMCPX",
           "LeftThumbMCPY", "LeftThumbIPX", "LeftThumbIPY",
           "LeftThumbTIPX", "LeftThumbTIPY", "LeftIndexFingerMCPX",
           "LeftIndexFingerMCPY", "LeftIndexFingerPIPX", "LeftIndexFingerPIPY",
           "LeftIndexFingerDIPX", "LeftIndexFingerDIPY", "LeftIndexFingerTIPX",
           "LeftIndexFingerTIPY", "LeftMiddleFingerMCPX",
           "LeftMiddleFingerMCPY",
           "LeftMiddleFingerPIPX", "LeftMiddleFingerPIPY",
           "LeftMiddleFingerDIPX",
           "LeftMiddleFingerDIPY", "LeftMiddleFingerTIPX",
           "LeftMiddleFingerTIPY",
           "LeftRingFingerMCPX", "LeftRingFingerMCPY", "LeftRingFingerPIPX",
           "LeftRingFingerPIPY", "LeftRingFingerDIPX", "LeftRingFingerDIPY",
           "LeftRingFingerTIPX", "LeftRingFingerTIPY", "LeftPinkyMCPX",
           "LeftPinkyMCPY", "LeftPinkyPIPX", "LeftPinkyPIPY", "LeftPinkyDIPX",
           "LeftPinkyDIPY", "LeftPinkyTIPX", "LeftPinkyTIPY",
           "RightWristX", "RightWristY", "RightThumbCMCX", "RightThumbCMCY",
           "RightThumbMCPX", "RightThumbMCPY", "RightThumbIPX", "RightThumbIPY",
           "RightThumbTIPX", "RightThumbTIPY", "RightIndexFingerMCPX",
           "RightIndexFingerMCPY", "RightIndexFingerPIPX",
           "RightIndexFingerPIPY", "RightIndexFingerDIPX",
           "RightIndexFingerDIPY", "RightIndexFingerTIPX",
           "RightIndexFingerTIPY", "RightMiddleFingerMCPX",
           "RightMiddleFingerMCPY", "RightMiddleFingerPIPX",
           "RightMiddleFingerPIPY", "RightMiddleFingerDIPX",
           "RightMiddleFingerDIPY", "RightMiddleFingerTIPX",
           "RightMiddleFingerTIPY", "RightRingFingerMCPX",
           "RightRingFingerMCPY", "RightRingFingerPIPX",
           "RightRingFingerPIPY", "RightRingFingerDIPX",
           "RightRingFingerDIPY", "RightRingFingerTIPX",
           "RightRingFingerTIPY", "RightPinkyMCPX",
           "RightPinkyMCPY", "RightPinkyPIPX", "RightPinkyPIPY",
           "RightPinkyDIPX", "RightPinkyDIPY", "RightPinkyTIPX",
           "RightPinkyTIPY"]

# removing all whitespace that was in csv, and changing the values of co-ordinates to float so that the model can process
for col in columns:
    fingerspelling_training_features[col] = fingerspelling_training_features[col].str.replace("|", "", regex=True)
    fingerspelling_training_features[col] = fingerspelling_training_features[col].str.replace(" ", "",
                                                                                              regex=True).astype(
        "float32")

fingerspelling_training_features = np.array(fingerspelling_training_features)

for i in range(len(fingerspelling_training_labels)):
    fingerspelling_training_labels[i] = float(
        numpy.core.defchararray.replace(fingerspelling_training_labels[i], " ", ""))
fingerspelling_training_labels = np.asarray(fingerspelling_training_labels).astype("float32")

for col in columns:
    fingerspelling_testing_features[col] = fingerspelling_testing_features[col].str.replace("|", "", regex=True)
    fingerspelling_testing_features[col] = fingerspelling_testing_features[col].str.replace(" ", "", regex=True).astype(
        "float32")

fingerspelling_testing_features = np.array(fingerspelling_testing_features)
for i in range(len(fingerspelling_testing_labels)):
    fingerspelling_testing_labels[i] = float(numpy.core.defchararray.replace(fingerspelling_testing_labels[i], " ", ""))
fingerspelling_testing_labels = np.asarray(fingerspelling_testing_labels).astype("float32")

# creating the model and adding all the layers necessary:
# leaky relu for training weights, dropout for preventing overfitting and softmax to turn the weights into a prediction

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(units=32, activation=tf.keras.layers.LeakyReLU(alpha=0.01), input_shape=(84,)))

model.add(tf.keras.layers.Dropout(rate=0.2))

model.add(tf.keras.layers.Dense(units=29, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# training the model on the training data
history = model.fit(x=fingerspelling_training_features, y=fingerspelling_training_labels, batch_size=8,
                    epochs=40, validation_split=0.1)

# this stores what happened during the training on each epoch
epochs = history.epoch
hist = pd.DataFrame(history.history)

# plotting accuracy, loss and comparing loss with validation loss
plot_curve(epochs, hist, "accuracy")
plot_curve(epochs, hist, "loss")
comparing_losses_graph(epochs, hist, "loss", "val_loss")

# testing the model on the testing data
print("Evaluation:")
model.evaluate(x=fingerspelling_testing_features, y=fingerspelling_testing_labels, batch_size=8)

# saving the model
# model.save("C:\\Users\\ausaf\\Documents\\Computing NEA\\Main\\model\\my_model.h5")
model.summary()

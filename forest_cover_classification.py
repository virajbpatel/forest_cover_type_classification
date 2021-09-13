# Forest Cover Classification Project

'''

Need to predict Forest Cover type based on many variables. Data has been given in a CSV with 55 columns,
where the last column is the class corresponding to one of the following forest cover types:
Spruce/Fir
Lodgepole Pine
Ponderosa Pine
Cottonwood/Willow
Aspen
Douglas-fir
Krummholz

'''

# Import relevant modules

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt
import seaborn as sns

# Function to preprocess raw dataframe
def prep_data(df):
    data = df.values
    # Classes (labels) are in the final column, the rest are features
    x, y = data[:, :-1], data[:,-1] 
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42, stratify = y)
    # Normalize data using a standard scaler
    scaler = StandardScaler()
    x_train_normalised = scaler.fit_transform(x_train)
    x_test_normalised = scaler.transform(x_test)
    return x_train_normalised, x_test_normalised, y_train, y_test

# Builds classifier model from number of features
def build_model(num_features):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape = (num_features,)))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(32, activation = 'relu'))
    model.add(tf.keras.layers.Dense(8, activation = 'softmax'))
    model.compile(
        optimizer = 'adam',
        loss = 'sparse_categorical_crossentropy',
        metrics = ['accuracy']
    )
    return model

def plot_history(history, parameter):
    if parameter == "acc":
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title("Model Accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(['train', 'val'], loc = 'upper left')
        plt.savefig('accuracy.png')
        plt.show()
    elif parameter == "loss":
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model Loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(['train', 'val'], loc = 'upper right')
        plt.savefig('loss.png')
        plt.show()

# Plots heatmap based on confusion matrix
def plot_heatmap(class_names, y_pred, y_test):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(15, 15))
    heatmap = sns.heatmap(cm, fmt='g', cmap='Blues', annot=True, ax=ax)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names)
    ax.yaxis.set_ticklabels(class_names)
    # Save the heatmap to file
    heatmapfig = heatmap.get_figure()
    heatmapfig.savefig('confusion_matrix.png')

def main():
    # Import data file
    df = pd.read_csv("cover_data.csv")
    cols = df.columns.tolist()
    # Record features and labels
    features, labels = cols[:-1], cols[-1]
    # Preprocess data
    x_train, x_test, y_train, y_test = prep_data(df)
    # Build learning model
    model = build_model(len(features))
    print("Summary of Classifier: ")
    model.summary()
    # Trains model
    epochs = 100
    batch_size = 1024
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', min_delta = 0.0001, patience = 3)
    history = model.fit(
        x_train,
        y_train,
        epochs = epochs,
        batch_size = batch_size,
        callbacks = [earlystop_callback],
        validation_split = 0.1,
        verbose = 1
    )
    # Plots history of model
    plot_history(history, 'acc')
    plot_history(history, 'loss')
    # Tests model
    score = model.evaluate(x_test, y_test, verbose = 0)
    print(f'Loss: {score[0]}')
    print(f'Accuracy: {score[1]}')
    # Predicts classes based on test data
    y_pred = model.predict(x_test)
    y_pred = np.argmax(y_pred, axis = 1)
    class_names = ['Spruce/Fir','Lodgepole Pine','Ponderosa Pine',
    'Cottonwood/Willow','Aspen','Douglas-fir','Krummholz]']
    print(classification_report(y_test, y_pred, target_names = class_names))
    plot_heatmap(class_names, y_pred, y_test)


if __name__ == "__main__":
    main()
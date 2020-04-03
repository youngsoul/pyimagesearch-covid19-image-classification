# USAGE
# python train.py --dataset dataset

# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import VGG16,VGG19, ResNet50V2, ResNet50
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import cv2
import os
import time

# initialize the initial learning rate, number of epochs to train for,
# and batch size
INIT_LR = 1e-3
EPOCHS = 25
BS = 8



def run_model(baseModel, model_name, dataset_dir_name, trainX, testX, trainY, testY, lb):
    """

    :param baseModel:
    :type baseModel:
    :param model_name:
    :type model_name:
    :param trainX:
    :type trainX:
    :param testX:
    :type testX:
    :param trainY:
    :type trainY:
    :param testY:
    :type testY:
    :param lb:
    :type lb:
    :return:  [confusion matrix, classification report, model name]
    :rtype: List
    """
    saved_model_path = os.path.sep.join(['.', 'models', f'{model_name}-{dataset_dir_name}-model.h5'])
    # load the VGG16 network, ensuring the head FC layer sets are left
    # off
    # baseModel = VGG16(weights="imagenet", include_top=False,
    # 						input_tensor=Input(shape=(224, 224, 3)))
    # construct the head of the model that will be placed on top of the
    # the base model
    headModel = baseModel.output
    headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
    headModel = Flatten(name="flatten")(headModel)
    headModel = Dense(64, activation="relu")(headModel)
    headModel = Dropout(0.5)(headModel)
    headModel = Dense(3, activation="softmax")(headModel)
    # place the head FC model on top of the base model (this will become
    # the actual model we will train)
    model = Model(inputs=baseModel.input, outputs=headModel)
    # loop over all layers in the base model and freeze them so they will
    # *not* be updated during the first training process
    for layer in baseModel.layers:
        layer.trainable = False
    # compile our model
    print("[INFO] compiling model...")
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # initialize the training data augmentation object
    trainAug = ImageDataGenerator(
        rotation_range=15,
        fill_mode="nearest")

    # setup callback to save the best model
    fname = os.path.sep.join(['.', 'models', f"best-{model_name}-{dataset_dir_name}-model.h5"])
    checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min",
                                 save_best_only=True, verbose=1)
    callbacks = [checkpoint]

    # train the head of the network
    print("[INFO] training head...")
    H = model.fit_generator(
        trainAug.flow(trainX, trainY, batch_size=BS),
        steps_per_epoch=len(trainX) // BS,
        validation_data=(testX, testY),
        validation_steps=len(testX) // BS,
        epochs=EPOCHS,
        verbose=1,
        callbacks=callbacks)

    # make predictions on the testing set
    print("[INFO] evaluating network...")
    best_model = load_model(fname)
    predIdxs = best_model.predict(testX, batch_size=BS)

    # for each image in the testing set we need to find the index of the
    # label with corresponding largest predicted probability
    predIdxs = np.argmax(predIdxs, axis=1)
    # show a nicely formatted classification report
    class_report = classification_report(testY.argmax(axis=1), predIdxs,
                                target_names=lb.classes_)
    print(class_report)
    # compute the confusion matrix and and use it to derive the raw
    # accuracy, sensitivity, and specificity
    cm = confusion_matrix(testY.argmax(axis=1), predIdxs)

    # plot the training loss and accuracy
    N = EPOCHS
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title(f"Model: {model_name} Training Loss and Accuracy on COVID-19 Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(os.path.sep.join(['.', 'model_performance', f'{model_name}-{dataset_dir_name}-plot.png']))
    # serialize the model to disk
    print("[INFO] saving COVID-19 detector model...")
    model.save(saved_model_path, save_format="h5")

    return [cm, class_report, model_name]

def train_covid_models(dataset_dir, models=None):
    """

    :param dataset_dir: root path to a directory with folders named for the target labels.
    :type dataset_dir:
    :param models:
    :type models:
    :return: List of List of [accuracy, sensitivity, specificity, model name]
    :rtype:
    """

    # grab the list of images in our dataset directory, then initialize
    # the list of data (i.e., images) and class images
    print("[INFO] loading images...")
    imagePaths = list(paths.list_images(dataset_dir))
    data = []
    labels = []

    # loop over the image paths
    for imagePath in imagePaths:
        # extract the class label from the filename
        label = imagePath.split(os.path.sep)[-2]

        # load the image, swap color channels, and resize it to be a fixed
        # 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # update the data and labels lists, respectively
        data.append(image)
        labels.append(label)

    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 255]
    data = np.array(data) / 255.0
    labels = np.array(labels)
    # print(f"Labels: {labels}")

    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    # print(f"Labels: {labels}")
    print(f"Class Labels: {lb.classes_}")

    # partition the data into training and testing splits using 80% of
    # the data for training and the remaining 20% for testing
    (trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                      test_size=0.20, stratify=labels, random_state=42)

    if models is None:
        MODELS = [
            {
                "base_model": VGG16(weights="imagenet", include_top=False,
                                    input_tensor=Input(shape=(224, 224, 3))),
                "name": "vgg16"
            },
            {
                "base_model": VGG19(weights="imagenet", include_top=False,
                                    input_tensor=Input(shape=(224, 224, 3))),
                "name": "vgg19"
            },
            # {
            #     "base_model": ResNet50(weights="imagenet", include_top=False,
            #                            input_tensor=Input(shape=(224, 224, 3))),
            #     "name": "resnet50"
            #
            # },
            # {
            #     "base_model": ResNet50V2(weights="imagenet", include_top=False,
            #                            input_tensor=Input(shape=(224, 224, 3))),
            #     "name": "resnet50v2"
            #
            # }
        ]
    else:
        MODELS = models

    all_model_run_results = []
    dataset_dir_name = dataset_dir.split("/")[-1]
    for model in MODELS:
        print("---------------------------------------------------")
        print(f"Running Model: {model['name']}")
        start = time.time()
        model_results = run_model(model['base_model'], model['name'], dataset_dir_name, trainX, testX, trainY, testY, lb)
        end = time.time()
        all_model_run_results.append(model_results)
        print(f"Finished Model: {model['name']} took {(end-start)/60} minutes")

    return all_model_run_results


if __name__ == '__main__':
    print('Running Train COVID19 Models')
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=False, default='./dataset/0402',
                    help="path to input dataset")
    ap.add_argument("--epochs", required=False, type=int, default=25,
                    help="Number of training epochs")
    args = vars(ap.parse_args())

    dataset_dir = args["dataset"]
    EPOCHS = args['epochs']

    print(f"Using dataset directory: {dataset_dir}")

    results = train_covid_models(dataset_dir)
    print("------------  Summary -------------")
    for result in results:
        print(f"\n--------- Model: {result[2]} -------------")
        print("Classification Report:")
        print(result[1])
        print("Confusion Matrix")
        df_cm = pd.DataFrame(
            result[0], index=['covid', 'normal', 'pneumonia'], columns=['covid', 'normal', 'pneumonia'],
        )
        print(df_cm.head())


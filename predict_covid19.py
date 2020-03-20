from tensorflow.keras.models import load_model
import argparse
from imutils import paths
import cv2
import numpy as np


def predict_with_model(model_path, dataset_path ):
    model = load_model(model_path)

    imagePaths = list(paths.list_images(dataset_path))
    data = []

    # loop over the image paths
    for imagePath in imagePaths:

        # load the image, swap color channels, and resize it to be a fixed
        # 224x224 pixels while ignoring aspect ratio
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))

        # update the data and labels lists, respectively
        data.append(image)

    # convert the data and labels to NumPy arrays while scaling the pixel
    # intensities to the range [0, 255]
    data = np.array(data) / 255.0

    # predictions [COVID, NORMAL]
    predictions = model.predict(data, batch_size=8)
    predictions = np.round(predictions, 1)
    normal_label = dataset_path.split("/")[-1]
    print(f"[COVID, {normal_label}]")
    print(predictions)
    total_images = len(imagePaths)
    total_normal = 0
    for prediction in predictions:
        if prediction[1] >= 0.5:
            total_normal += 1

    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"{normal_label} Accuracy: {total_normal/total_images}")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="full path to model")
    ap.add_argument("--dataset-negative", required=True, help="path to dataset of images not COVID-19")

    args = vars(ap.parse_args())
    model_path = args['model']
    dataset_path = args['dataset_negative']

    predict_with_model(model_path, dataset_path)



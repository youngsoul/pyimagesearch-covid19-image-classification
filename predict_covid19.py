from tensorflow.keras.models import load_model
import argparse
from imutils import paths
import cv2
import numpy as np

"""
--model
./original-keras-covid-19/covid19_0318.h5
--dataset-negative
/Volumes/MacBackup/kaggle-chest-x-ray-images/chest_xray/test/PNEUMONIA
"""

def predict_with_model(model_path, dataset_path, expected_label_index, expected_label):
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
    prediction_indexes = np.argmax(predictions, axis=1)
    covid_count = list(prediction_indexes).count(0)
    normal_count = list(prediction_indexes).count(1)
    pneumonia_count = list(prediction_indexes).count(2)
    print(f"Counts (covid,normal,pneumonia) when dataset is: {expected_label}: {covid_count}, {normal_count}, {pneumonia_count}")

    covid_percent = np.round(covid_count/len(predictions), 2)
    normal_percent = np.round(normal_count/len(predictions), 2)
    pneumonia_percent = np.round(pneumonia_count/len(predictions), 2)

    predictions = np.round(predictions, 1)
    print(f"[COVID, NORMAL, PNEUMONIA]")
    print(predictions)

    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    if expected_label_index == 0:
        acc = covid_percent
    elif expected_label_index == 1:
        acc = normal_percent
    elif expected_label_index == 2:
        acc = pneumonia_percent
    else:
        acc = -1
    print(f"{expected_label} Accuracy: {acc}")
    print(f"COVID-19 %: {covid_percent}")
    print(f"Normal %: {normal_percent}")
    print(f"Pneumonia %: {pneumonia_percent}")
    print(f"NON-COVID-19 %: {normal_percent+pneumonia_percent}")


if __name__ == '__main__':
    labels = ['covid', 'normal', 'pneumonia']

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=False, default='./models/best-vgg16-0402-model.h5', help="full path to model")
    ap.add_argument("--dataset", required=False, default='/Volumes/MacBackup/kaggle-chest-x-ray-images/chest_xray/test/NORMAL', help="path to dataset of images to test against")
    ap.add_argument("--expected-label", required=False, default='normal', help="one of: covid, normal, pneumonia")

    args = vars(ap.parse_args())
    model_path = args['model']
    dataset_path = args['dataset']
    expected_label_index = labels.index(args['expected_label'])

    predict_with_model(model_path, dataset_path, expected_label_index, args['expected_label'])



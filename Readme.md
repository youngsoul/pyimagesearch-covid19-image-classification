# PyimageSearch Keras Covid-19 Project

This work is based on the blog post here:

https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/

## COVID-19 Chest XRay Dataset

https://github.com/ieee8023/covid-chestxray-dataset

## Kaggle Chest XRay Dataset

https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

## Stanford ML Group Chexpert Chest XRay Dataset

https://stanfordmlgroup.github.io/competitions/chexpert/

## Models

The approach taken in the [PyImageSearch blog](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/) was to use transfer learning.

The VGG16 model was used for the Convolutional Layers it was trained with, but the top fully connected layers were removed.

The Convolutional Layers are used for feature extraction and the weights learned in those layers are not changed.  The new fully connected layer is trained and its weights are updated to learn the specifics of images with and without COVID-19.
 

### VGG16

```text
              precision    recall  f1-score   support

       covid       0.93      1.00      0.96        13
      normal       1.00      0.93      0.96        14

    accuracy                           0.96        27
   macro avg       0.96      0.96      0.96        27
weighted avg       0.97      0.96      0.96        27

[[13  0]
 [ 1 13]]
acc: 0.9630
sensitivity: 1.0000
specificity: 0.9286
[INFO] saving COVID-19 detector model...
```
### VGG19

```text
              precision    recall  f1-score   support

       covid       0.92      0.85      0.88        13
      normal       0.87      0.93      0.90        14

    accuracy                           0.89        27
   macro avg       0.89      0.89      0.89        27
weighted avg       0.89      0.89      0.89        27

[[11  2]
 [ 1 13]]
acc: 0.8889
sensitivity: 0.8462
sensitivity, aka recall or true positive rate indicates that when
    the model says the patient was positive for COVID-19, how often was it correct.  This does not
    mean that it found all of the COVID-19 cases, just that when it predicted COVID-19 how accurate was it.
specificity: 0.9286
specificity, aka precision or true negative, measures the proportion of actual negatives that are correctly
    identified.  The percentage of healthy people who are correctly identified as NOT having the
    condition. If this is not 1.0, then the model falsely said a patient
    was healthy, and free from COVID-19, when in fact they did have COVID-19.
[INFO] saving COVID-19 detector model...
Finished Model: vgg19

```


### ResNet50

This one performed very poorly.

```text
              precision    recall  f1-score   support

       covid       0.48      1.00      0.65        13
      normal       0.00      0.00      0.00        14

    accuracy                           0.48        27
   macro avg       0.24      0.50      0.33        27
weighted avg       0.23      0.48      0.31        27

[[13  0]
 [14  0]]
acc: 0.4815
sensitivity: 1.0000
specificity: 0.0000
[INFO] saving COVID-19 detector model...
Finished Model: resnet50

```

### ResNet50V2

```text
              precision    recall  f1-score   support

       covid       1.00      0.92      0.96        13
      normal       0.93      1.00      0.97        14

    accuracy                           0.96        27
   macro avg       0.97      0.96      0.96        27
weighted avg       0.97      0.96      0.96        27

[[12  1]
 [ 0 14]]
acc: 0.9630
sensitivity: 0.9231
sensitivity, aka recall or true positive rate indicates that when
    the model says the patient was positive for COVID-19, how often was it correct.  This does not
    mean that it found all of the COVID-19 cases, just that when it predicted COVID-19 how accurate was it.
specificity: 1.0000
specificity, aka precision or false negative indicates when the model indicated the patient
    was healthy, how often was it correct.  If this is not 1.0, then the model falsely said a patient
    was healthy, and free from COVID-19, when in fact they did have COVID-19.
[INFO] saving COVID-19 detector model...
Finished Model: resnet50v2

```

## Kaggle Chest XRay Images

### ResNet50V2
#### Test - NORMAL Accuracy

Using the NORMAL test images, we would expect our classifier to classify the chest XRays as normal.

```text
Normal Accuracy: 0.9871794871794872
```

#### Test - PNEUMONIA Accuracy

Using the PNEUMONIA test images, we would expecct our classifier to know these are NOT COVID and our classifier would consider them NORMAL.

```text
Normal/PNEUMONIA Accuracy: 0.9846153846153847
```

### VGG16
#### Test - NORMAL Accuracy

Using the NORMAL test images, we would expect our classifier to classify the chest XRays as normal.

```text
Normal Accuracy: 0.8076923076923077
```

#### Test - PNEUMONIA Accuracy

Using the PNEUMONIA test images, we would expecct our classifier to know these are NOT COVID and our classifier would consider them NORMAL.

```text
Normal Accuracy: 0.6333333333333333
```

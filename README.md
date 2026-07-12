# CatDogs: Dog and Cat CNN Detector

CatDogs is a simple web application that uses a custom TensorFlow CNN model to classify an uploaded image as either a dog or a cat.

## Online Demo

Try the live app here:
[https://catdog.averxon.com/](https://catdog.averxon.com/)
or here
[https://catdogs-vmahp.ondigitalocean.app/](https://catdogs-vmahp.ondigitalocean.app/)

## Project Overview

This project was created as a beginner-friendly machine learning web app. The goal is to show students how a CNN can be trained from scratch, saved, deployed, and used in a real web page.

The app allows a user to:

- Upload a JPG or PNG image.
- Run the image through a TensorFlow CNN model.
- Display whether the model predicts `Cat` or `Dog`.
- Show the prediction confidence.
- Show cat and dog probabilities.
- View model accuracy and F1 scores.
- View the confusion matrix.
- View the model layer and weight summaries.
- Click training result images to view them in a larger modal preview.

Uploaded images are saved with a prediction-based filename, for example:

```text
cat_74point85percent_jul121314.jpg
dog_57point93percent_jul121326.png
```

## Model Details

The model is a custom CNN built with TensorFlow/Keras. It does not use a pretrained model.

Input:

```text
128 x 128 RGB image
```

Architecture summary:

```text
Input Layer
Data Augmentation
Conv2D hidden layer 1
MaxPooling2D
Conv2D hidden layer 2
MaxPooling2D
Conv2D hidden layer 3
MaxPooling2D
Conv2D hidden layer 4
MaxPooling2D
Flatten
Dense hidden layer
Dropout
Sigmoid output layer
```

Classes:

```text
Cat = 0
Dog = 1
```

Saved model weights:

```text
dog_cat_tensorflow_model.weights.h5
```

## Current Model Metrics

The online app displays saved metrics from the latest local training run.

```text
Test Accuracy: 67.50%
Cat F1: 0.6632
Dog F1: 0.6860
Macro F1: 0.6746
Weighted F1: 0.6746
```

Confusion matrix:

```text
Actual Cat: 64 predicted Cat, 36 predicted Dog
Actual Dog: 29 predicted Cat, 71 predicted Dog
```

## Repository Structure

```text
.
├── app.py
├── web_app.py
├── dog_cat_tensorflow_model.weights.h5
├── requirements.txt
├── Procfile
├── passenger_wsgi.py
├── .python-version
├── templates/
│   └── index.html
├── static/
│   ├── styles.css
│   └── uploads/
├── tensorflow_actual_model_diagram.png
├── tensorflow_training_history.png
├── tensorflow_confusion_matrix.png
├── tensorflow_sample_prediction.png
└── README.md
```

## Important Files

- `app.py` - main Flask app used for deployment.
- `web_app.py` - duplicate of the Flask app kept for project clarity.
- `requirements.txt` - Python dependencies.
- `Procfile` - tells DigitalOcean to run the app using Gunicorn.
- `.python-version` - pins Python to `3.11.9` so TensorFlow can install correctly.
- `templates/index.html` - main web page.
- `static/styles.css` - styling.
- `static/uploads/` - uploaded images are saved here.
- `dog_cat_tensorflow_model.weights.h5` - trained CNN model weights.

## Local Setup

Create and activate a Python virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the app:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5001
```

## DigitalOcean Deployment

This app is deployed on DigitalOcean App Platform.

You can use this DigitalOcean referral link for possible credits or discounts:

```text
https://m.do.co/c/7df005da8dcb
```

Recommended settings:

```text
Build command: pip install -r requirements.txt
Run command: gunicorn app:app
Python version: 3.11.9
```

The `Procfile` contains:

```text
web: gunicorn app:app
```

## Notes

The training and test images are not included in this deployment repository to keep the online app lightweight.

TensorFlow is a large dependency. Some shared hosting providers do not support it. DigitalOcean App Platform works when Python is pinned to a TensorFlow-compatible version.

## Educational Purpose

This project is intended for teaching:

- CNN image classification
- TensorFlow/Keras model loading
- Flask web deployment
- Model evaluation metrics
- Upload handling
- Basic production deployment with GitHub and DigitalOcean

# Potato Leaf Disease Detection

A Machine Learning and Deep Learning based application for detecting potato leaf diseases from images.

## Features

* Upload potato leaf images
* Detect Early Blight, Late Blight, and Healthy leaves
* Display prediction confidence
* Simple web interface using Streamlit

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Streamlit
* Pillow

## Dataset

PlantVillage Potato Leaf Disease Dataset

Classes:

* Potato Early Blight
* Potato Late Blight
* Potato Healthy

## Project Structure

```text
PotatoLeafDiseaseDetection
│
├── dataset
├── model
├── app.py
├── train.py
├── potato_model.h5
├── requirements.txt
├── runtime.txt
└── README.md
```

## Installation

1. Clone the repository

```bash
git clone https://github.com/Deepuunani/PotatoLeafDiseaseDetection.git
```

2. Navigate to the project folder

```bash
cd PotatoLeafDiseaseDetection
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Model Training

To train the model again:

```bash
python train.py
```

## Output

The application predicts:

* Early Blight
* Late Blight
* Healthy

along with the prediction confidence score.

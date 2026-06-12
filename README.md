# ML4IoT - Intelligent Light Control Classifier

A machine learning project that classifies natural language commands for IoT light control using scikit-learn. This project supports both **English** and **Swahili** commands.

## Overview

This repository contains a text classification pipeline designed to work within an IoT ecosystem. It allows users to control lights connected to an ESP32 through a web dashboard by typing natural language commands.

### System Architecture

1.  **Web Dashboard**: User interface for entering commands.
2.  **API**: Receives text commands, loads the pre-trained model, and returns a prediction (`ON` or `OFF`).
3.  **Hardware (ESP32)**: Receives the action from the dashboard/API to switch the physical light.

## Project Structure

*   `classifier4IoT.ipynb`: The main development notebook containing data exploration, model training, and serialization.
*   `data.csv`: Training dataset containing command-action pairs in English and Swahili.
*   `model.joblib`: Trained Random Forest Classifier.
*   `vectorizer.joblib`: TF-IDF Vectorizer for text processing.
*   `encoder.joblib`: Label encoder for mapping `ON`/`OFF` to numeric values.

## Getting Started

### Prerequisites

*   Python 3.8+
*   Libraries: `pandas`, `scikit-learn`, `joblib`, `notebook`

```bash
pip install pandas scikit-learn joblib notebook
```

### Usage

#### Training
To retrain the model with new data in `data.csv`, run the `classifier4IoT.ipynb` notebook.

#### Inference
You can load the trained model in your API as follows:

```python
import joblib

# Load the saved components
classifier = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
encoder = joblib.load('encoder.joblib')

# Make a prediction
message = "washa taa"
vect = vectorizer.transform([message])
prediction = classifier.predict(vect)
action = encoder.inverse_transform(prediction)[0]

print(f"Action: {action}") # Output: ON
```



# Project: Classifier4IoT

## Project Overview
This project is a machine learning-based text classifier designed for an IoT system. It classifies natural language commands (e.g., "turn on the lights", "zima taa") into binary outputs (`ON`, `OFF`) to control hardware via an ESP32.

### System Architecture
1. **Web Dashboard**: Users input natural language commands.
2. **API**: Receives requests from the dashboard, loads the serialized model components, and performs inference.
3. **Control Loop**: The prediction is sent back to the dashboard, which then triggers the corresponding action on an **ESP32** microcontroller to control connected lights.

### Main Technologies
- **Python**: Core programming language for model training and API logic.
- **scikit-learn**: Used for text vectorization (`TfidfVectorizer`), label encoding (`LabelEncoder`), and classification (`RandomForestClassifier`).
- **pandas**: Used for data manipulation.
- **joblib**: Used for serializing and deserializing the trained model and its components.

## Project Structure
- `classifier4IoT.ipynb`: Jupyter notebook containing the full pipeline for data loading, preprocessing, model training, evaluation, and serialization.
- `data.csv`: The training dataset containing command messages and their corresponding output labels.
- `model.joblib`: The serialized `RandomForestClassifier` model.
- `vectorizer.joblib`: The serialized `TfidfVectorizer` used for text feature extraction.
- `encoder.joblib`: The serialized `LabelEncoder` used for target label mapping.

## Building and Running

### Prerequisites
Ensure you have Python installed along with the following libraries:
```bash
pip install pandas scikit-learn joblib notebook
```

### Retraining the Model
1. Open `classifier4IoT.ipynb` in a Jupyter environment.
2. Run all cells to process the data in `data.csv`, train the model, and export the updated `.joblib` files.

### Inference
To use the model for inference, load the joblib files and transform your input message:
```python
import joblib

# Load components
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')
encoder = joblib.load('encoder.joblib')

# Predict
message = "turn on the lights"
vect = vectorizer.transform([message])
pred = model.predict(vect)
label = encoder.inverse_transform(pred)[0]
print(f"Command: {message} -> Action: {label}")
```

## Development Conventions
- **Language Support**: The dataset includes English and Swahili commands. New data should maintain this diversity if required.
- **Model Pipeline**: The current pipeline uses TF-IDF vectorization followed by a Random Forest Classifier.
- **Version Control**: The `.joblib` files are included in the repository to allow for immediate inference without retraining.

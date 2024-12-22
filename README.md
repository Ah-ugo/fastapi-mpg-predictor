# Auto MPG Prediction API

This repository contains a FastAPI-based application for predicting the miles per gallon (MPG) of a car using a machine learning model. The model is trained on the Auto MPG dataset, which includes features such as the number of cylinders, displacement, horsepower, weight, acceleration, model year, and origin.

## Features
- **Dataset Handling**: Automatically downloads, cleans, and preprocesses the Auto MPG dataset.
- **Data Normalization**: Implements custom normalization using dataset statistics.
- **Model Training**: Builds and trains a TensorFlow-based neural network for regression.
- **Prediction Endpoint**: Provides an HTTP POST `/predict/` endpoint to make MPG predictions based on input features.

## Technologies Used
- **FastAPI**: For building the RESTful API.
- **TensorFlow/Keras**: For constructing and training the neural network.
- **Pandas & NumPy**: For data manipulation and preprocessing.
- **Pydantic**: For request validation.

## Setup Instructions

### Prerequisites
- Python 3.8 or later
- A virtual environment is recommended

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Ah-ugo/fastapi-mpg-predictor.git
   cd fastapi-mpg-predictor
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   uvicorn main:app --reload
   ```
4. Access the API documentation at:
   - [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) (Swagger UI)
   - [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc) (ReDoc)

## Usage

### Predict MPG
Send a POST request to `/predict/` with the following JSON payload:
```json
{
  "cylinders": 4,
  "displacement": 140.0,
  "horsepower": 90.0,
  "weight": 2264.0,
  "acceleration": 15.5,
  "model_year": 76,
  "origin": 1
}
```

Example response:
```json
{
  "prediction": [23.5]
}
```

### Model Training
The TensorFlow model is trained on the Auto MPG dataset using an 80-20 train-test split. Normalization is applied to ensure consistent scaling of input features. You can modify the training parameters in the code if needed.

## File Structure
- **main.py**: Contains the FastAPI app and prediction logic.
- **requirements.txt**: Lists the required Python dependencies.
- **auto_mpg.csv**: Dataset file.

## Dataset
The Auto MPG dataset is sourced from the UCI Machine Learning Repository. It contains data on various car attributes and their MPG.

- Dataset URL: [Auto MPG Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data)

- Built by **Ahuekwe Prince** for **Austin Diamond's undergraduate Computer Engineering final year project**.



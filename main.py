import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from tensorflow import keras

app = FastAPI()

class PredictionInput(BaseModel):
    cylinders: int
    displacement: float = Field(..., ge=0, le=10000)
    horsepower: float = Field(..., ge=0, le=2000)
    weight: float = Field(..., ge=0, le=10000)
    acceleration: float = Field(..., ge=0, le=100)
    model_year: int
    origin: int  # 1=USA, 2=Europe, 3=Japan


def load_data(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The dataset file {file_path} does not exist.")
    data = pd.read_csv(file_path)
    return data

def handle_missing_values(dataset):
    dataset.replace("?", np.nan, inplace=True)
    dataset = dataset.apply(pd.to_numeric, errors="coerce")
    dataset.fillna(dataset.mean(), inplace=True)
    return dataset

def preprocess_dataset(dataset):
    dataset = dataset.drop(columns=["car_name"])  # Drop unnecessary column
    origin = dataset.pop("origin")
    dataset["USA"] = (origin == 1) * 1.0
    dataset["Europe"] = (origin == 2) * 1.0
    dataset["Japan"] = (origin == 3) * 1.0
    return dataset

def split_dataset(dataset, train_ratio):
    train_dataset = dataset.sample(frac=train_ratio, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    return train_dataset, test_dataset

def get_dataset_description(dataset):
    return dataset.describe().transpose()

def norm(x, mean, std):
    std = np.where(std == 0, 1e-6, std)  # Prevent division by zero
    return (x - mean) / std

def build_model(input_shape):
    model = keras.Sequential([
        keras.layers.InputLayer(input_shape=input_shape),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])
    return model

# Load and preprocess the dataset
dataset = load_data("auto_mpg.csv")
dataset = handle_missing_values(dataset)
dataset = preprocess_dataset(dataset)

train_dataset, test_dataset = split_dataset(dataset, 0.8)
train_labels = train_dataset.pop("mpg")
test_labels = test_dataset.pop("mpg")

train_stats = get_dataset_description(train_dataset)

normed_train_data = norm(train_dataset, train_stats["mean"], train_stats["std"])
normed_test_data = norm(test_dataset, train_stats["mean"], train_stats["std"])

model = build_model((train_dataset.shape[1],))
model.fit(normed_train_data, train_labels, epochs=50, validation_split=0.2, verbose=1)

@app.post("/predict/")
async def predict(input_data: PredictionInput):
    input_array = np.array([[
        input_data.cylinders,
        input_data.displacement,
        input_data.horsepower,
        input_data.weight,
        input_data.acceleration,
        input_data.model_year,
        (input_data.origin == 1) * 1.0,
        (input_data.origin == 2) * 1.0,
        (input_data.origin == 3) * 1.0,
    ]], dtype=np.float64)

    input_df = pd.DataFrame(input_array, columns=train_dataset.columns)

    normed_input = norm(input_df, train_stats["mean"], train_stats["std"])
    prediction = model.predict(normed_input).flatten()
    return {"prediction": prediction.tolist()}

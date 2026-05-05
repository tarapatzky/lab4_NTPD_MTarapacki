from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

# Inicjalizacja aplikacji FastAPI
app = FastAPI(title="ML Model API", description="API do serwowania modelu klasyfikacji Iris")

# Trenowanie uproszczonego modelu w pamięci przy starcie serwera
iris = load_iris()
X, y = iris.data, iris.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)


# Definicja wymaganego schematu danych wejściowych (walidacja Pydantic)
class IrisData(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Zadanie 1: Główny endpoint
@app.get("/")
def read_root():
    return {"message": "Witaj w API modelu ML! Przejdź do /docs aby zobaczyć dokumentację."}


# Zadanie 4: Endpoint info (informacje o modelu)
@app.get("/info")
def get_info():
    return {
        "model_type": "LogisticRegression",
        "features_count": X.shape[1],
        "dataset": "Iris"
    }


# Zadanie 4: Endpoint health (status serwera)
@app.get("/health")
def health_check():
    return {"status": "ok"}


# Zadania 2 i 3: Endpoint predykcji z automatyczną walidacją błędów
@app.post("/predict")
def predict(data: IrisData):
    try:
        # Konwersja odebranych danych JSON na tablicę numpy
        features = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        # Wykonanie predykcji
        prediction = model.predict(features)
        predicted_class = int(prediction[0])

        return {
            "prediction": predicted_class,
            "class_name": iris.target_names[predicted_class]
        }
    except Exception as e:
        # Awaryjna obsługa innych błędów
        raise HTTPException(status_code=400, detail=str(e))
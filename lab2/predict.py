import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")

run_id = "a8312d785c9a4d29af58e7e451aa92fc"
model_uri = f"runs:/{run_id}/model"

# Wczytanie modelu z MLflow
loaded_model = mlflow.sklearn.load_model(model_uri)

# Przykładowa próbka danych ze zbioru Wine (powinna dać klasę 0)
sample = [[14.23, 1.71, 2.43, 15.6, 127.0, 2.8, 3.06, 0.28, 2.29, 5.64, 1.04, 3.92, 1065.0]]

# Wyświetlamy przewidywane wartości i prawdopodobieństwa
prediction = loaded_model.predict(sample)
probabilities = loaded_model.predict_proba(sample)

print(f"Przewidywana klasa: {prediction[0]}")
print(f"Prawdopodobieństwa poszczególnych klas: {probabilities[0]}")
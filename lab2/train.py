import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")

wine = load_wine()
X = wine.data
y = wine.target

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Zadanie 3: Pętla z różnymi hiperparametrami
n_estimators_list = [10, 50, 100]

for n_estimators in n_estimators_list:
    with mlflow.start_run():
        # Inicjalizacja modelu
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

        # Trenowanie
        model.fit(X_train, y_train)

        # Predykcje i metryka
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Logowanie do MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"Zakończono: n_estimators={n_estimators}, Accuracy={acc:.4f}")
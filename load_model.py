import joblib

loaded_model = joblib.load('model_v1.joblib')
sample_record = [[5.1, 3.5, 1.4, 0.2]]
prediction = loaded_model.predict(sample_record)

print(f"Predykcja dla [5.1, 3.5, 1.4, 0.2]: {prediction[0]}")
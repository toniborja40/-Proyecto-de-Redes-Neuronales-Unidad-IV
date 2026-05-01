import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report


df = pd.read_csv("dataset_rutas_logisticas.csv")

X = df[["distancia_km", "trafico", "tiempo_min", "combustible_litros"]]
y = df["ruta_optima"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

adaline = SGDClassifier(
    loss="squared_error",
    max_iter=1000,
    random_state=42
)

adaline.fit(X_train, y_train)

pred_adaline = adaline.predict(X_test)

print("Resultados del modelo ADALINE")
print("Precisión:", accuracy_score(y_test, pred_adaline))
print(classification_report(y_test, pred_adaline))
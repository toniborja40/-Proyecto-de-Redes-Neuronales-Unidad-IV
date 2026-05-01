import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv("dataset_rutas_logisticas.csv")

X = df[["distancia_km", "trafico", "tiempo_min", "combustible_litros"]]
y = df["ruta_optima"]

encoder = LabelEncoder()
y = encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp = Sequential()

mlp.add(Dense(16, activation="relu", input_shape=(4,)))
mlp.add(Dense(8, activation="relu"))
mlp.add(Dense(3, activation="softmax"))

mlp.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = mlp.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test, y_test)
)

loss, accuracy = mlp.evaluate(X_test, y_test)

y_pred_prob = mlp.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)

print("Resultados del modelo MLP")
print("Precisión:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))
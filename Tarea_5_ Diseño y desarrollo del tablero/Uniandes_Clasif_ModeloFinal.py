# Importar librerías necesarias
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Cargar y preparar los datos
data = pd.read_csv('DatosSingapore.csv')  # Reemplazar con el archivo limpio
data.dropna(inplace=True)  # Eliminar datos faltantes

# Crear la variable objetivo para clasificación
data['recommended'] = np.where(
    (data['price'] <= 200) & 
    (data['review_scores_rating'] >= 4.5) & 
    (data['bedrooms'] >= 1) &
    (data['amenities_number'] >= 5) &
    (data['host_response_rate'] >= 0.79), 1, 0
)

# Separar características (X) y variable objetivo (y)
X = data.drop(columns=['recommended'])
y = data['recommended']

# Codificar variables categóricas
X = pd.get_dummies(X, drop_first=True)

# Normalizar las variables numéricas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Crear el modelo de clasificación
model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Activación sigmoid para clasificación binaria
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_split=0.2, verbose=1)

# Evaluar el modelo
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype("int32")

# Reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_test, y_pred_classes))

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Recomendado', 'Recomendado'], yticklabels=['No Recomendado', 'Recomendado'])
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.title('Matriz de confusión')
plt.show()

# Calcular métricas adicionales
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes)
recall = recall_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes)
roc_auc = roc_auc_score(y_test, y_pred)

# Mostrar métricas adicionales
print("Métricas adicionales:")
print(f"Exactitud (Accuracy): {accuracy:.2f}")
print(f"Precisión (Precision): {precision:.2f}")
print(f"Cobertura (Recall): {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")
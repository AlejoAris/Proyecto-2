import pandas as pd
import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Cargar datos originales
data = pd.read_csv('DatosSingapore.csv')

# Ver qué columnas comienzan con "property_type:" o "room_type:"
property_cols = [col for col in data.columns if col.startswith('property_type:')]
room_cols = [col for col in data.columns if col.startswith('room_type:')]

print("=== PROPERTY_TYPE COLUMNS ===")
print(f"Total: {len(property_cols)}")
for col in property_cols:
    print(f"  - {col}")

print("\n=== ROOM_TYPE COLUMNS ===")
print(f"Total: {len(room_cols)}")
for col in room_cols:
    print(f"  - {col}")

# Cargar scaler para obtener características numéricas
scaler = joblib.load("scaler_regresion.pkl")
num_features = list(scaler.feature_names_in_)

print(f"\n=== CARACTERÍSTICAS NUMÉRICAS ===")
print(f"Total: {len(num_features)}")
for feat in num_features:
    print(f"  - {feat}")

# Calcular total
total = len(num_features) + len(property_cols) + len(room_cols)
print(f"\n=== TOTAL ===")
print(f"Numéricas: {len(num_features)}")
print(f"Property type: {len(property_cols)}")
print(f"Room type: {len(room_cols)}")
print(f"TOTAL: {total}")

# Guardar esta información en un archivo para usar en la app
config = {
    'num_features': num_features,
    'property_columns': property_cols,
    'room_columns': room_cols,
}

import json
with open('feature_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print("\n✓ Configuración guardada en feature_config.json")

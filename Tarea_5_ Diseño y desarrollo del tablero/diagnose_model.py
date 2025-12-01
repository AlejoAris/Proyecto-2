import joblib
import numpy as np
from tensorflow.keras.models import load_model

# Cargar modelo y scaler
try:
    modelo = load_model("modelo_regresion_uniandes.h5")
    print("✓ Modelo cargado exitosamente")
    print(f"  Forma de entrada esperada: {modelo.input_shape}")
except Exception as e:
    print(f"✗ Error al cargar modelo: {e}")

try:
    scaler = joblib.load("scaler_regresion.pkl")
    print("✓ Scaler cargado exitosamente")
    print(f"  Características del scaler: {len(scaler.feature_names_in_)}")
    print(f"  Nombres: {list(scaler.feature_names_in_)}")
except Exception as e:
    print(f"✗ Error al cargar scaler: {e}")

# Intentar cargar datos de entrenamiento si existen
try:
    data_train = joblib.load("X_train_scaled.pkl")
    print(f"✓ X_train encontrado: shape {data_train.shape}")
except:
    pass

# Mostrar diferencia
print(f"\nSÍNTESIS:")
print(f"  Modelo espera: 57 características")
print(f"  Scaler tiene: {len(scaler.feature_names_in_)} características numéricas")
print(f"  Dummies actuales: 8 (property_type) + 3 (room_type) = 11")
print(f"  Total enviado: {len(scaler.feature_names_in_)} + 11 = {len(scaler.feature_names_in_) + 11}")
print(f"  FALTAN: {57 - (len(scaler.feature_names_in_) + 11)} características")

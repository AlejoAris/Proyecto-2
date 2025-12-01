import pandas as pd
import joblib

# Cargar datos para ver qué categorías existen
data = pd.read_csv('DatosSingapore.csv')

# Ver columnas disponibles
print("Columnas disponibles:")
print(data.columns.tolist())

# Ver valores únicos en property_type y room_type
print("\n=== PROPERTY_TYPE ===")
print(f"Valores únicos: {data['property_type'].unique()}")
print(f"Count: {len(data['property_type'].unique())}")

print("\n=== ROOM_TYPE ===")
print(f"Valores únicos: {data['room_type'].unique()}")
print(f"Count: {len(data['room_type'].unique())}")

# Crear dummies para ver cuántas se generan
cat_prefixes = ["property_type", "room_type"]
cat_df = pd.get_dummies(data[cat_prefixes].astype(str), prefix=cat_prefixes)

print(f"\n=== DUMMIES CREADAS ===")
print(f"Total de columnas dummy: {cat_df.shape[1]}")
print(f"Nombres de columnas dummy:")
for col in cat_df.columns:
    print(f"  - {col}")

# Cargar scaler para ver cuántas características numéricas hay
scaler = joblib.load("scaler_regresion.pkl")
num_features = list(scaler.feature_names_in_)
print(f"\n=== CARACTERÍSTICAS NUMÉRICAS ===")
print(f"Total: {len(num_features)}")
print(f"Nombres:")
for feat in num_features:
    print(f"  - {feat}")

# Calcular total esperado
total_expected = len(num_features) + cat_df.shape[1]
print(f"\n=== TOTAL ESPERADO ===")
print(f"Numéricas: {len(num_features)}")
print(f"Dummies: {cat_df.shape[1]}")
print(f"Total: {total_expected}")

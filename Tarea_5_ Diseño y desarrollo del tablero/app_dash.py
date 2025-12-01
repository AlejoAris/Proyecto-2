# Importar librerías necesarias
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ==================== CARGAR MODELO UNIANDES REGRESIÓN ====================
modelo_regresion_uniandes = load_model("modelo_regresion_uniandes.h5")
scaler_uniandes = joblib.load("scaler_regresion.pkl")

# Cargar configuración de features
with open('feature_config.json', 'r') as f:
    feature_config = json.load(f)

num_features = feature_config['num_features']
property_columns = feature_config['property_columns']
room_columns = feature_config['room_columns']

print(f"Configuración cargada:")
print(f"  Características numéricas: {len(num_features)}")
print(f"  Columnas property_type: {len(property_columns)}")
print(f"  Columnas room_type: {len(room_columns)}")
print(f"  Total esperado: {len(num_features) + len(property_columns) + len(room_columns)}")

# ==================== CARGAR MODELO TEC REGRESIÓN ====================
# Cargar datos y entrenar modelo TEC Regresión (USA TODOS LOS FEATURES CON DUMMIES)
data_tec_reg = pd.read_csv("DatosSingapore.csv")
data_tec_reg = data_tec_reg.drop(['id', 'host_id'], axis=1)

X_tec_reg = data_tec_reg.drop('price', axis=1)
y_tec_reg = data_tec_reg["price"]

# Entrenar modelo TEC (sin scaling)
model_tec_reg = LinearRegression()
model_tec_reg.fit(X_tec_reg, y_tec_reg)
feature_cols_tec_reg = X_tec_reg.columns.tolist()

print(f"Modelo TEC Regresión cargado con {len(feature_cols_tec_reg)} features")

# ==================== CARGAR MODELO TEC CLASIFICACIÓN ====================
# Cargar datos y entrenar modelo TEC Clasificación
data_tec_clf = pd.read_csv('DatosSingapore.csv')
features_tec_clf = ["price","accommodates","bedrooms","bathrooms","amenities_number","review_scores_rating"]

X_tec_clf = data_tec_clf[features_tec_clf]
y_tec_clf = data_tec_clf['review_scores_rating'] >= 4.5  # Clasificar por rating

X_tec_clf_scaled = StandardScaler().fit_transform(X_tec_clf)
model_tec_clf = LogisticRegression(random_state=42)
model_tec_clf.fit(X_tec_clf_scaled, y_tec_clf)
scaler_tec_clf = StandardScaler()
scaler_tec_clf.fit(X_tec_clf)

print(f"Modelo TEC Clasificación cargado con {len(features_tec_clf)} features")

# ==================== CARGAR MODELO UNIANDES CLASIFICACIÓN ====================
# Cargar datos y entrenar modelo Uniandes Clasificación
try:
    model_uniandes_clf = load_model("../Tarea_4_ Modelamiento Clasificacion/modelo_final_clasificacion.h5")
    print("Modelo Uniandes Clasificación cargado desde Tarea_4")
except:
    # Si no existe, entrenar un modelo simple
    print("Modelo Uniandes Clasificación no encontrado, entrenando modelo simple...")
    data_uniandes_clf = pd.read_csv('DatosSingapore.csv')
    data_uniandes_clf.dropna(inplace=True)
    
    # Crear variable objetivo
    data_uniandes_clf['recommended'] = np.where(
        (data_uniandes_clf['price'] <= 200) & 
        (data_uniandes_clf['review_scores_rating'] >= 4.5), 1, 0
    )
    
    X_uniandes_clf = data_uniandes_clf.drop(columns=['recommended', 'price'], errors='ignore')
    X_uniandes_clf = pd.get_dummies(X_uniandes_clf, drop_first=True)
    y_uniandes_clf = data_uniandes_clf['recommended']
    
    scaler_uniandes_clf = StandardScaler()
    X_uniandes_clf_scaled = scaler_uniandes_clf.fit_transform(X_uniandes_clf)
    
    # Crear modelo simple
    model_uniandes_clf = Sequential([
        Dense(64, activation='relu', input_dim=X_uniandes_clf_scaled.shape[1]),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model_uniandes_clf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model_uniandes_clf.fit(X_uniandes_clf_scaled, y_uniandes_clf, epochs=10, verbose=0)
    
    feature_cols_uniandes_clf = X_uniandes_clf.columns.tolist()
    print(f"Modelo Uniandes Clasificación entrenado con {len(feature_cols_uniandes_clf)} features")

data_uniandes_clf = pd.read_csv('DatosSingapore.csv')
data_uniandes_clf.dropna(inplace=True)
data_uniandes_clf['recommended'] = np.where(
    (data_uniandes_clf['price'] <= 200) & 
    (data_uniandes_clf['review_scores_rating'] >= 4.5), 1, 0
)
X_uniandes_clf = data_uniandes_clf.drop(columns=['recommended', 'price'], errors='ignore')
X_uniandes_clf = pd.get_dummies(X_uniandes_clf, drop_first=True)
scaler_uniandes_clf = StandardScaler()
scaler_uniandes_clf.fit(X_uniandes_clf)
feature_cols_uniandes_clf = X_uniandes_clf.columns.tolist()

# Inicializar la app de Dash con Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Dashboard de Modelos"

# Layout de la aplicación
app.layout = dbc.Container(
    [
        html.H1("Dashboard de Modelos de Predicción", className="text-center mb-4"),
        dcc.Tabs([
            # ======================== PESTAÑA 1: UNIANDES REGRESIÓN ========================
            dcc.Tab(label="Regresión Uniandes", children=[
                dbc.Row([
                        dbc.Col(html.H3("Predicción de Precio - Modelo Uniandes"), width=12),
                        dbc.Col([
                            dbc.Label("Capacidad de alojamiento:"),
                            dbc.Input(id='uniandes-reg-accommodates', type='number', placeholder='Ejemplo: 4', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Baños privados:"),
                            dbc.Input(id='uniandes-reg-private-bathrooms', type='number', placeholder='Ejemplo: 2', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Baños compartidos:"),
                            dbc.Input(id='uniandes-reg-shared-bathrooms', type='number', placeholder='Ejemplo: 1', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Habitaciones:"),
                            dbc.Input(id='uniandes-reg-bedrooms', type='number', placeholder='Ejemplo: 2', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Camas:"),
                            dbc.Input(id='uniandes-reg-beds', type='number', placeholder='Ejemplo: 3', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Estancia mínima (noches):"),
                            dbc.Input(id='uniandes-reg-minimum-nights', type='number', placeholder='Ejemplo: 2', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Rating (0-5):"),
                            dbc.Input(id='uniandes-reg-rating', type='number', placeholder='Ejemplo: 4.7', min=0, max=5, step=0.1),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Comunicación (0-5):"),
                            dbc.Input(id='uniandes-reg-communication', type='number', placeholder='Ejemplo: 4.5', min=0, max=5, step=0.1),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Tiempo respuesta (horas):"),
                            dbc.Input(id='uniandes-reg-response-time', type='number', placeholder='Ejemplo: 1', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Tasa aceptación (0-1):"),
                            dbc.Input(id='uniandes-reg-acceptance-rate', type='number', placeholder='Ejemplo: 0.9', min=0, max=1, step=0.01),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Tipo de Propiedad:"),
                            dcc.Dropdown(id='uniandes-reg-property-type', 
                                options=[{'label': col.replace('property_type:', ''), 'value': col} for col in property_columns],
                                placeholder="Seleccione tipo"),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Tipo de Habitación:"),
                            dcc.Dropdown(id='uniandes-reg-room-type',
                                options=[{'label': col.replace('room_type:', ''), 'value': col} for col in room_columns],
                                placeholder="Seleccione tipo"),
                        ], width=6),
                        dbc.Col(dbc.Button('Predecir', id='uniandes-reg-predict-button', n_clicks=0, color="primary", className="mt-3"),
                            width=12, className="text-center"),
                        dbc.Col(html.Div(id='uniandes-reg-output', className="text-center text-success mt-3", style={'font-size': '20px'}),
                            width=12),
                    ], className="mb-4")
            ]),
            
            # ======================== PESTAÑA 2: TEC REGRESIÓN ========================
            dcc.Tab(label="Regresión TEC", children=[
                dbc.Row([
                        dbc.Col(html.H3("Predicción de Precio - Modelo TEC"), width=12),
                        dbc.Col(html.P("Este modelo utiliza todas las características disponibles en el dataset (sin dummies de property_type ni room_type).", className="text-muted"), width=12),
                        dbc.Col([
                            dbc.Label("Accommodates:"),
                            dbc.Input(id='tec-reg-accommodates', type='number', placeholder='Ejemplo: 4', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Bathrooms:"),
                            dbc.Input(id='tec-reg-bathrooms', type='number', placeholder='Ejemplo: 1', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Bedrooms:"),
                            dbc.Input(id='tec-reg-bedrooms', type='number', placeholder='Ejemplo: 2', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Beds:"),
                            dbc.Input(id='tec-reg-beds', type='number', placeholder='Ejemplo: 3', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Minimum Nights:"),
                            dbc.Input(id='tec-reg-minimum-nights', type='number', placeholder='Ejemplo: 2', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Rating (0-5):"),
                            dbc.Input(id='tec-reg-rating', type='number', placeholder='Ejemplo: 4.7', min=0, max=5, step=0.1),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Comunicación — Rating (0-5):"),
                            dbc.Input(id='tec-reg-communication', type='number', placeholder='Ejemplo: 4.5', min=0, max=5, step=0.1),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Ubicación — Rating (0-5):"),
                            dbc.Input(id='tec-reg-location', type='number', placeholder='Ejemplo: 4.5', min=0, max=5, step=0.1),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Tiempo respuesta (horas):"),
                            dbc.Input(id='tec-reg-response-time', type='number', placeholder='Ejemplo: 1', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Tasa aceptación (0-1):"),
                            dbc.Input(id='tec-reg-acceptance-rate', type='number', placeholder='Ejemplo: 0.9', min=0, max=1, step=0.01),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Host Response Rate (0-1):"),
                            dbc.Input(id='tec-reg-host-response', type='number', placeholder='Ejemplo: 0.9', min=0, max=1, step=0.01),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Tipo de Propiedad:"),
                            dcc.Dropdown(id='tec-reg-property-type', 
                                options=[{'label': col.replace('property_type:', ''), 'value': col} for col in property_columns],
                                placeholder="Seleccione tipo"),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Tipo de Habitación:"),
                            dcc.Dropdown(id='tec-reg-room-type',
                                options=[{'label': col.replace('room_type:', ''), 'value': col} for col in room_columns],
                                placeholder="Seleccione tipo"),
                        ], width=6),
                        dbc.Col(dbc.Button('Predecir', id='tec-reg-predict-button', n_clicks=0, color="primary", className="mt-3"),
                            width=12, className="text-center"),
                        dbc.Col(html.Div(id='tec-reg-output', className="text-center text-success mt-3", style={'font-size': '20px'}),
                            width=12),
                    ], className="mb-4")
            ]),
            
            # ======================== PESTAÑA 3: TEC CLASIFICACIÓN ========================
            dcc.Tab(label="Clasificación TEC", children=[
                dbc.Row([
                        dbc.Col(html.H3("Clasificación de Propiedad Recomendada - Modelo TEC"), width=12),
                        dbc.Col([
                            dbc.Label("Price (USD):"),
                            dbc.Input(id='tec-clf-price', type='number', placeholder='Ejemplo: 150', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Accommodates:"),
                            dbc.Input(id='tec-clf-accommodates', type='number', placeholder='Ejemplo: 4', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Bedrooms:"),
                            dbc.Input(id='tec-clf-bedrooms', type='number', placeholder='Ejemplo: 2', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Bathrooms:"),
                            dbc.Input(id='tec-clf-bathrooms', type='number', placeholder='Ejemplo: 1', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Amenities Count:"),
                            dbc.Input(id='tec-clf-amenities', type='number', placeholder='Ejemplo: 20', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Review Score Rating (0-5):"),
                            dbc.Input(id='tec-clf-rating', type='number', placeholder='Ejemplo: 4.5', min=0, max=5, step=0.1),
                        ], width=6),
                        dbc.Col(dbc.Button('Clasificar', id='tec-clf-predict-button', n_clicks=0, color="primary", className="mt-3"),
                            width=12, className="text-center"),
                        dbc.Col(html.Div(id='tec-clf-output', className="text-center text-success mt-3", style={'font-size': '20px'}),
                            width=12),
                    ], className="mb-4")
            ]),
            
            # ======================== PESTAÑA 4: UNIANDES CLASIFICACIÓN ========================
            dcc.Tab(label="Clasificación Uniandes", children=[
                dbc.Row([
                        dbc.Col(html.H3("Clasificación de Propiedad Recomendada - Modelo Uniandes"), width=12),
                        dbc.Col(html.P("Criterios: price ≤ $200 AND rating ≥ 4.5 AND bedrooms ≥ 1 AND amenities ≥ 5 AND host_response ≥ 0.79", className="text-muted"), width=12),
                        # Controles para ajustar criterios dinámicamente
                        dbc.Col([
                            dbc.Label("Umbral Precio (USD):"),
                            dbc.Input(id='uni-clf-price-threshold', type='number', value=200, min=0, step=1),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Umbral Rating (0-5):"),
                            dbc.Input(id='uni-clf-rating-threshold', type='number', value=4.5, min=0, max=5, step=0.1),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Umbral Bedrooms (min):"),
                            dbc.Input(id='uni-clf-bedrooms-threshold', type='number', value=1, min=0, step=1),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Umbral Amenities (min):"),
                            dbc.Input(id='uni-clf-amenities-threshold', type='number', value=5, min=0, step=1),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Umbral Host Response (0-1):"),
                            dbc.Input(id='uni-clf-host-response-threshold', type='number', value=0.79, min=0, max=1, step=0.01),
                        ], width=2),
                        dbc.Col([
                            dbc.Label("Price (USD):"),
                            dbc.Input(id='uni-clf-price', type='number', placeholder='Ejemplo: 150', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Review Score Rating (0-5):"),
                            dbc.Input(id='uni-clf-rating', type='number', placeholder='Ejemplo: 4.5', min=0, max=5, step=0.1),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Bedrooms:"),
                            dbc.Input(id='uni-clf-bedrooms', type='number', placeholder='Ejemplo: 2', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Amenities Count:"),
                            dbc.Input(id='uni-clf-amenities', type='number', placeholder='Ejemplo: 8', min=0),
                        ], width=6),
                        dbc.Col([
                            dbc.Label("Host Response Rate (0-1):"),
                            dbc.Input(id='uni-clf-host-response', type='number', placeholder='Ejemplo: 0.9', min=0, max=1, step=0.01),
                        ], width=6),
                        dbc.Col(dbc.Button('Clasificar', id='uni-clf-predict-button', n_clicks=0, color="primary", className="mt-3"),
                            width=12, className="text-center"),
                        dbc.Col(html.Div(id='uni-clf-output', className="text-center text-success mt-3", style={'font-size': '20px'}),
                            width=12),
                    ], className="mb-4")
            ]),
        ])
    ],
    fluid=True,
)

# ================= CALLBACK 1: UNIANDES REGRESIÓN =================
@app.callback(
    Output('uniandes-reg-output', 'children'),
    Input('uniandes-reg-predict-button', 'n_clicks'),
    [
        Input('uniandes-reg-accommodates', 'value'),
        Input('uniandes-reg-private-bathrooms', 'value'),
        Input('uniandes-reg-shared-bathrooms', 'value'),
        Input('uniandes-reg-bedrooms', 'value'),
        Input('uniandes-reg-beds', 'value'),
        Input('uniandes-reg-minimum-nights', 'value'),
        Input('uniandes-reg-rating', 'value'),
        Input('uniandes-reg-communication', 'value'),
        Input('uniandes-reg-response-time', 'value'),
        Input('uniandes-reg-acceptance-rate', 'value'),
        Input('uniandes-reg-property-type', 'value'),
        Input('uniandes-reg-room-type', 'value'),
    ]
)
def predict_uniandes_regression(n_clicks, accommodates, private_bathrooms, shared_bathrooms, bedrooms, beds,
                                 minimum_nights, rating, communication, response_time, acceptance_rate,
                                 property_type, room_type):
    if n_clicks == 0:
        return ""
    
    if None in [accommodates, private_bathrooms, shared_bathrooms, bedrooms, beds, minimum_nights, 
                rating, communication, response_time, acceptance_rate, property_type, room_type]:
        return "Completa todos los campos"
    
    try:
        numeric_data = {
            "accommodates": accommodates,
            "number_of_private_bathrooms": private_bathrooms,
            "number_of_shared_bathrooms": shared_bathrooms,
            "bedrooms": bedrooms,
            "beds": beds,
            "minimum_nights": minimum_nights,
            "review_scores_rating": rating,
            "review_scores_communication": communication,
            "review_scores_location": 4.5,
            "estimated_response_time_hours": response_time,
            "host_acceptance_rate": acceptance_rate,
            "host_response_rate": 1.0,
        }
        
        numeric_df = pd.DataFrame([numeric_data], columns=num_features)
        numeric_scaled = scaler_uniandes.transform(numeric_df)
        
        property_dummies = np.zeros(len(property_columns))
        property_idx = property_columns.index(property_type) if property_type in property_columns else 0
        property_dummies[property_idx] = 1
        
        room_dummies = np.zeros(len(room_columns))
        room_idx = room_columns.index(room_type) if room_type in room_columns else 0
        room_dummies[room_idx] = 1
        
        final_input = np.concatenate([numeric_scaled[0], property_dummies, room_dummies]).reshape(1, -1)
        prediction = modelo_regresion_uniandes.predict(final_input, verbose=0)
        
        return f"Precio estimado: ${round(prediction[0][0], 2)}"
    except Exception as e:
        return f"Error: {str(e)}"

# ================= CALLBACK 2: TEC REGRESIÓN =================
@app.callback(
    Output('tec-reg-output', 'children'),
    Input('tec-reg-predict-button', 'n_clicks'),
    [
        Input('tec-reg-accommodates', 'value'),
        Input('tec-reg-bathrooms', 'value'),
        Input('tec-reg-bedrooms', 'value'),
        Input('tec-reg-beds', 'value'),
        Input('tec-reg-minimum-nights', 'value'),
        Input('tec-reg-rating', 'value'),
        Input('tec-reg-communication', 'value'),
        Input('tec-reg-location', 'value'),
        Input('tec-reg-response-time', 'value'),
        Input('tec-reg-acceptance-rate', 'value'),
        Input('tec-reg-host-response', 'value'),
        Input('tec-reg-property-type', 'value'),
        Input('tec-reg-room-type', 'value'),
    ]
)
def predict_tec_regression(n_clicks, accommodates, bathrooms, bedrooms, beds, minimum_nights,
                           rating, communication, location, response_time, acceptance_rate,
                           host_response, property_type, room_type):
    if n_clicks == 0:
        return ""
    
    if None in [accommodates, bathrooms, bedrooms, beds, minimum_nights, rating, communication,
                location, response_time, acceptance_rate, host_response, property_type, room_type]:
        return "Completa todos los campos"
    
    try:
        # Crear dict con valores básicos (todos en 0)
        sample_dict = {}
        for col in feature_cols_tec_reg:
            sample_dict[col] = 0
        
        # Asignar valores numéricos
        numeric_cols = {
            'accommodates': accommodates,
            'bathrooms': bathrooms,
            'bedrooms': bedrooms,
            'beds': beds,
            'minimum_nights': minimum_nights,
            'review_scores_rating': rating,
            'review_scores_communication': communication,
            'review_scores_location': location,
            'estimated_response_time_hours': response_time,
            'host_acceptance_rate': acceptance_rate,
            'host_response_rate': host_response,
        }
        
        for col, val in numeric_cols.items():
            if col in sample_dict:
                sample_dict[col] = val
        
        # Asignar dummies de property_type y room_type
        if property_type in sample_dict:
            sample_dict[property_type] = 1
        if room_type in sample_dict:
            sample_dict[room_type] = 1
        
        # Crear DataFrame con el orden correcto de features
        sample_df = pd.DataFrame([sample_dict], columns=feature_cols_tec_reg)
        
        # Predicción
        prediction = model_tec_reg.predict(sample_df)[0]
        return f"Precio estimado: ${round(prediction, 2)}"
    except Exception as e:
        return f"Error: {str(e)}"

# ================= CALLBACK 3: TEC CLASIFICACIÓN =================
@app.callback(
    Output('tec-clf-output', 'children'),
    Input('tec-clf-predict-button', 'n_clicks'),
    [
        Input('tec-clf-price', 'value'),
        Input('tec-clf-accommodates', 'value'),
        Input('tec-clf-bedrooms', 'value'),
        Input('tec-clf-bathrooms', 'value'),
        Input('tec-clf-amenities', 'value'),
        Input('tec-clf-rating', 'value'),
    ]
)
def predict_tec_classification(n_clicks, price, accommodates, bedrooms, bathrooms, amenities, rating):
    if n_clicks == 0:
        return ""
    
    if None in [price, accommodates, bedrooms, bathrooms, amenities, rating]:
        return "Completa todos los campos"
    
    try:
        # Modelo (probabilístico)
        x_data = np.array([[price, accommodates, bedrooms, bathrooms, amenities, rating]])
        x_scaled = scaler_tec_clf.transform(x_data)
        prob = model_tec_clf.predict_proba(x_scaled)[0][1]
        model_decision = "RECOMENDADO" if prob > 0.5 else "NO RECOMENDADO"

        # Regla basada en Composite Score (explicativa)
        # Normalizamos por máximos del dataset para una comparación simple
        max_price = data_tec_clf['price'].max() if 'price' in data_tec_clf else 1.0
        max_amen = data_tec_clf['amenities_number'].max() if 'amenities_number' in data_tec_clf else 1.0
        norm_rating = rating / 5.0
        norm_price = price / max_price if max_price else 0.0
        norm_amen = amenities / max_amen if max_amen else 0.0
        composite = norm_rating - norm_price + norm_amen

        comp_series = (data_tec_clf['review_scores_rating'] / 5.0) - (data_tec_clf['price'] / max_price) + (data_tec_clf['amenities_number'] / max_amen)
        threshold = comp_series.median()
        rule_decision = "RECOMENDADO" if composite >= threshold else "NO RECOMENDADO"

        return f"Modelo: {model_decision} (Confianza: {prob*100:.1f}%) — Regla(composite={composite:.2f} threshold={threshold:.2f}): {rule_decision}"
    except Exception as e:
        return f"Error: {str(e)}"

# ================= CALLBACK 4: UNIANDES CLASIFICACIÓN =================
@app.callback(
    Output('uni-clf-output', 'children'),
    Input('uni-clf-predict-button', 'n_clicks'),
    [
        Input('uni-clf-price', 'value'),
        Input('uni-clf-rating', 'value'),
        Input('uni-clf-bedrooms', 'value'),
        Input('uni-clf-amenities', 'value'),
        Input('uni-clf-host-response', 'value'),
        # Threshold inputs (editable by user)
        Input('uni-clf-price-threshold', 'value'),
        Input('uni-clf-rating-threshold', 'value'),
        Input('uni-clf-bedrooms-threshold', 'value'),
        Input('uni-clf-amenities-threshold', 'value'),
        Input('uni-clf-host-response-threshold', 'value'),
    ]
)
def predict_uniandes_classification(n_clicks, price, rating, bedrooms, amenities, host_response,
                                    price_thr, rating_thr, bedrooms_thr, amenities_thr, host_response_thr):
    if n_clicks == 0:
        return ""
    
    if None in [price, rating, bedrooms, amenities, host_response, price_thr, rating_thr, bedrooms_thr, amenities_thr, host_response_thr]:
        return "Completa todos los campos"
    
    try:
        # Aplicar regla estricta de Uniandes
        # Aplicar regla usando umbrales provistos por el usuario
        rule_result = (
            (price <= price_thr) and
            (rating >= rating_thr) and
            (bedrooms >= bedrooms_thr) and
            (amenities >= amenities_thr) and
            (host_response >= host_response_thr)
        )
        
        decision = "✅ RECOMENDADO" if rule_result else "❌ NO RECOMENDADO"
        
        # Mostrar detalles de cada criterio
        price_ok = "✓" if price <= price_thr else "✗"
        rating_ok = "✓" if rating >= rating_thr else "✗"
        bedrooms_ok = "✓" if bedrooms >= bedrooms_thr else "✗"
        amenities_ok = "✓" if amenities >= amenities_thr else "✗"
        response_ok = "✓" if host_response >= host_response_thr else "✗"
        
        details = f"""
        {decision}
        
        Criterios usados (umbral):
        {price_ok} Precio ≤ ${price_thr}: ${price}
        {rating_ok} Rating ≥ {rating_thr}: {rating}
        {bedrooms_ok} Bedrooms ≥ {bedrooms_thr}: {bedrooms}
        {amenities_ok} Amenities ≥ {amenities_thr}: {amenities}
        {response_ok} Host Response ≥ {host_response_thr}: {host_response:.2f}
        """
        
        return details
    except Exception as e:
        return f"Error: {str(e)}"

# ================= EJECUTAR APP =================
if __name__ == "__main__":
    app.run(debug=True)
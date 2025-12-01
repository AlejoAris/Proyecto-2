# DASH APP COMPLETO PARA 3 MODELOS (Regresión, Clasificación TEC, Clasificación Uniandes)

# NOTA: Este archivo asume que tienes en la misma carpeta:
# - TEC_regresion_singapore.py
# - TEC_modelo_clasificacion.py
# - Uniandes_ClasifModeloFinal.py
# y que cada uno expone una función predict(...) documentada.

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd

# ---- IMPORTAR MODELOS ----
import TEC_regresion_singapore as regTEC
import TEC_modelo_clasificacion as clasTEC
import Uniandes_ClasifModeloFinal as clasUNI

# ---- CARGAR CSV PARA SACAR OPCIONES ----
df = pd.read_csv('DatosSingapore.csv')

cat_cols = ['room_type', 'property_type', 'bed_type', 'cancellation_policy']

dropdowns = {
    col: [{'label': v, 'value': v} for v in sorted(df[col].dropna().unique())]
    for col in cat_cols
}

# ---- CREAR APP ----
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# ---- LAYOUT ----
app.layout = dbc.Container([
    html.H1("Dashboard de Predicción Airbnb – Singapur"),
    dcc.Tabs([
        # ---------------- REGRESIÓN ----------------
        dcc.Tab(label="Regresión TEC", children=[
            html.Br(),
            html.H3("Predicción de Precio"),
            dbc.Row([
                dbc.Col([
                    html.Label("room_type"),
                    dcc.Dropdown(id="reg-room", options=dropdowns['room_type']),
                    html.Label("property_type"),
                    dcc.Dropdown(id="reg-prop", options=dropdowns['property_type']),
                    html.Label("bed_type"),
                    dcc.Dropdown(id="reg-bed", options=dropdowns['bed_type']),
                    html.Label("cancellation_policy"),
                    dcc.Dropdown(id="reg-cancel", options=dropdowns['cancellation_policy']),
                ], width=4),
                dbc.Col([
                    html.Label("accommodates"),
                    dcc.Input(id="reg-acc", type="number", value=2),
                    html.Label("bathrooms"),
                    dcc.Input(id="reg-bath", type="number", value=1),
                    html.Label("bedrooms"),
                    dcc.Input(id="reg-bedr", type="number", value=1),
                    html.Label("beds"),
                    dcc.Input(id="reg-beds", type="number", value=1),
                    html.Br(), html.Br(),
                    dbc.Button("Predecir Precio", id="btn-reg", color="primary"),
                    html.H3(id="out-reg")
                ], width=4)
            ])
        ]),

        # ---------------- CLASIFICACIÓN TEC ----------------
        dcc.Tab(label="Clasificación TEC", children=[
            html.Br(),
            html.H3("Predicción: ¿Recomendado?"),
            dbc.Row([
                dbc.Col([
                    html.Label("accommodates"), dcc.Input(id="ct-acc", type="number", value=2),
                    html.Label("bedrooms"), dcc.Input(id="ct-bed", type="number", value=1),
                    html.Label("bathrooms"), dcc.Input(id="ct-bath", type="number", value=1),
                    html.Label("beds"), dcc.Input(id="ct-beds", type="number", value=1),
                    html.Label("amenities_number"), dcc.Input(id="ct-amn", type="number", value=5),
                ], width=4),

                dbc.Col([
                    html.Label("review_scores_rating"), dcc.Input(id="ct-rate", type="number", value=4.7),
                    html.Label("host_response_rate"), dcc.Input(id="ct-resp", type="number", value=0.9),
                    html.Br(), html.Br(),
                    dbc.Button("Clasificar", id="btn-ct", color="success"),
                    html.H3(id="out-ct")
                ], width=4)
            ])
        ]),

        # ---------------- CLASIFICACIÓN UNIANDES ----------------
        dcc.Tab(label="Clasificación UNIANDES", children=[
            html.Br(),
            html.H3("Modelo Final – ¿Recomendado?"),

            dbc.Row([
                dbc.Col([
    html.Div([
        html.Label(col),
        dcc.Dropdown(id=f"uni-{col}", options=dropdowns[col])
    ]) 
    for col in cat_cols
], width=4),


                dbc.Col([
                    html.Label("accommodates"), dcc.Input(id="uni-acc", type="number", value=2),
                    html.Label("bedrooms"), dcc.Input(id="uni-bed", type="number", value=1),
                    html.Label("bathrooms"), dcc.Input(id="uni-bath", type="number", value=1),
                    html.Label("beds"), dcc.Input(id="uni-beds", type="number", value=1),
                    html.Label("price"), dcc.Input(id="uni-price", type="number", value=150),
                    html.Label("amenities_number"), dcc.Input(id="uni-amn", type="number", value=6),
                    html.Label("review_scores_rating"), dcc.Input(id="uni-rate", type="number", value=4.6),
                    html.Label("host_response_rate"), dcc.Input(id="uni-resp", type="number", value=0.8),
                    html.Br(), html.Br(),
                    dbc.Button("Clasificar", id="btn-uni", color="warning"),
                    html.H3(id="out-uni")
                ], width=4)
            ])
        ])
    ])
])

# ---- CALLBACKS ----

# --- REGRESIÓN TEC ---
@app.callback(
    Output('out-reg', 'children'),
    Input('btn-reg', 'n_clicks'),
    [State('reg-room', 'value'), State('reg-prop', 'value'), State('reg-bed', 'value'), State('reg-cancel', 'value'),
     State('reg-acc', 'value'), State('reg-bath', 'value'), State('reg-bedr', 'value'), State('reg-beds', 'value')]
)
def predict_reg(n, room, prop, bed, cancel, acc, bath, bedr, beds):
    if not n:
        return ""
    pred = regTEC.predict(room_type=room, property_type=prop, bed_type=bed,
                          cancellation_policy=cancel, accommodates=acc,
                          bathrooms=bath, bedrooms=bedr, beds=beds)
    return f"Precio estimado: ${pred:.2f} SGD"


# --- CLASIFICACIÓN TEC ---
@app.callback(
    Output('out-ct', 'children'),
    Input('btn-ct', 'n_clicks'),
    [State('ct-acc', 'value'), State('ct-bed', 'value'), State('ct-bath', 'value'), State('ct-beds', 'value'),
     State('ct-amn', 'value'), State('ct-rate', 'value'), State('ct-resp', 'value')]
)
def predict_ct(n, acc, bed, bath, beds, amn, rate, resp):
    if not n: return ""
    pred = clasTEC.predict(accommodates=acc, bedrooms=bed, bathrooms=bath,
                           beds=beds, amenities_number=amn,
                           review_scores_rating=rate, host_response_rate=resp)
    msg = "Recomendado" if pred == 1 else "No recomendado"
    return f"Resultado: {msg}"


# --- CLASIFICACIÓN UNIANDES ---
@app.callback(
    Output('out-uni', 'children'),
    Input('btn-uni', 'n_clicks'),
    [State(f"uni-{c}", 'value') for c in cat_cols] +
    [State('uni-acc', 'value'), State('uni-bed', 'value'), State('uni-bath', 'value'), State('uni-beds', 'value'),
     State('uni-price', 'value'), State('uni-amn', 'value'), State('uni-rate', 'value'), State('uni-resp', 'value')]
)
def predict_uni(n, room, prop, bedt, cancel, acc, bed, bath, beds, price, amn, rate, resp):
    if not n: return ""
    pred = clasUNI.predict(room_type=room, property_type=prop, bed_type=bedt,
                           cancellation_policy=cancel, accommodates=acc,
                           bedrooms=bed, bathrooms=bath, beds=beds, price=price,
                           amenities_number=amn, review_scores_rating=rate,
                           host_response_rate=resp)
    msg = "Recomendado" if pred == 1 else "No recomendado"
    return f"Resultado: {msg}"


# ---- MAIN ----
if __name__ == '__main__':
    app.run_server(debug=True)

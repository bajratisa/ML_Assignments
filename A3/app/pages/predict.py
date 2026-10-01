"""The A3 model, logistic regression written from scratch, predicting a price class."""

import os
import json
import pickle

import joblib
import numpy as np
import pandas as pd
import dash
from dash import html, dcc, callback, Input, Output, State

dash.register_page(__name__, path="/predict", name="Predict")

# The model folder sits at the project root, one level above this app folder.
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "..", "model"))


class ModelUnpickler(pickle.Unpickler):
    """The notebook imports the class as app.logistic_regression. Here app.py is the running
    script, so the same file is importable only as logistic_regression."""

    def find_class(self, module, name):
        if module == "app.logistic_regression":
            module = "logistic_regression"
        return super().find_class(module, name)


with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = ModelUnpickler(f).load()

preprocessor = joblib.load(os.path.join(MODEL_DIR, "preprocessor.joblib"))

with open(os.path.join(MODEL_DIR, "price_edges.json"), "r") as f:
    price_edges = json.load(f)["price_edges"]

COLUMNS     = list(preprocessor.feature_names_in_)
NUMERIC     = ["year", "km_driven", "mileage", "engine", "max_power", "seats", "owner"]
CATEGORICAL = ["brand", "fuel", "seller_type", "transmission"]

# Dropdown options are the categories the encoder learned, so they always match the model.
encoder = preprocessor.named_transformers_["categorical"].named_steps["onehot"]
CATEGORIES = {col: list(cats) for col, cats in zip(CATEGORICAL, encoder.categories_)}
BRAND_LABELS = {"Land": "Land Rover"}

# Min and max of the A3 training data (X_train), used as the limits of the number inputs.
RANGES = {
    "year":      (1983, 2020),
    "km_driven": (1000, 2360457),
    "mileage":   (9.0, 28.4),
    "engine":    (624, 3604),
    "max_power": (34.2, 282.0),
    "seats":     (2, 14),
}


def price_range(c):
    """The price range of class c, in rupees, from the training quartiles."""
    if c == 0:
        return f"up to {price_edges[0]:,.0f}"
    if c == len(price_edges):
        return f"above {price_edges[-1]:,.0f}"
    return f"{price_edges[c - 1]:,.0f} to {price_edges[c]:,.0f}"


def field(label, component):
    return html.Div([html.Label(label), component],
                    style={"marginBottom": "10px", "maxWidth": "300px"})


def number(col, placeholder, step="any"):
    low, high = RANGES[col]
    return dcc.Input(id=f"p-{col}", type="number", min=low, max=high, step=step,
                     placeholder=placeholder)


def dropdown(col):
    options = [{"label": BRAND_LABELS.get(v, v), "value": v} for v in CATEGORIES[col]]
    return dcc.Dropdown(id=f"p-{col}", options=options, placeholder="optional")


layout = html.Div([
    html.H1("Predict the price class"),

    html.P("Enter what you know about the car and press Predict. Any field can be left "
           "blank and the model fills it with the typical value from its training data."),

    html.P("The model puts the car in one of four classes, each a quarter of the training "
           "cars: " + "; ".join(f"class {c}: {price_range(c)}" for c in range(4)) +
           " rupees."),

    field("Year", number("year", "2015", step=1)),
    field("Max power (bhp)", number("max_power", "82")),
    field("Brand", dropdown("brand")),
    field("Engine (cc)", number("engine", "1197", step=1)),
    field("Km driven", number("km_driven", "45000", step=1)),
    field("Mileage (kmpl)", number("mileage", "21.4")),
    field("Fuel", dropdown("fuel")),
    field("Transmission", dropdown("transmission")),
    field("Seller type", dropdown("seller_type")),
    field("Owner", dcc.Dropdown(id="p-owner", options=[
        {"label": "First owner", "value": 1},
        {"label": "Second owner", "value": 2},
        {"label": "Third owner", "value": 3},
        {"label": "Fourth or more", "value": 4},
    ], placeholder="optional")),
    field("Seats", number("seats", "5", step=1)),

    html.Button("Predict", id="p-submit", n_clicks=0, style={"marginTop": "10px"}),
    html.Div(id="p-output", style={"marginTop": "20px"}),
])


@callback(
    Output("p-output", "children"),
    Input("p-submit", "n_clicks"),
    State("p-brand", "value"), State("p-year", "value"), State("p-km_driven", "value"),
    State("p-fuel", "value"), State("p-seller_type", "value"), State("p-transmission", "value"),
    State("p-owner", "value"), State("p-mileage", "value"), State("p-engine", "value"),
    State("p-max_power", "value"), State("p-seats", "value"),
    prevent_initial_call=True,
)
def predict(n, brand, year, km_driven, fuel, seller_type, transmission,
            owner, mileage, engine, max_power, seats):

    row = {"year": year, "km_driven": km_driven, "mileage": mileage, "engine": engine,
           "max_power": max_power, "seats": seats, "owner": owner, "brand": brand,
           "fuel": fuel, "seller_type": seller_type, "transmission": transmission}

    blank = [k for k, v in row.items() if v is None]
    outside = [k for k, (low, high) in RANGES.items()
               if row[k] is not None and not low <= row[k] <= high]
    if outside:
        return html.P("Outside the range of the training data: " + ", ".join(outside))

    df = pd.DataFrame([row], columns=COLUMNS)

    # A blank field would otherwise leave the column as text, which the imputer rejects.
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # The imputer only fills NaN. A blank dropdown arrives as None, which the encoder would
    # treat as an unknown category instead.
    for col in CATEGORICAL:
        df[col] = df[col].astype(object).where(df[col].notna(), np.nan)

    try:
        X = preprocessor.transform(df)
        c = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
    except Exception as error:
        return html.P("Could not make a prediction: " + str(error))

    labels = [f"Class {k}<br>{price_range(k)}" for k in range(len(proba))]
    figure = {
        "data": [{"type": "bar", "x": labels, "y": [float(p) for p in proba],
                  "text": [f"{p:.0%}" for p in proba], "textposition": "outside",
                  "marker": {"color": ["#1f77b4" if k == c else "#c7c7c7"
                                       for k in range(len(proba))]}}],
        "layout": {"title": {"text": "Probability of each class"},
                   "yaxis": {"range": [0, 1.1], "tickformat": ".0%"},
                   "margin": {"t": 50, "b": 60}, "height": 350},
    }

    out = [html.H2(f"Predicted class: {c}"),
           html.P(f"Price range: {price_range(c)} rupees "
                  f"(probability {proba[c]:.0%})")]
    if blank:
        out.append(html.P("Filled in with typical values: " + ", ".join(blank)))
    out.append(dcc.Graph(figure=figure, config={"displayModeBar": False}))
    return out

"""The A2 model, linear regression written from scratch."""

import os
import json
import pickle

import pandas as pd
import dash
from dash import html, dcc, callback, Input, Output, State

from predict import predict_price

dash.register_page(__name__, path="/new", name="New model")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "..", "..", "model"))

with open(os.path.join(MODEL_DIR, "a2_model.pkl"), "rb") as f:
    art = pickle.load(f)

with open(os.path.join(MODEL_DIR, "a2_metadata.json"), "r") as f:
    meta = json.load(f)

NUMERIC = ["year", "km_driven", "mileage", "engine", "max_power", "seats", "owner"]


def field(label, component):
    return html.Div([html.Label(label), component],
                    style={"marginBottom": "10px", "maxWidth": "300px"})


layout = html.Div([
    html.H1("New model: linear regression from scratch"),

    html.P("Enter what you know about the car and press Estimate. Any field can be left "
           "blank and the model fills it with the typical value from its training data."),

    html.P(f"This model explains {meta['performance']['test_r2'] * 100:.1f} percent of the "
           f"variation in price on unseen cars, and half of its estimates fall within "
           f"{meta['performance']['median_percent_error']} percent of the true value."),

    html.P("Compared with the old model, this one is 7 KB against 9.9 MB, predicts with a "
           "single matrix multiplication rather than averaging hundreds of trees, and gives "
           "one readable coefficient per feature. The old model is about one percentage "
           "point more accurate."),

    field("Year", dcc.Input(id="n-year", type="number", placeholder="2015")),
    field("Max power", dcc.Input(id="n-max_power", type="number", placeholder="82")),
    field("Brand", dcc.Dropdown(id="n-brand", options=meta["brands"], placeholder="optional")),
    field("Engine", dcc.Input(id="n-engine", type="number", placeholder="1197")),
    field("Km driven", dcc.Input(id="n-km_driven", type="number", placeholder="45000")),
    field("Mileage", dcc.Input(id="n-mileage", type="number", placeholder="21.4")),
    field("Fuel", dcc.Dropdown(id="n-fuel", options=meta["fuel_types"], placeholder="optional")),
    field("Transmission", dcc.Dropdown(id="n-transmission", options=meta["transmission_types"],
                                       placeholder="optional")),
    field("Seller type", dcc.Dropdown(id="n-seller_type", options=meta["seller_types"],
                                      placeholder="optional")),
    field("Owner", dcc.Dropdown(id="n-owner", options=[
        {"label": "First owner", "value": 1},
        {"label": "Second owner", "value": 2},
        {"label": "Third owner", "value": 3},
        {"label": "Fourth or more", "value": 4},
    ], placeholder="optional")),
    field("Seats", dcc.Input(id="n-seats", type="number", placeholder="5")),

    html.Button("Estimate", id="n-submit", n_clicks=0, style={"marginTop": "10px"}),
    html.Div(id="n-output", style={"marginTop": "20px"}),
])


@callback(
    Output("n-output", "children"),
    Input("n-submit", "n_clicks"),
    State("n-brand", "value"), State("n-year", "value"), State("n-km_driven", "value"),
    State("n-fuel", "value"), State("n-seller_type", "value"), State("n-transmission", "value"),
    State("n-owner", "value"), State("n-mileage", "value"), State("n-engine", "value"),
    State("n-max_power", "value"), State("n-seats", "value"),
    prevent_initial_call=True,
)
def estimate(n, brand, year, km_driven, fuel, seller_type, transmission,
             owner, mileage, engine, max_power, seats):

    row = {"year": year, "km_driven": km_driven, "mileage": mileage, "engine": engine,
           "max_power": max_power, "seats": seats, "owner": owner, "brand": brand,
           "fuel": fuel, "seller_type": seller_type, "transmission": transmission}

    blank = [k for k, v in row.items() if v is None]
    df = pd.DataFrame([row])

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        price = float(predict_price(df, art)[0])
    except Exception as error:
        return html.P("Could not make a prediction: " + str(error))

    out = [html.H2(f"Estimated price: {price:,.0f}")]
    if blank:
        out.append(html.P("Filled in with typical values: " + ", ".join(blank)))
    return out
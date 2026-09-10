"""The A1 Random Forest, carried over from the previous assignment."""

import os
import json
import pickle

import numpy as np
import pandas as pd
import dash
from dash import html, dcc, callback, Input, Output, State

dash.register_page(__name__, path="/old", name="Old model")

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "..", "..", "model"))

with open(os.path.join(MODEL_DIR, "car_price_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r") as f:
    meta = json.load(f)

NUMERIC = ["year", "km_driven", "mileage", "engine", "max_power", "seats", "owner"]


def field(label, component):
    return html.Div([html.Label(label), component],
                    style={"marginBottom": "10px", "maxWidth": "300px"})


layout = html.Div([
    html.H1("Old model: Random Forest"),

    html.P("Enter what you know about the car and press Estimate. Any field can be left "
           "blank and the model fills it with the typical value from its training data."),
    html.P(f"This model explains {meta['performance']['test_r2'] * 100:.1f} percent of the "
           f"variation in price on unseen cars, and half of its estimates fall within "
           f"{meta['performance']['median_percent_error']} percent of the true value."),

    field("Year", dcc.Input(id="o-year", type="number", placeholder="2015")),
    field("Max power", dcc.Input(id="o-max_power", type="number", placeholder="82")),
    field("Brand", dcc.Dropdown(id="o-brand", options=meta["brands"], placeholder="optional")),
    field("Engine", dcc.Input(id="o-engine", type="number", placeholder="1197")),
    field("Km driven", dcc.Input(id="o-km_driven", type="number", placeholder="45000")),
    field("Mileage", dcc.Input(id="o-mileage", type="number", placeholder="21.4")),
    field("Fuel", dcc.Dropdown(id="o-fuel", options=meta["fuel_types"], placeholder="optional")),
    field("Transmission", dcc.Dropdown(id="o-transmission", options=meta["transmission_types"],
                                       placeholder="optional")),
    field("Seller type", dcc.Dropdown(id="o-seller_type", options=meta["seller_types"],
                                      placeholder="optional")),
    field("Owner", dcc.Dropdown(id="o-owner", options=[
        {"label": "First owner", "value": 1},
        {"label": "Second owner", "value": 2},
        {"label": "Third owner", "value": 3},
        {"label": "Fourth or more", "value": 4},
    ], placeholder="optional")),
    field("Seats", dcc.Input(id="o-seats", type="number", placeholder="5")),

    html.Button("Estimate", id="o-submit", n_clicks=0, style={"marginTop": "10px"}),
    html.Div(id="o-output", style={"marginTop": "20px"}),
])


@callback(
    Output("o-output", "children"),
    Input("o-submit", "n_clicks"),
    State("o-brand", "value"), State("o-year", "value"), State("o-km_driven", "value"),
    State("o-fuel", "value"), State("o-seller_type", "value"), State("o-transmission", "value"),
    State("o-owner", "value"), State("o-mileage", "value"), State("o-engine", "value"),
    State("o-max_power", "value"), State("o-seats", "value"),
    prevent_initial_call=True,
)
def estimate(n, brand, year, km_driven, fuel, seller_type, transmission,
             owner, mileage, engine, max_power, seats):

    row = {"year": year, "km_driven": km_driven, "mileage": mileage, "engine": engine,
           "max_power": max_power, "seats": seats, "owner": owner, "brand": brand,
           "fuel": fuel, "seller_type": seller_type, "transmission": transmission}

    blank = [k for k, v in row.items() if v is None]
    df = pd.DataFrame([row])

    # A blank field would otherwise leave the column as text, which the imputer rejects.
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        price = float(np.exp(model.predict(df)[0]))
    except Exception as error:
        return html.P("Could not make a prediction: " + str(error))

    out = [html.H2(f"Estimated price: {price:,.0f}")]
    if blank:
        out.append(html.P("Filled in with typical values: " + ", ".join(blank)))
    return out
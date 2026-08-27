"""
A1: Predicting Car Price

A simple Dash web app that loads the model trained in the notebook and predicts
the selling price of a car from the values the user types in.
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State


# Finding the model folder. Locally it sits two folders up from this file, but inside
# Docker it is copied somewhere else, so I read the location from an environment variable
# and fall back to the local path.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "..", "..", "model"))

# Loading the model once when the app starts, not on every prediction
with open(os.path.join(MODEL_DIR, "car_price_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r") as f:
    metadata = json.load(f)

print("Model loaded:", metadata["model_type"])


# A small helper so each field is a label with its input underneath, instead of
# repeating the same three lines eleven times
def field(label, component):
    return html.Div([html.Label(label), component],
                    style={"marginBottom": "10px", "maxWidth": "300px"})


app = Dash(__name__, title="A1__Predicting_Car_Price")



app.layout = html.Div(style={"maxWidth": "600px", "margin": "40px auto",
                             "fontFamily": "sans-serif"}, children=[

    html.H1("A1__Predicting_Car_Price"),

    # Instructions, which Task 3 asks for
    html.P("Enter what you know about the car and press Estimate. "
           "You can leave any field blank and the model will fill it in with the typical "
           "value from its training data. Year and max power affect the answer most."),
    html.P(f"The model explains {metadata['performance']['test_r2'] * 100:.1f} percent of the "
           f"variation in price on cars it has never seen, and half of its estimates are "
           f"within {metadata['performance']['median_percent_error']} percent of the true value."),

    field("Year", dcc.Input(id="year", type="number", placeholder="2015")),
    field("Max power", dcc.Input(id="max_power", type="number", placeholder="82")),
    field("Brand", dcc.Dropdown(id="brand", options=metadata["brands"], placeholder="optional")),
    field("Engine", dcc.Input(id="engine", type="number", placeholder="1197")),
    field("Km driven", dcc.Input(id="km_driven", type="number", placeholder="45000")),
    field("Mileage", dcc.Input(id="mileage", type="number", placeholder="21.4")),
    field("Fuel", dcc.Dropdown(id="fuel", options=metadata["fuel_types"], placeholder="optional")),
    field("Transmission", dcc.Dropdown(id="transmission", options=metadata["transmission_types"],
                                       placeholder="optional")),
    field("Seller type", dcc.Dropdown(id="seller_type", options=metadata["seller_types"],
                                      placeholder="optional")),
    field("Owner", dcc.Dropdown(id="owner", options=[
        {"label": "First owner", "value": 1},
        {"label": "Second owner", "value": 2},
        {"label": "Third owner", "value": 3},
        {"label": "Fourth or more", "value": 4},
    ], placeholder="optional")),
    field("Seats", dcc.Input(id="seats", type="number", placeholder="5")),

    html.Button("Estimate", id="submit", n_clicks=0, style={"marginTop": "10px"}),

    # The result is printed here, below the form
    html.Div(id="output", style={"marginTop": "20px"}),
])


# The callback runs when the button is clicked. The button is an Input so it triggers the
# function, while the form fields are States so their values are passed in without the
# prediction rerunning on every keystroke.
@app.callback(
    Output("output", "children"),
    Input("submit", "n_clicks"),
    State("brand", "value"),
    State("year", "value"),
    State("km_driven", "value"),
    State("fuel", "value"),
    State("seller_type", "value"),
    State("transmission", "value"),
    State("owner", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("seats", "value"),
    prevent_initial_call=True,
)
def estimate(n_clicks, brand, year, km_driven, fuel, seller_type,
             transmission, owner, mileage, engine, max_power, seats):

    # Building a one row table with the exact column names the pipeline expects.
    # Any field the user left blank is None here, which becomes NaN below.
    row = {
        "year": year, "km_driven": km_driven, "mileage": mileage, "engine": engine,
        "max_power": max_power, "seats": seats, "owner": owner, "brand": brand,
        "fuel": fuel, "seller_type": seller_type, "transmission": transmission,
    }

    blank = [name for name, value in row.items() if value is None]

    df = pd.DataFrame([row])

    # Forcing the number columns to be numeric. A blank field would otherwise make the
    # column a text type and the imputer inside the pipeline would reject it.
    for col in ["year", "km_driven", "mileage", "engine", "max_power", "seats", "owner"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    try:
        # The model predicts the log of the price, so exp reverses it
        price = float(np.exp(model.predict(df)[0]))
    except Exception as error:
        return html.P("Could not make a prediction: " + str(error))

    result = [html.H2(f"Estimated price: {price:,.0f}")]

    if blank:
        result.append(html.P("Filled in with typical values: " + ", ".join(blank)))

    return result


if __name__ == "__main__":
    # host 0.0.0.0 is required inside Docker, because the default would only accept
    # connections from inside the container itself
    app.run(host="0.0.0.0", port=8050, debug=False)
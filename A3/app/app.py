"""
A3: Predicting Car Price

Two pages: a home page and the price class prediction.
Routing uses the Dash pages feature, so each page lives in its own file.
"""

import os
import dash
from dash import Dash, html, dcc

app = Dash(__name__, use_pages=True, title="Predicting Car Price")

nav_style = {"marginRight": "20px", "textDecoration": "none", "fontWeight": "bold"}

app.layout = html.Div(
    style={"maxWidth": "700px", "margin": "40px auto", "fontFamily": "sans-serif"},
    children=[
        html.Div([
            dcc.Link("Home", href="/", style=nav_style),
            dcc.Link("Predict", href="/predict", style=nav_style),
        ], style={"marginBottom": "30px", "paddingBottom": "10px",
                  "borderBottom": "1px solid #ccc"}),

        dash.page_container,
    ]
)

if __name__ == "__main__":
    # Host and port come from the environment so the same image runs locally and on the
    # server.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8050))
    app.run(host=host, port=port, debug=False)

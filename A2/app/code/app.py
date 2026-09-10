"""
A2: Predicting Car Price

Three pages: a home page, the old A1 model and the new A2 model.
Routing uses the Dash pages feature, so each page lives in its own file.
"""

import os
import dash
from dash import Dash, html, dcc

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(BASE_DIR, "..", "..", "model"))

app = Dash(__name__, use_pages=True, title="Predicting Car Price")

nav_style = {"marginRight": "20px", "textDecoration": "none", "fontWeight": "bold"}

app.layout = html.Div(
    style={"maxWidth": "700px", "margin": "40px auto", "fontFamily": "sans-serif"},
    children=[
        html.Div([
            dcc.Link("Home", href="/", style=nav_style),
            dcc.Link("Old model (A1)", href="/old", style=nav_style),
            dcc.Link("New model (A2)", href="/new", style=nav_style),
        ], style={"marginBottom": "30px", "paddingBottom": "10px",
                  "borderBottom": "1px solid #ccc"}),

        dash.page_container,
    ]
)

if __name__ == "__main__":
    # Host and port come from the environment so the same image runs locally and on the
    # server, where Traefik expects port 80.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8050))
    app.run(host=host, port=port, debug=False)

    
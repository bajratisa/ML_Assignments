import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

layout = html.Div([
    html.H1("Predicting Car Price"),

    html.P("Two models are available. Both estimate the selling price of a used car from "
           "details such as year, engine size and power."),

    html.H3("Old model (A1)"),
    html.P("A tuned Random Forest built with scikit-learn. Slightly more accurate, but the "
           "saved file is 9.9 MB and its reasoning cannot be read directly."),

    html.H3("New model (A2)"),
    html.P("Linear regression written from scratch and trained with gradient descent, using "
           "polynomial features. The saved file is 7 KB, about 1,400 times smaller, and it "
           "predicts with a single matrix multiplication. Every feature has one coefficient, "
           "so the effect of each input can be read off directly."),

    html.P("On accuracy the old model is ahead, explaining 89.6 percent of the variation "
           "against 88.4 percent. The gap is small. The new model is chosen where size, "
           "speed and explainability matter more than the last percentage point."),

    dcc.Link("Try the new model", href="/new"),
])
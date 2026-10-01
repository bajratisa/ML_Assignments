import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

layout = html.Div([
    html.H1("Predicting Car Price"),

    html.P("This model sorts a used car into one of four price classes from details such as "
           "year, engine size and power. Class 0 is the cheapest quarter of the training "
           "cars and class 3 the most expensive quarter."),

    html.H3("The model"),
    html.P("Multinomial logistic regression written from scratch and trained with batch "
           "gradient descent, with a small ridge penalty. The settings (learning rate 0.5, "
           "3000 iterations, lambda 0.0001) were chosen on a validation split from a grid "
           "of 16 runs tracked in MLflow."),

    html.P("On unseen test cars it picks the right class 73.2 percent of the time, with a "
           "macro f1 of 0.735. It is most reliable for the cheapest and most expensive "
           "classes, and mixes up the two middle classes more often."),

    html.P("Besides the class, the prediction page shows the probability of every class, "
           "so a car close to a class boundary is easy to spot."),

    dcc.Link("Try the model", href="/predict"),
])

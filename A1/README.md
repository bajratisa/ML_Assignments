# A1: Predicting Car Price

AT82.03 Machine Learning, Asian Institute of Technology

**Name:** Tisa Bajracharya
**Student ID:** st126686

A machine learning model that predicts the selling price of a used car from its
specifications, served through a Dash web application running in Docker.

## Result

A tuned Random Forest reached an R squared of 0.908 on a held out test set of 1366 cars
that were not used at any point during training or model selection. Half of its estimates
are within 11.8 percent of the true price and 72 percent are within 20 percent.

## Repository structure
A1/
├── A1_Predicting_Car_Price.ipynb the full analysis, from loading to inference
├── Cars.csv 
├── README.md
├── .dockerignore
├── model/
│ ├── car_price_model.pkl the trained pipeline, 9.5 MB
│ └── model_metadata.json brand lists, feature order, recorded scores
└── app/
├── Dockerfile
├── docker-compose.yaml
└── code/
├── app.py the Dash web application
└── requirements.txt pinned library versions


## Running the web application

Requires Docker Desktop. From inside the `app` folder:

```bash
docker compose up --build
```

Then open http://localhost:8050 in a browser.

To stop it:

```bash
docker compose down
```

### Running without Docker

From inside `app/code`, using Python 3.12:

```bash
pip install -r requirements.txt
python app.py
```

The library versions in `requirements.txt` must match the ones that created the model file.
A pickle saved under scikit-learn 1.8.0 will produce warnings and possibly wrong results if
loaded under a different version.

## How to use the app

Enter whatever is known about the car and press Estimate. Every field can be left blank.
Blank fields are filled in by the imputer inside the model pipeline, using the same median
values that were used during training. Year and max power are the two fields that affect
the result most, together accounting for 87 percent of the model's decision.

## What the notebook does

**Task 1, preparing the data.** All the cleaning steps from the assignment brief: owner
mapped to 1 to 5, CNG and LPG rows removed, units stripped from mileage, engine and
max_power, brand reduced to the first word, torque dropped, test drive cars removed, and
the price log transformed.

Two cleaning steps beyond the brief. I removed 1202 exact duplicate rows, because if the
same car appears on both sides of the train test split the model is tested on a car it has
already seen. I also converted impossible zero values in mileage, engine and max_power to
missing, since a car cannot have zero fuel efficiency or a zero size engine, so those are
recording errors rather than measurements.

**Exploratory analysis.** Distribution of the target before and after the log transform,
distribution of every feature, scatter plots and box plots against price, and a correlation
matrix. The log transform reduced the skewness of the price from 5.418 to minus 0.186.

**Splitting and preprocessing.** An 80 to 20 split, made before any preprocessing so that
no information from the test set could influence how the training data was prepared. All
imputation, scaling and encoding lives inside a scikit-learn Pipeline, which means it is
refitted from scratch inside every cross validation fold and cannot leak.

**Model selection.** Six algorithms compared with five fold cross validation, then the two
best tuned with grid search. Support Vector Regression scored 0.0504 and Random Forest
0.0519. Comparing them fold by fold showed the difference was smaller than the variation
between folds, so they were tied. Random Forest was chosen because it reports feature
importance directly, which the report needed, and an RBF kernel SVR does not.

**Deployment size.** The tuned model was 79 MB, which is awkward to deploy and close to
GitHub's file size limit. The grid search had already shown that 200 trees scored the same
as 400, so I tested smaller configurations and took the smallest one that was still within
the noise threshold of the best. That gave 100 trees at 9.5 MB, an 88 percent reduction for
a change in test error of 0.5 percentage points.

## Known limitations

The model is least accurate on cheap and old cars, with a median error of 17.8 percent for
the cheapest fifth of the test set against 9.3 percent for the most expensive fifth. Old
cars are priced mostly by their condition, service history and accident record, none of
which this dataset contains. All ten of the model's worst predictions were over-predictions
of cars that looked good on paper.

The data is Indian used car listings, so the model should not be assumed to transfer to
another market without retraining.

## Environment

Trained on Python 3.12.10 with pandas 3.0.5, numpy 2.4.1 and scikit-learn 1.8.0.
The Docker image pins these same versions.
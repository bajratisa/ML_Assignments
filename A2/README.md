# A2: Predicting Car Price

Linear regression written from scratch and trained with gradient descent, replacing the
scikit-learn models used in A1.

**Live site:** https://web-st126686.ml.brain.cs.ait.ac.th

The site has two pages. The old page serves the A1 Random Forest. The new page serves the
model built here.

## Final model

Normal regression on polynomial features, batch gradient descent, learning rate 0.01, zeros
initialisation, momentum 0.9, 500 epochs.

| Metric | Cross validation | Test set |
|---|---|---|
| mse | 0.0673 | 0.0644 |
| r2 | 0.8851 | 0.8841 |

## Contents

- `A2_Predicting_Car_Price.ipynb` — cleaning, the class, the experiment, and the report
- `model/` — saved weights for both models
- `app/` — Dash application and Docker files
- `experiment_results.csv` — all 144 configurations

## Running locally

    cd app
    docker compose build
    docker run --rm -p 8050:8050 tisa99/a2-predicting-car-price:latest

Then open http://127.0.0.1:8050

Note the site is reachable from the AIT network. The domain may not resolve elsewhere.
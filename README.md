# Machine Learning Assignments

Coursework for AT82.03 Machine Learning at the Asian Institute of Technology, August, 2026.

Three assignments built on the same dataset of Indian used car listings. The first uses scikit-learn, and the next two replace it with models written from scratch in NumPy. Each one is deployed as a Dash web app in Docker.

| # | Project | What it does | Result |
|---|---|---|---|
| A1 | Car Price Prediction | Cleans the data, compares six regression models with cross validation, and tunes a Random Forest. The final model was shrunk from 79 MB to 9.5 MB with almost no loss in accuracy. | R² 0.908 on test set |
| A2 | Linear Regression from Scratch | Linear regression with polynomial features, trained with gradient descent and momentum. 144 configurations compared. | R² 0.884 on test set |
| A3 | Logistic Regression from Scratch | Multinomial logistic regression with a ridge penalty, predicting which of four price classes a car falls into. Experiments tracked in MLflow, with unit tests and a CI/CD pipeline on GitHub Actions. | Macro F1 0.735 on test set |

Each folder has its own README with setup steps, results and known limitations.

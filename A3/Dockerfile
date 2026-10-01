FROM python:3.12-slim

WORKDIR /app

COPY app/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Application code, including the pages folder and the model class
COPY app/ ./app/

# The exported model, preprocessor and price edges
COPY model/ ./model/

ENV MODEL_DIR=/app/model

EXPOSE 8050

# Run as a script: python -m app.app cannot import logistic_regression when unpickling.
CMD ["python", "app/app.py"]

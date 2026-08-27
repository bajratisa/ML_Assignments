
FROM python:3.12-slim


WORKDIR /app

COPY app/code/requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copying the application code
COPY app/code/ .

# Copying the trained model.
COPY model/ ./model/

ENV MODEL_DIR=/app/model

EXPOSE 8050

CMD ["python", "app.py"]
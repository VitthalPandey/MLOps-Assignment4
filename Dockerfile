# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY model.joblib .
COPY quant_params.joblib .

CMD ["python", "src/predict.py"]

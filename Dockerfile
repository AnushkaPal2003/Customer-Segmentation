FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# MLflow folder inside container
RUN mkdir -p /app/mlruns

CMD ["python", "train.py"]
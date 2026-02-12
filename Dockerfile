# Use official Python image
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy all files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose MLflow port
EXPOSE 5000

# Default command
CMD ["mlflow", "ui", "--host", "0.0.0.0"]

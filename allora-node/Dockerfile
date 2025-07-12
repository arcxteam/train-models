# Use Python 3.10 as the base image
FROM python:3.10-slim as project_env

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip setuptools \
    && pip install --timeout=360 -r requirements.txt

# Install TensorFlow with CPU support
RUN pip install tensorflow-cpu==2.13.0

# Second stage for cleaner image
FROM project_env

COPY gunicorn_conf.py app.py config.py utils.py init_data.py /app/

# Set the entrypoint command
CMD ["gunicorn", "--conf", "/app/gunicorn_conf.py", "app:app"]

#!/bin/bash

# Enable Docker BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Build the Docker image with cache
docker build -f Dockerfile_train_models -t allora-train-model:1.0.0 .

# Run the container with volume mounts
docker run -v $(pwd)/models:/app/models -e DATABASE_PATH=/app/data/prices.db -v $(pwd)/source-data:/app/data -d allora-train-model:1.0.0
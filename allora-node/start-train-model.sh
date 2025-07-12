#!/bin/bash

# Enable Docker BuildKit for faster builds
export DOCKER_BUILDKIT=1

# Build the Docker image with cache
docker build -f Dockerfile_train_models -t allora-train-model:1.0.9 .

# Run the container with volume mounts and Tiingo environment variables
docker run \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/source-data:/app/data \
  -e TIINGO_API_TOKEN=xxa99de96faaf30fc292a9c5b785974efa85708646 \
  -e TIINGO_DATA_DIR=/app/data/tiingo_data \
  -e TIINGO_CACHE_TTL=150 \
  -d allora-train-model:1.0.9

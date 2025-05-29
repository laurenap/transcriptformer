#!/bin/bash

# Default values
DOCKER_IMAGE=""
MODEL_VARIANT=""
INPUT_FILE=""
OUTPUT_FILE=""
PRETRAINED_EMBEDDING=""
GENE_COL_NAME=""
PRECISION=""
BATCH_SIZE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --docker-image)
      DOCKER_IMAGE="$2"
      shift; shift
      ;;
    --model-variant)
      MODEL_VARIANT="$2"
      shift; shift
      ;;
    --input-file)
      INPUT_FILE="$2"
      shift; shift
      ;;
    --output-file)
      OUTPUT_FILE="$2"
      shift; shift
      ;;
    --pretrained-embedding)
      PRETRAINED_EMBEDDING="$2"
      shift; shift
      ;;
    --gene-col-name)
      GENE_COL_NAME="$2"
      shift; shift
      ;;
    --precision)
      PRECISION="$2"
      shift; shift
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift; shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Check required arguments
if [[ -z "$DOCKER_IMAGE" || -z "$MODEL_VARIANT" || -z "$INPUT_FILE" || -z "$OUTPUT_FILE" ]]; then
  echo "Missing required arguments."
  echo "Usage: $0 --docker-image <name> --model-variant <variant> --input-file <path> --output-file <path> [--pretrained-embedding <path>] [--gene-col-name <value>] [--precision <value>] [--batch-size <value>]"
  exit 1
fi

# Resolve input and output paths
INPUT_FILE_ABS=$(realpath "$INPUT_FILE")
INPUT_DIR=$(dirname "$INPUT_FILE_ABS")

# Use realpath -m to resolve possibly non-existent output file
OUTPUT_FILE_ABS=$(realpath -m "$OUTPUT_FILE")
OUTPUT_DIR=$(dirname "$OUTPUT_FILE_ABS")

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Check for GPU availability
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
  GPU_FLAG="--gpus all"
  echo "GPU detected. Running with GPU acceleration."
else
  GPU_FLAG=""
  echo "Warning: No GPU detected. Running on CPU."
fi

# Build docker run command
DOCKER_CMD=(docker run --rm -d $GPU_FLAG \
  -v "$INPUT_DIR":/data/input:ro \
  -v "$OUTPUT_DIR":/data/output \
  "$DOCKER_IMAGE" \
  --model-variant "$MODEL_VARIANT" \
  --checkpoint-path "model_data/$MODEL_VARIANT" \
  --input-file "/data/input/$(basename "$INPUT_FILE_ABS")" \
  --output-file "/data/output/$(basename "$OUTPUT_FILE_ABS")")

# Add optional arguments
if [[ -n "$PRETRAINED_EMBEDDING" ]]; then
  PRETRAINED_EMBEDDING_ABS=$(realpath "$PRETRAINED_EMBEDDING")
  PRETRAINED_EMBEDDING_DIR=$(dirname "$PRETRAINED_EMBEDDING_ABS")
  DOCKER_CMD+=(-v "$PRETRAINED_EMBEDDING_DIR":/data/embedding)
  DOCKER_CMD+=(--pretrained-embedding "/data/embedding/$(basename "$PRETRAINED_EMBEDDING_ABS")")
fi

[[ -n "$GENE_COL_NAME" ]] && DOCKER_CMD+=(--gene-col-name "$GENE_COL_NAME")
[[ -n "$PRECISION" ]] && DOCKER_CMD+=(--precision "$PRECISION")
[[ -n "$BATCH_SIZE" ]] && DOCKER_CMD+=(--batch-size "$BATCH_SIZE")

# Run the container
CONTAINER_ID=$("${DOCKER_CMD[@]}")

# Stream logs
docker logs -f "$CONTAINER_ID"

# Wait for container to finish
docker wait "$CONTAINER_ID" > /dev/null

# Remove container
docker rm "$CONTAINER_ID" > /dev/null

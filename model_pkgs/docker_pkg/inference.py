#!/usr/bin/env python3

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Configure logging to output to stdout
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run TranscriptFormer inference via CLI")
    parser.add_argument("--model-variant", required=True, help="Model variant (e.g., tf_sapiens)")
    parser.add_argument("--checkpoint-path", required=True, type=Path, help="Path to model checkpoint directory")
    parser.add_argument("--input-file", required=True, type=Path, help="Path to input .h5ad file")
    parser.add_argument("--output-file", required=True, type=Path, help="Path to output embeddings .h5ad file")
    parser.add_argument("--batch-size", type=int, help="Batch size to use")
    parser.add_argument("--gene-col-name", default="ensembl_id", help="Gene column name")
    parser.add_argument("--precision", default="16-mixed", help="Precision setting")
    parser.add_argument("--pretrained-embedding", type=Path, help="Path to pretrained embedding file")
    return parser.parse_args()


def _get_default_batch_size(model_variant) -> int:
    # Empirical defaults for Tesla T4
    if model_variant == "tf_sapiens":
        return 32
    elif model_variant == "tf_exemplar":
        return 8
    elif model_variant == "tf_metazoa":
        return 2
    return 16  # Safe fallback


def main():
    args = parse_args()

    output_path = args.output_file.parent
    output_filename = args.output_file.name
    model_variant = args.model_variant

    cmd = [
        "transcriptformer",
        "inference",
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--data-file",
        str(args.input_file),
        "--output-path",
        str(output_path),
        "--output-filename",
        output_filename,
        "--gene-col-name",
        args.gene_col_name,
        "--precision",
        args.precision,
    ]

    if args.batch_size:
        cmd.extend(["--batch-size", str(args.batch_size)])
    else:
        cmd.extend(["--batch-size", str(_get_default_batch_size(model_variant))])

    if args.pretrained_embedding:
        cmd.extend(["--pretrained-embedding", str(args.pretrained_embedding)])

    logger.info("Inference started.")

    subprocess.run(cmd, check=True)

    logger.info(f"Inference finished! Results: {args.output_file}")


if __name__ == "__main__":
    main()

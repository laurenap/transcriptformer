"""
Run inference with a packaged Transcriptformer MLflow model.

* Accepts CLI flags for a **single** input & output file.
* Converts them into a one-row DataFrame so the model can still be batched.
* Passes optional batch-wide params (`--precision`, etc.).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True, help="Path to MLflow model")
    p.add_argument("--input-file", required=True, help="Input .h5ad path")
    p.add_argument("--output-file", required=True, help="Output .h5ad path")
    # Optional knobs (must match ParamSchema names)
    p.add_argument("--gene-col-name", default="ensembl_id")
    p.add_argument("--precision", default="16-mixed")
    p.add_argument("--pretrained-embedding", default="")
    p.add_argument("--batch-size", type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Build a single-row DataFrame.  This still satisfies the column schema
    # and lets you concat more rows later for true batch calls.
    df = pd.DataFrame(
        [
            {
                "input_file": str(Path(args.input_file).expanduser()),
                "output_file": str(Path(args.output_file).expanduser()),
            }
        ]
    )

    # Batch-wide params that MLflow will validate against ParamSchema.
    params = {
        "gene_col_name": args.gene_col_name,
        "precision": args.precision,
        "pretrained_embedding": args.pretrained_embedding,
    }
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size

    model = mlflow.pyfunc.load_model(args.model_path)
    output_series = model.predict(df, params=params)  # validated call

    print("✓ Inference complete")
    print("Output written to:", output_series.iloc[0])


if __name__ == "__main__":
    main()

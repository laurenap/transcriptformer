"""
Package a Transcriptformer model checkpoint into an MLflow PythonModel.

This script saves an MLflow model using a specific model variant and checkpoint directory.
It supports environment packaging via `uv` and accepts optional pip requirements.

Example usage:

    python package.py \
        --model-variant tf_sapiens \
        --checkpoint-path /checkpoints/tf_sapiens \
        --output-dir mlflow_models \
        --requirements requirements-mlflow-pkg.txt
"""

import argparse
from pathlib import Path

import mlflow.pyfunc
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, ParamSchema, ParamSpec, Schema
from model_code.tf_mlflow_model import TranscriptformerMLflowModel


def parse_args():
    parser = argparse.ArgumentParser(description="Package a transcriptformer model as an MLflow PythonModel")
    parser.add_argument(
        "--model-variant",
        required=True,
        type=str,
        help="Name of the model variant (e.g., tf_sapiens, tf_exemplar, tf_metazoa)",
    )
    parser.add_argument("--checkpoint-path", required=True, type=Path, help="Path to the model checkpoint directory")
    parser.add_argument("--output-dir", default="mlflow_models", type=Path, help="Directory to save the MLflow model")
    parser.add_argument(
        "--requirements", default="requirements-mlflow-pkg.txt", type=Path, help="Path to pip requirements file"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_save_path = args.output_dir / f"transcriptformer_{args.model_variant}"

    # Define the input schema: a single string column for the input file path
    input_schema = Schema([ColSpec("string")])

    # Define the output schema: a single string column for the output file path
    output_schema = Schema([ColSpec("string", "output_file")])

    # Define the parameters schema with appropriate types
    params_schema = ParamSchema(
        [
            ParamSpec(name="output_file", dtype="string", default="output.h5ad"),
            ParamSpec(name="gene_col_name", dtype="string", default="ensembl_id"),
            ParamSpec(name="precision", dtype="string", default="16-mixed"),
            ParamSpec(name="pretrained_embedding", dtype="string", default=""),
            ParamSpec(name="batch_size", dtype="integer", default=16),
        ]
    )

    # Combine the schemas into a model signature
    signature = ModelSignature(inputs=input_schema, outputs=output_schema, params=params_schema)

    print(f"Packaging model variant '{args.model_variant}' with checkpoint '{args.checkpoint_path}'")
    print(f"Output path: {model_save_path}")

    mlflow.pyfunc.save_model(
        path=str(model_save_path),
        python_model=TranscriptformerMLflowModel(args.model_variant),
        artifacts={"checkpoint_path": str(args.checkpoint_path)},
        pip_requirements=str(args.requirements),
        code_paths=[str((Path(__file__).parent / "model_code").resolve())],
        signature=signature,
        metadata={"tags": {"model_variant": args.model_variant}},
    )

    print(f"Model saved successfully at: {model_save_path}")


if __name__ == "__main__":
    main()

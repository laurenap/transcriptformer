import subprocess
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd


class TranscriptformerMLflowModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_variant: str):
        self.model_variant = model_variant

    def load_context(self, context: mlflow.pyfunc.PythonModelContext) -> None:
        self.checkpoint_path = context.artifacts["checkpoint_path"]

    def _get_default_batch_size(self) -> int:
        # Empirical defaults for Tesla T4
        if self.model_variant == "tf_sapiens":
            return 32
        elif self.model_variant == "tf_exemplar":
            return 8
        elif self.model_variant == "tf_metazoa":
            return 2
        return 16  # Safe fallback

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame, params: dict[str, Any] | None = None
    ) -> pd.DataFrame:
        # Extract the file path from the DataFrame
        if model_input.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame with the input file path.")
        input_filepath = Path(model_input.iloc[0, 0])

        if not input_filepath.is_file():
            raise ValueError(f"model_input must be a valid file path: {input_filepath}")

        if not params or "output_file" not in params:
            raise ValueError("params must include 'output_file'.")

        output_file = Path(params["output_file"])
        output_path = output_file.parent
        output_filename = output_file.name

        batch_size = params.get("batch_size", self._get_default_batch_size())
        gene_col_name = params.get("gene_col_name", "ensembl_id")
        precision = params.get("precision", "16-mixed")
        pretrained_embedding = params.get("pretrained_embedding", None)

        cmd = [
            "transcriptformer",
            "inference",
            "--checkpoint-path",
            str(self.checkpoint_path),
            "--data-file",
            str(input_filepath),
            "--output-path",
            str(output_path),
            "--output-filename",
            output_filename,
            "--batch-size",
            str(batch_size),
            "--gene-col-name",
            gene_col_name,
            "--precision",
            precision,
        ]

        if pretrained_embedding:
            cmd.extend(["--pretrained-embedding", str(pretrained_embedding)])

        subprocess.run(cmd, check=True)

        return pd.DataFrame([{"output_file": str(output_file)}])

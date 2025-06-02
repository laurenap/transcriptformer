import subprocess
from pathlib import Path
from typing import Any

import mlflow


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
        self,
        context: mlflow.pyfunc.PythonModelContext,
        model_input: list[dict[str, str]],
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, str]]:
        results = []

        for input_dict in model_input:
            input_filepath = Path(input_dict["data_file"])

            if not input_filepath.is_file():
                raise ValueError(f"model_input must be a valid file path: {input_filepath}")

            if "output_file" not in input_dict:
                raise ValueError("params must include 'output_file'.")

            output_file = Path(input_dict["output_file"])
            output_path = output_file.parent
            output_filename = output_file.name

            batch_size = input_dict.get("batch_size", self._get_default_batch_size())
            gene_col_name = input_dict.get("gene_col_name", "ensembl_id")
            precision = input_dict.get("precision", "16-mixed")
            pretrained_embedding = input_dict.get("pretrained_embedding", None)

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
            results.append({"output_file": str(output_file)})

        return results

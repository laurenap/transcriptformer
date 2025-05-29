# Steps to create and test `MLflow Model` package
1. Create the following directory structure at the root of the repo
```
model_pkgs/
└── mlflow_pkg
    ├── model_code
    └── model_data
```
2. Download model weights and auxiliary data to `model-data` directory
by running the following command at the root of the repo
```
$ transcriptformer download all --checkpoint-dir model_pkgs/mlflow_pkg/model_data/
```
3. Create `tf_mlflow_model.py` that creates a custom `MLflow PythonModel` that provides an uniform interface to wrap `transcriptformer` inference interface

4. Create `package.py` to save the 3 model variants as a `pyfunc` model

5. Create `predict.py` to invoke prediction on the `pyfunc` wrapped model

6. Run `package.py` to create `MLflow Model` artifact

```
$ python model_pkgs/mlflow_pkg/package.py --model-variant tf_sapiens   --checkpoint-path model_pkgs/model_data/tf_sapiens --output-dir model_pkgs/mlflow_pkg/mlflow_models --requirements model_pkgs/mlflow_pkg/requirements-mlflow-pkg.txt
```

```
$ tree model_pkgs/mlflow_pkg/mlflow_models/

model_pkgs/mlflow_pkg/mlflow_models/
└── transcriptformer_tf_sapiens
    ├── MLmodel
    ├── artifacts
    │   └── tf_sapiens
    │       ├── config.json
    │       ├── model_weights.pt
    │       └── vocabs
    │           ├── assay_vocab.json
    │           └── homo_sapiens_gene.h5
    ├── code
    │   └── model_code
    │       ├── __init__.py
    │       └── tf_mlflow_model.py
    ├── conda.yaml
    ├── python_env.yaml
    ├── python_model.pkl
    └── requirements.txt
```

7. Run `predict.py` using the `MLflow Model` artifact

```
$ python model_pkgs/mlflow_pkg/predict.py --model-path model_pkgs/mlflow_pkg/mlflow_models/transcriptformer_tf_sapiens --input-file ~/.cz-benchmarks/datasets/tsv2_bladder.h5ad --output-file model_pkgs/mlflow_pkg/tf_results/tsv2_bladder_embeddings.h5ad --gene-col-name ensembl_id --precision 16-mixed
```

8. Explore results of inference:

```
$ python
Python 3.11.12 (main, Apr  9 2025, 04:04:00) [Clang 20.1.0 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import anndata as ad
>>> adata = ad.read_h5ad("model_pkgs/mlflow_pkg/tf_results/tsv2_bladder_embeddings.h5ad")
>>> adata.shape
(36715, 0)
>>> adata.obsm.keys()
KeysView(AxisArrays with keys: embeddings)
>>> adata.obsm['embeddings'][:10]
array([[-8.3201475e-02, -2.1799840e-01, -7.6243706e-02, ...,
        -4.7113899e-02,  9.1581464e-02, -1.6075600e-02],
       [-2.1389551e-01, -2.7368796e-01, -1.1780801e-01, ...,
        -1.4133304e-01,  5.3645898e-02, -1.6838877e-01],
       [-5.7556093e-01, -9.6078448e-02, -1.4714500e-01, ...,
        -1.4922236e-01,  1.9487566e-01, -1.0581499e+00],
       ...,
       [-7.5616086e-01, -2.6275128e-01, -8.9282475e-02, ...,
         2.7112734e-01,  2.5229374e-01, -6.3524050e-01],
       [-1.1220228e-01, -2.6371574e-01, -3.2646282e-04, ...,
        -2.9754400e-01,  2.5283948e-01, -4.2796448e-01],
       [-2.5829804e-01, -1.9586168e-01, -2.1707835e-02, ...,
        -6.4148486e-02,  2.1838287e-02, -1.0138144e-01]],
      shape=(10, 2048), dtype=float32)
```

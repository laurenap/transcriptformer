# Steps to create and run docker image

1. `$ cd model_pkgs`
2. Build docker image
```
$ DOCKER_BUILDKIT=1 docker build -f docker_pkg/Dockerfile.tf_sapiens -t transcriptformer:tf_sapiens .

```
3. View the docker image
```
$ docker images
REPOSITORY         TAG          IMAGE ID       CREATED             SIZE
transcriptformer   tf_sapiens   9d6fbc512361   About an hour ago   11.8GB
```
4. `$ chmod +x docker_run.sh`
5. Run inference

```
$ ./docker_run.sh --docker-image transcriptformer:tf_sapiens \
  --model-variant tf_sapiens \
  --input-file ~/.cz-benchmarks/datasets/tsv2_bladder.h5ad \
  --output-file docker_pkg/tf_results/tsv2_bladder_embeddings.h5ad \
  --gene-col-name ensembl_id \
  --precision 16-mixed
```
6. Explore results of inference

```
$ python
Python 3.11.12 (main, Apr  9 2025, 04:04:00) [Clang 20.1.0 ] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import anndata as ad
>>> adata = ad.read_h5ad("model_pkgs/docker_pkg/tf_results/tsv2_bladder_embeddings.h5ad")
>>> adata.shape
(36715, 0)
>>> adata.obsm.keys()
KeysView(AxisArrays with keys: embeddings)
>>> adata.obsm['embeddings'][:10]
array([[-8.32014829e-02, -2.17998400e-01, -7.62437060e-02, ...,
        -4.71138954e-02,  9.15814638e-02, -1.60756037e-02],
       [-2.13895530e-01, -2.73687929e-01, -1.17808014e-01, ...,
        -1.41333044e-01,  5.36459051e-02, -1.68388769e-01],
       [-5.75560868e-01, -9.60784480e-02, -1.47145003e-01, ...,
        -1.49222359e-01,  1.94875658e-01, -1.05814993e+00],
       ...,
       [-7.56160855e-01, -2.62751251e-01, -8.92824754e-02, ...,
         2.71127313e-01,  2.52293766e-01, -6.35240555e-01],
       [-1.12202279e-01, -2.63715744e-01, -3.26461915e-04, ...,
        -2.97544032e-01,  2.52839446e-01, -4.27964479e-01],
       [-2.58298039e-01, -1.95861682e-01, -2.17078384e-02, ...,
        -6.41484782e-02,  2.18382869e-02, -1.01381443e-01]],
      shape=(10, 2048), dtype=float32)
```

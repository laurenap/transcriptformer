# Steps to create and run docker image

1. `cd model_pkgs`
2. Build docker image
```
docker build -f docker_pkg/Dockerfile.tf_sapiens -t transcriptformer:tf_sapiens .
```
3. `chmod +x docker_run.sh`
4. Run inference

```
./docker_run.sh --docker-image transcriptformer:tf_sapiens \
  --model-variant tf_sapiens \
  --input-file ~/.cz-benchmarks/datasets/tsv2_bladder.h5ad \
  --output-file docker_pkg/tf_results/tsv2_bladder_embeddings.h5ad \
  --gene-col-name ensembl_id \
  --precision 16-mixed
```

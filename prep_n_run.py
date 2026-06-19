import os
import scanpy as sc
import numpy as np


# 1 PREP FLY CELL ATLAS DATA

input_path = "fly cell_atlas_raw.h5ad" 

print(f"-loading raw data set from{input_path}-")
adata = sc.read_h5ad(input_path)

# take random 10000 cell slice to avoid memory issues 
np.random.seed(42)
if adata.shape[0] > 10000:
    print("subsetting matrix to 10000 representative cells")
    sampled_indices = np.random.choice(adata.shape[0], 10000, replace=False)
    adata_subset = adata[sampled_indices, :]
    
else:
    adata_subset = adata.copy()

# transcriptformer expects gene names to be in a column called 'ensembl_id'
# function purpose is to map gene index names to model token's vocab

    adata_subset.var['ensembl_id'] = adata_subset.var_names

# save the subsetted data to a new file
subset_output = "fly_cell_atlas_1ksubset.h5ad"
adata_subset.write_h5ad(subset_output)
print(f"Success! Saved 1,000-cell input slice to: {subset_output}\n")

# 2 RUN TRANSCRIPTFORMER command line interference CLI 
print("Running Transcriptformer CLI on the subsetted data")
cli_command = (
    "python3 transcriptformer/src/transcriptformer/model/inference.py "
    "--config-name=inference_config.yaml "
    "model.checkpoint_path=./checkpoints/tf_exemplar "
    f"model.inference_config.data_files.0={subset_output} "
    "model.inference_config.batch_size=4 "
    "model.inference_config.output_path=./inference_results"
)

# send command to shell enviro 
os.system(cli_command)
print("Transcriptformer inference complete! Results saved to ./inference_results")
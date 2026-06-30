
import anndata
import mygene
import pandas as pd

print("Loading the heart subset data...")
adata = anndata.read_h5ad("heart_subset.h5ad")
gene_symbols = adata.var_names.tolist()

mg = mygene.MyGeneInfo()


print(f"Querying gene symbols for {len(gene_symbols)} Drosophila genes for Ensemble IDs...")
mg = mygene.MyGeneInfo()

results =mg.querymany(
    gene_symbols,
    scopes = "symbol",
    fields = "flybase, ensembl.gene",
    species = "7227", 
    as_dataframe=True

)

print(results.head())
print(results.columns)

results = mg.querymany(gene_symbols, scopes='symbol', fields='ensembl.gene', species='dmelanogaster')

mapping = {} 

for symbol, row in results.iterrows():
    fbgn = row.get('flybase')
    if isinstance(fbgn, str) and fbgn.startswith('FBgn'):
        mapping[symbol] = fbgn

print("valid mappings:", len(mapping))

adata.var["ensembl_id"] = adata.var_names.map(mapping)

print(adata.var["ensembl_id"].head(30))
print("missing:", adata.var["ensembl_id"].isna().sum())

adata = adata[:, adata.var["ensembl_id"].notna()].copy()
adata.write_h5ad("heart_subset_fixed.h5ad")


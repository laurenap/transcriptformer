
import anndata
import mygene

print("Loading the heart subset data...")
adata = anndata.read_h5ad("data/heart_subset.h5ad")

gene_symbols = adata.var_names.tolist()

print(f"Querying gene symbols for {len(gene_symbols)} Drosophila genes for Ensemble IDs...")
mg = mygene.MyGeneInfo()

results = mg.querymany(gene_symbols, scopes='symbol', fields='ensembl.gene', species='dmelanogaster')

mapping = {} 
if 'ensembl.gene' in results.columns:
    for symbol, row in results.iterrows():
        ensembl_id = row['ensembl.gene']
        if isinstance(ensembl_id, str) and ensembl_id.startswith('FBgn'):
            mapping[symbol] = ensembl_id

print(f"Mapping completed. Found {len(mapping)} valid mappings.")

new_ids = [mapping.get(symbol, symbol) for symbol in gene_symbols]
adata.var['ensembl_id'] = new_ids

adata.write_h5ad("data/heart_subset_with_ensembl.h5ad")
print("saved updated AnnData object with Ensembl IDs to 'data/heart_subset_with_ensembl.h5ad'")



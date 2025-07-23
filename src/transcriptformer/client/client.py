"""TranscriptFormer Python Client."""

import os
import json
from typing import Any, List
from dataclasses import fields, is_dataclass
from omegaconf import OmegaConf
import anndata
import logging
from transcriptformer.data.dataclasses import InferenceConfig, DataConfig
from transcriptformer.model.inference import run_inference


class TranscriptFormerClient:
    """Python client for TranscriptFormer inference and downloads."""

    def inference(
        self,
        data_file: str | anndata.AnnData,
        checkpoint_path: str,
        log_level: int = logging.ERROR,
        **kwargs
    ) -> Any:
        """Run inference with TranscriptFormer model.
        
        Args:
            data_file: Path to input AnnData file
            checkpoint_path: Path to model checkpoint directory
            **kwargs: Additional parameters that map to InferenceConfig and DataConfig
            
        Returns:
            AnnData object with embeddings and results
        """
        # Set logging level for inference
        original_level = logging.getLogger().level
        logging.getLogger().setLevel(log_level)
        
        # Split kwargs into appropriate dataclass parameters
        inference_kwargs, data_kwargs, unknown_kwargs = self._split_kwargs(kwargs)
        
        # Add required parameters
        inference_kwargs.update({
            'data_files': None,
            'checkpoint_path': checkpoint_path,
            'output_keys': ['embeddings'],  # Default output
            'obs_keys': ['all'],  # Default to return all obs
        })
        
        # Create configs with defaults
        inference_config = self._create_config(InferenceConfig, inference_kwargs)
        data_config = self._create_config(DataConfig, data_kwargs)
        
        # Create Hydra-compatible config structure
        cfg = self._create_hydra_config(
            inference_config, 
            data_config, 
            checkpoint_path,
            unknown_kwargs
        )

        result = run_inference(cfg, data_files=[data_file])

        # Restore original logging level
        logging.getLogger().setLevel(original_level)
        
        return result

    def download_model(
        self,
        model: str,
        checkpoint_dir: str = "./checkpoints"
    ) -> None:
        """Download TranscriptFormer model weights and artifacts.
        
        Args:
            model: Model to download. Options: 'tf-sapiens', 'tf-exemplar', 'tf-metazoa', 'all', 'all-embeddings'
            checkpoint_dir: Directory to store downloaded checkpoints (default: ./checkpoints)
        """
        from transcriptformer.cli.download_artifacts import download_and_extract
        
        valid_models = ["tf-sapiens", "tf-exemplar", "tf-metazoa", "all", "all-embeddings"]
        if model not in valid_models:
            raise ValueError(f"Invalid model '{model}'. Must be one of: {valid_models}")
        
        models_map = {
            "tf-sapiens": "tf_sapiens",
            "tf-exemplar": "tf_exemplar", 
            "tf-metazoa": "tf_metazoa",
            "all-embeddings": "all_embeddings",
        }
        
        if model == "all":
            # Download all models and embeddings
            for model_name in ["tf_sapiens", "tf_exemplar", "tf_metazoa", "all_embeddings"]:
                download_and_extract(model_name, checkpoint_dir)
        elif model == "all-embeddings":
            # Download only embeddings
            download_and_extract("all_embeddings", checkpoint_dir)
        else:
            download_and_extract(models_map[model], checkpoint_dir)

    def download_data(
        self,
        species: List[str],
        output_dir: str = "./data/cellxgene",
        processes: int = 4,
        max_retries: int = 5,
        save_metadata: bool = True,
        test_only: bool = False
    ) -> int:
        """Download CellxGene Discover datasets by species.
        
        Args:
            species: List of species names to download (e.g., ['homo sapiens', 'mus musculus'])
            output_dir: Directory where datasets will be saved (default: ./data/cellxgene)
            processes: Number of parallel processes for downloading (default: 4)
            max_retries: Maximum number of retry attempts per dataset (default: 5)
            save_metadata: Whether to save dataset metadata to JSON file (default: True)
            test_only: Only test API connectivity, don't download datasets (default: False)
            
        Returns:
            Number of successfully downloaded datasets
        """
        from transcriptformer.cli.download_data import main as download_data_main
        
        if not species and not test_only:
            raise ValueError("species list cannot be empty unless test_only=True")
        
        return download_data_main(
            species=species,
            output_dir=output_dir,
            n_processes=processes,
            max_retries=max_retries,
            save_metadata=save_metadata,
            test_only=test_only
        )
    

    def download_dataset(
        self,
        dataset: str,
        organism: str = None,
        tissue: str = None,
        version: str = "v2",
        path: str = None,
        force_download: bool = False,
        **kwargs
    ):
        """Download and load datasets using the unified client interface.
        
        Args:
            dataset: Dataset to download. Options: 'bgee-testis-evolution', 'tabula-sapiens', 'all-embeddings'  
            organism: For bgee-testis-evolution: 'marmoset', 'rhesus_macaque', 'human', 'chimpanzee', etc.
            tissue: For tabula-sapiens: 'lymphnode', 'heart', 'testis', etc.
            version: For tabula-sapiens: 'v1' or 'v2' (default: v2)
            path: Custom save path (optional)
            force_download: Whether to force re-download (default: False)
            **kwargs: Additional arguments passed to the dataset function
            
        Returns:
            AnnData object for bgee-testis-evolution and tabula-sapiens, None for all-embeddings
            
        Examples:
            # Download testis data for macaque
            tf.download_dataset('bgee-testis-evolution', organism='macaque')
            
            # Download heart tissue from Tabula Sapiens
            tf.download_dataset('tabula-sapiens', tissue='heart', version='v2')
            
            # Download all embeddings
            tf.download_dataset('all-embeddings')
        """
        from transcriptformer.data.datasets import bgee_testis_evolution, tabula_sapiens, download_all_embeddings
        
        if dataset == "bgee-testis-evolution":
            if not organism:
                raise ValueError("organism is required for bgee-testis-evolution dataset")
            return bgee_testis_evolution(
                organism=organism,
                path=path,
                force_download=force_download,
                **kwargs
            )
        elif dataset == "tabula-sapiens":
            if not tissue:
                raise ValueError("tissue is required for tabula-sapiens dataset")
            return tabula_sapiens(
                tissue=tissue,
                path=path,
                force_download=force_download,
                version=version,
                **kwargs
            )
        elif dataset == "all-embeddings":
            download_all_embeddings(
                path=path,
                force_download=force_download,
                **kwargs
            )
            return None
        else:
            raise ValueError(f"Unknown dataset '{dataset}'. Options: 'bgee-testis-evolution', 'tabula-sapiens', 'all-embeddings'")

    def _split_kwargs(self, kwargs):
        """Split kwargs into InferenceConfig, DataConfig, and unknown parameters."""
        inference_fields = {f.name for f in fields(InferenceConfig)}
        data_fields = {f.name for f in fields(DataConfig)}
        
        inference_kwargs = {}
        data_kwargs = {}
        unknown_kwargs = {}
        
        for key, value in kwargs.items():
            if key in inference_fields:
                inference_kwargs[key] = value
            elif key in data_fields:
                data_kwargs[key] = value
            else:
                unknown_kwargs[key] = value
                
        return inference_kwargs, data_kwargs, unknown_kwargs
    
    def _create_config(self, config_class, kwargs):
        """Create a dataclass config with defaults."""
        # Get default values for required fields
        defaults = self._get_defaults(config_class)
        defaults.update(kwargs)
        
        # Filter to only fields that exist in the dataclass
        field_names = {f.name for f in fields(config_class)}
        filtered_kwargs = {k: v for k, v in defaults.items() if k in field_names}
        
        return config_class(**filtered_kwargs)
    
    def _get_defaults(self, config_class):
        """Get reasonable defaults for required dataclass fields."""
        if config_class == InferenceConfig:
            return {
                'output_keys': ['embeddings'],
                'batch_size': 8,
                'obs_keys': ['all'],
                'data_files': None,
                'load_checkpoint': None,
            }
        elif config_class == DataConfig:
            return {
                'aux_vocab_path': '',  # Will be set automatically
                'pin_memory': True,
                'aux_cols': 'assay',  # Should be string, not list - gets split later
                'gene_col_name': 'ensembl_id',
                'clip_counts': 30,
                'filter_to_vocabs': True,
                'filter_outliers': 0.0,
                'pad_zeros': False,
                'normalize_to_scale': 0,
                'n_data_workers': 8,
                'sort_genes': False,
                'randomize_genes': False,
                'min_expressed_genes': 0,
                'gene_pad_token': '[PAD]',
                'aux_pad_token': '[PAD]',
            }
        return {}
    
    def _create_hydra_config(self, inference_config, data_config, checkpoint_path, unknown_kwargs):
        """Create Hydra-compatible configuration structure."""
        # Load model config from checkpoint
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path) as f:
            model_config = json.load(f)
        
        # Start with the saved model config as base
        cfg_dict = model_config.copy()
        
        # Override with client-provided configs
        # Merge data_config, giving preference to saved model config for essential fields
        saved_data_config = cfg_dict['model'].get('data_config', {})
        client_data_config = self._dataclass_to_dict(data_config)
        
        # Keep essential fields from saved config, override others with client config
        essential_fields = ['esm2_mappings', 'special_tokens', 'esm2_mappings_path']
        merged_data_config = client_data_config.copy()
        for field in essential_fields:
            if field in saved_data_config:
                merged_data_config[field] = saved_data_config[field]
        
        # Update the model config structure
        cfg_dict['model']['checkpoint_path'] = checkpoint_path
        cfg_dict['model']['model_type'] = inference_config.model_type
        cfg_dict['model']['inference_config'] = self._dataclass_to_dict(inference_config)
        cfg_dict['model']['data_config'] = merged_data_config
        
        # Set checkpoint paths automatically
        cfg_dict['model']['inference_config']['load_checkpoint'] = os.path.join(
            checkpoint_path, "model_weights.pt"
        )
        cfg_dict['model']['data_config']['aux_vocab_path'] = os.path.join(
            checkpoint_path, "vocabs"
        )
        cfg_dict['model']['data_config']['esm2_mappings_path'] = os.path.join(
            checkpoint_path, "vocabs"
        )
        
        # Add any unknown kwargs as overrides
        for key, value in unknown_kwargs.items():
            cfg_dict['model'][key] = value
            
        return OmegaConf.create(cfg_dict)
    
    def _dataclass_to_dict(self, obj):
        """Convert dataclass to dictionary."""
        if is_dataclass(obj):
            return {f.name: getattr(obj, f.name) for f in fields(obj)}
        return obj 
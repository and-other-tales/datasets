#!/usr/bin/env python3
"""
Dataset Manager for Legal Llama Datasets

This module provides comprehensive dataset management functionality including:
- Loading and saving datasets using Hugging Face datasets library
- Adding data to existing datasets
- Deleting datasets
- Clearing cache
- Editing dataset fields
- Managing metadata

Compliant with Hugging Face datasets documentation best practices.
"""

import os
import sys
import json
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk, disable_caching, enable_caching
import pyarrow as pa
import pyarrow.parquet as pq

# Try to import pandas, but make it optional
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("Pandas not available. Some functionality may be limited.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages datasets for Legal Llama training pipelines"""
    
    def __init__(self, base_dir: str = "generated"):
        """Initialize dataset manager
        
        Args:
            base_dir: Base directory for dataset storage
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Dataset subdirectories
        self.datasets_dir = self.base_dir / "managed_datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        
        # Metadata storage
        self.metadata_dir = self.base_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
        # Cache directory (following HF datasets conventions)
        self.cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "legal_llama"
        
    def list_datasets(self) -> List[Dict[str, Any]]:
        """List all available datasets with their metadata
        
        Returns:
            List of dataset information dictionaries
        """
        datasets = []
        
        # Check for saved datasets (arrow format)
        for dataset_path in self.datasets_dir.glob("*"):
            if dataset_path.is_dir() and (dataset_path / "dataset_info.json").exists():
                try:
                    # Load dataset info
                    info = self._load_dataset_info(dataset_path)
                    datasets.append(info)
                except Exception as e:
                    logger.warning(f"Could not load dataset info from {dataset_path}: {e}")
        
        # Check for parquet files
        for parquet_file in self.base_dir.rglob("*.parquet"):
            if parquet_file.parent != self.datasets_dir:
                try:
                    # Get basic info from parquet file
                    parquet_metadata = pq.read_metadata(parquet_file)
                    datasets.append({
                        'name': parquet_file.stem,
                        'path': str(parquet_file),
                        'format': 'parquet',
                        'num_rows': parquet_metadata.num_rows,
                        'size_bytes': parquet_file.stat().st_size,
                        'modified': datetime.fromtimestamp(parquet_file.stat().st_mtime).isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Could not read parquet metadata from {parquet_file}: {e}")
        
        # Check for JSON files
        for json_file in self.base_dir.rglob("*.json"):
            if json_file.parent != self.metadata_dir and json_file.name != "dataset_info.json":
                try:
                    # Get basic info
                    datasets.append({
                        'name': json_file.stem,
                        'path': str(json_file),
                        'format': 'json',
                        'size_bytes': json_file.stat().st_size,
                        'modified': datetime.fromtimestamp(json_file.stat().st_mtime).isoformat()
                    })
                except Exception as e:
                    logger.warning(f"Could not read JSON file info from {json_file}: {e}")
        
        return sorted(datasets, key=lambda x: x.get('modified', ''), reverse=True)
    
    def load_dataset(self, dataset_path: Union[str, Path]) -> Union[Dataset, DatasetDict]:
        """Load a dataset from disk
        
        Args:
            dataset_path: Path to the dataset
            
        Returns:
            Loaded dataset or dataset dictionary
        """
        dataset_path = Path(dataset_path)
        
        try:
            if dataset_path.is_dir() and (dataset_path / "dataset_info.json").exists():
                # Load arrow dataset
                return load_from_disk(str(dataset_path))
            elif dataset_path.suffix == '.parquet':
                # Load parquet file
                return Dataset.from_parquet(str(dataset_path))
            elif dataset_path.suffix == '.json':
                # Load JSON file
                return Dataset.from_json(str(dataset_path))
            else:
                raise ValueError(f"Unsupported dataset format: {dataset_path}")
        except Exception as e:
            logger.error(f"Failed to load dataset from {dataset_path}: {e}")
            raise
    
    def save_dataset(self, dataset: Union[Dataset, DatasetDict], name: str, 
                    metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Save a dataset to disk with metadata
        
        Args:
            dataset: Dataset to save
            name: Name for the dataset
            metadata: Optional metadata dictionary
            
        Returns:
            Path where dataset was saved
        """
        dataset_path = self.datasets_dir / name
        
        try:
            # Save dataset in arrow format (most efficient)
            dataset.save_to_disk(str(dataset_path))
            
            # Save additional metadata
            if metadata:
                metadata_path = dataset_path / "custom_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Update dataset info
            self._update_dataset_info(dataset_path, dataset, metadata)
            
            logger.info(f"Saved dataset '{name}' to {dataset_path}")
            return dataset_path
            
        except Exception as e:
            logger.error(f"Failed to save dataset '{name}': {e}")
            raise
    
    def add_to_dataset(self, dataset_name: str, new_data: Union[Dict, List[Dict], Dataset]) -> Dataset:
        """Add new data to an existing dataset
        
        Args:
            dataset_name: Name of the dataset to update
            new_data: Data to add (dict, list of dicts, or Dataset)
            
        Returns:
            Updated dataset
        """
        dataset_path = self.datasets_dir / dataset_name
        
        try:
            # Load existing dataset
            if dataset_path.exists():
                existing_dataset = self.load_dataset(dataset_path)
            else:
                raise ValueError(f"Dataset '{dataset_name}' not found")
            
            # Convert new data to Dataset if needed
            if isinstance(new_data, dict):
                new_dataset = Dataset.from_dict({k: [v] for k, v in new_data.items()})
            elif isinstance(new_data, list):
                new_dataset = Dataset.from_list(new_data)
            elif isinstance(new_data, Dataset):
                new_dataset = new_data
            else:
                raise ValueError(f"Unsupported data type: {type(new_data)}")
            
            # Ensure schemas match
            if set(existing_dataset.column_names) != set(new_dataset.column_names):
                logger.warning("Column mismatch. Attempting to align schemas...")
                # Add missing columns with None values
                for col in existing_dataset.column_names:
                    if col not in new_dataset.column_names:
                        new_dataset = new_dataset.add_column(col, [None] * len(new_dataset))
                for col in new_dataset.column_names:
                    if col not in existing_dataset.column_names:
                        existing_dataset = existing_dataset.add_column(col, [None] * len(existing_dataset))
            
            # Concatenate datasets
            from datasets import concatenate_datasets
            updated_dataset = concatenate_datasets([existing_dataset, new_dataset])
            
            # Save updated dataset
            self.save_dataset(updated_dataset, dataset_name)
            
            logger.info(f"Added {len(new_dataset)} rows to dataset '{dataset_name}'")
            return updated_dataset
            
        except Exception as e:
            logger.error(f"Failed to add data to dataset '{dataset_name}': {e}")
            raise
    
    def delete_dataset(self, dataset_name: str, confirm: bool = True) -> bool:
        """Delete a dataset
        
        Args:
            dataset_name: Name of the dataset to delete
            confirm: Whether to ask for confirmation
            
        Returns:
            True if deleted, False otherwise
        """
        dataset_path = self.datasets_dir / dataset_name
        
        if not dataset_path.exists():
            logger.warning(f"Dataset '{dataset_name}' not found")
            return False
        
        try:
            if confirm:
                response = input(f"Are you sure you want to delete dataset '{dataset_name}'? (y/N): ")
                if response.lower() != 'y':
                    logger.info("Deletion cancelled")
                    return False
            
            # Remove dataset directory
            shutil.rmtree(dataset_path)
            logger.info(f"Deleted dataset '{dataset_name}'")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete dataset '{dataset_name}': {e}")
            raise
    
    def clear_cache(self, dataset_name: Optional[str] = None) -> None:
        """Clear dataset cache
        
        Args:
            dataset_name: Specific dataset to clear cache for, or None for all
        """
        try:
            if dataset_name:
                # Clear specific dataset cache
                dataset_cache = self.cache_dir / dataset_name
                if dataset_cache.exists():
                    shutil.rmtree(dataset_cache)
                    logger.info(f"Cleared cache for dataset '{dataset_name}'")
                else:
                    logger.info(f"No cache found for dataset '{dataset_name}'")
            else:
                # Clear all cache
                if self.cache_dir.exists():
                    shutil.rmtree(self.cache_dir)
                    logger.info("Cleared all dataset cache")
                else:
                    logger.info("No cache found")
                    
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            raise
    
    def edit_dataset_field(self, dataset_name: str, field_name: str, 
                          transform_func: callable) -> Dataset:
        """Edit a field in the dataset using a transformation function
        
        Args:
            dataset_name: Name of the dataset
            field_name: Name of the field to edit
            transform_func: Function to apply to each value
            
        Returns:
            Updated dataset
        """
        dataset_path = self.datasets_dir / dataset_name
        
        try:
            # Load dataset
            dataset = self.load_dataset(dataset_path)
            
            if field_name not in dataset.column_names:
                raise ValueError(f"Field '{field_name}' not found in dataset")
            
            # Apply transformation
            dataset = dataset.map(lambda x: {field_name: transform_func(x[field_name])})
            
            # Save updated dataset
            self.save_dataset(dataset, dataset_name)
            
            logger.info(f"Updated field '{field_name}' in dataset '{dataset_name}'")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to edit dataset field: {e}")
            raise
    
    def add_dataset_field(self, dataset_name: str, field_name: str, 
                         values: Union[List, callable]) -> Dataset:
        """Add a new field to the dataset
        
        Args:
            dataset_name: Name of the dataset
            field_name: Name of the new field
            values: List of values or function to generate values
            
        Returns:
            Updated dataset
        """
        dataset_path = self.datasets_dir / dataset_name
        
        try:
            # Load dataset
            dataset = self.load_dataset(dataset_path)
            
            if field_name in dataset.column_names:
                raise ValueError(f"Field '{field_name}' already exists in dataset")
            
            # Add new column
            if callable(values):
                # Generate values using function
                new_values = [values(row) for row in dataset]
            else:
                # Use provided values
                if len(values) != len(dataset):
                    raise ValueError(f"Number of values ({len(values)}) doesn't match dataset size ({len(dataset)})")
                new_values = values
            
            dataset = dataset.add_column(field_name, new_values)
            
            # Save updated dataset
            self.save_dataset(dataset, dataset_name)
            
            logger.info(f"Added field '{field_name}' to dataset '{dataset_name}'")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to add dataset field: {e}")
            raise
    
    def remove_dataset_field(self, dataset_name: str, field_name: str) -> Dataset:
        """Remove a field from the dataset
        
        Args:
            dataset_name: Name of the dataset
            field_name: Name of the field to remove
            
        Returns:
            Updated dataset
        """
        dataset_path = self.datasets_dir / dataset_name
        
        try:
            # Load dataset
            dataset = self.load_dataset(dataset_path)
            
            if field_name not in dataset.column_names:
                raise ValueError(f"Field '{field_name}' not found in dataset")
            
            # Remove column
            dataset = dataset.remove_columns([field_name])
            
            # Save updated dataset
            self.save_dataset(dataset, dataset_name)
            
            logger.info(f"Removed field '{field_name}' from dataset '{dataset_name}'")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to remove dataset field: {e}")
            raise
    
    def get_dataset_metadata(self, dataset_name: str) -> Dict[str, Any]:
        """Get metadata for a dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Metadata dictionary
        """
        dataset_path = self.datasets_dir / dataset_name
        
        try:
            if not dataset_path.exists():
                raise ValueError(f"Dataset '{dataset_name}' not found")
            
            # Load dataset info
            info = self._load_dataset_info(dataset_path)
            
            # Load custom metadata if exists
            custom_metadata_path = dataset_path / "custom_metadata.json"
            if custom_metadata_path.exists():
                with open(custom_metadata_path, 'r') as f:
                    custom_metadata = json.load(f)
                info['custom_metadata'] = custom_metadata
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get dataset metadata: {e}")
            raise
    
    def update_dataset_metadata(self, dataset_name: str, metadata: Dict[str, Any]) -> None:
        """Update metadata for a dataset
        
        Args:
            dataset_name: Name of the dataset
            metadata: Metadata dictionary to update
        """
        dataset_path = self.datasets_dir / dataset_name
        
        try:
            if not dataset_path.exists():
                raise ValueError(f"Dataset '{dataset_name}' not found")
            
            # Load existing custom metadata
            custom_metadata_path = dataset_path / "custom_metadata.json"
            if custom_metadata_path.exists():
                with open(custom_metadata_path, 'r') as f:
                    existing_metadata = json.load(f)
            else:
                existing_metadata = {}
            
            # Update metadata
            existing_metadata.update(metadata)
            
            # Save updated metadata
            with open(custom_metadata_path, 'w') as f:
                json.dump(existing_metadata, f, indent=2)
            
            logger.info(f"Updated metadata for dataset '{dataset_name}'")
            
        except Exception as e:
            logger.error(f"Failed to update dataset metadata: {e}")
            raise
    
    def export_dataset(self, dataset_name: str, format: str, output_path: Optional[Path] = None) -> Path:
        """Export dataset to different formats
        
        Args:
            dataset_name: Name of the dataset
            format: Export format ('parquet', 'json', 'csv')
            output_path: Optional output path
            
        Returns:
            Path to exported file
        """
        dataset_path = self.datasets_dir / dataset_name
        
        try:
            # Load dataset
            dataset = self.load_dataset(dataset_path)
            
            # Determine output path
            if output_path is None:
                output_path = self.base_dir / "exports" / f"{dataset_name}.{format}"
                output_path.parent.mkdir(exist_ok=True)
            
            # Export based on format
            if format == 'parquet':
                dataset.to_parquet(str(output_path))
            elif format == 'json':
                dataset.to_json(str(output_path))
            elif format == 'csv':
                dataset.to_csv(str(output_path))
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported dataset '{dataset_name}' to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to export dataset: {e}")
            raise
    
    def _load_dataset_info(self, dataset_path: Path) -> Dict[str, Any]:
        """Load dataset information from dataset directory"""
        info = {
            'name': dataset_path.name,
            'path': str(dataset_path),
            'format': 'arrow'
        }
        
        # Load dataset_info.json if exists
        info_path = dataset_path / "dataset_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                dataset_info = json.load(f)
                info.update(dataset_info)
        
        # Get dataset stats
        try:
            dataset = load_from_disk(str(dataset_path))
            if isinstance(dataset, Dataset):
                info['num_rows'] = len(dataset)
                info['num_columns'] = len(dataset.column_names)
                info['columns'] = dataset.column_names
                info['features'] = str(dataset.features)
            elif isinstance(dataset, DatasetDict):
                info['splits'] = list(dataset.keys())
                info['num_rows'] = {split: len(ds) for split, ds in dataset.items()}
        except:
            pass
        
        # Get modification time
        info['modified'] = datetime.fromtimestamp(dataset_path.stat().st_mtime).isoformat()
        
        return info
    
    def _update_dataset_info(self, dataset_path: Path, dataset: Union[Dataset, DatasetDict], 
                           metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update dataset information file"""
        info = {
            'name': dataset_path.name,
            'created': datetime.now().isoformat(),
            'modified': datetime.now().isoformat()
        }
        
        if isinstance(dataset, Dataset):
            info['num_rows'] = len(dataset)
            info['num_columns'] = len(dataset.column_names)
            info['columns'] = dataset.column_names
        elif isinstance(dataset, DatasetDict):
            info['splits'] = list(dataset.keys())
            info['num_rows'] = {split: len(ds) for split, ds in dataset.items()}
        
        if metadata:
            info['metadata'] = metadata
        
        # Save dataset info
        info_path = dataset_path / "dataset_info.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)


def main():
    """Example usage of DatasetManager"""
    manager = DatasetManager()
    
    # List available datasets
    print("Available datasets:")
    datasets = manager.list_datasets()
    for ds in datasets:
        print(f"  - {ds['name']} ({ds['format']}) - {ds.get('num_rows', 'unknown')} rows")


if __name__ == "__main__":
    main()
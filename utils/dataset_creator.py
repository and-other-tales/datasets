#!/usr/bin/env python3
"""
UK Legislation Dataset Creator for LLM Fine-tuning

This script creates comprehensive datasets suitable for training language models
with complete contextual awareness of UK legislation and acts of law.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Iterator, Optional, Union
import re
from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer
import subprocess
import torch

def setup_pytorch():
    """Setup PyTorch with CUDA if available, otherwise CPU"""
    def check_cuda_available():
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def install_pytorch_with_cuda():
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', 'torchaudio', 
                '--index-url', 'https://download.pytorch.org/whl/cu118'
            ], check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def install_pytorch_cpu():
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'torch', 'torchvision', 'torchaudio', 
                '--index-url', 'https://download.pytorch.org/whl/cpu'
            ], check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    try:
        # Test if torch._C is available
        import torch._C
        print("PyTorch is already properly installed")
        return True
    except ImportError:
        print("PyTorch installation issue detected, reinstalling...")
        
        if check_cuda_available():
            print("CUDA detected, installing PyTorch with CUDA support...")
            if install_pytorch_with_cuda():
                print("Successfully installed PyTorch with CUDA")
                return True
            else:
                print("CUDA installation failed, falling back to CPU")
        
        print("Installing PyTorch with CPU support...")
        if install_pytorch_cpu():
            print("Successfully installed PyTorch with CPU")
            return True
        else:
            print("Failed to install PyTorch")
            return False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_creation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UKLegislationDatasetCreator:
    def __init__(
        self, 
        source_dir: str = "uk_legislation",
        output_dir: str = "uk_legislation_datasets",
        tokenizer_name: str = "microsoft/DialoGPT-medium"
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create dataset subdirectories
        self.raw_datasets_dir = self.output_dir / "raw"
        self.processed_datasets_dir = self.output_dir / "processed"
        self.final_datasets_dir = self.output_dir / "final"
        
        for dir_path in [self.raw_datasets_dir, self.processed_datasets_dir, self.final_datasets_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize tokenizer for text processing
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.warning(f"Could not load tokenizer {tokenizer_name}: {e}")
            self.tokenizer = None
        
        # Dataset configuration
        self.max_length = 2048  # Maximum sequence length for training
        self.chunk_overlap = 100  # Overlap between chunks to maintain context
        self.min_text_length = 50  # Minimum text length to include
        
    def load_legislation_metadata(self) -> Dict[str, Dict]:
        """Load all legislation metadata"""
        logger.info("Loading legislation metadata...")
        
        metadata_dir = self.source_dir / "metadata"
        if not metadata_dir.exists():
            logger.error(f"Metadata directory not found: {metadata_dir}")
            return {}
        
        metadata = {}
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    item_metadata = json.load(f)
                    item_id = metadata_file.stem
                    metadata[item_id] = item_metadata
            except Exception as e:
                logger.warning(f"Could not load metadata from {metadata_file}: {e}")
        
        logger.info(f"Loaded metadata for {len(metadata)} legislation items")
        return metadata
    
    def create_raw_text_dataset(self) -> Dataset:
        """Create raw text dataset from downloaded legislation files"""
        logger.info("Creating raw text dataset...")
        
        text_dir = self.source_dir / "text"
        xml_dir = self.source_dir / "xml"
        metadata = self.load_legislation_metadata()
        
        raw_data = []
        
        # Process text files first
        if text_dir.exists():
            for text_file in text_dir.glob("*.txt"):
                item_id = text_file.stem
                try:
                    with open(text_file, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                    
                    if len(content) >= self.min_text_length:
                        item_metadata = metadata.get(item_id, {})
                        
                        raw_data.append({
                            'id': item_id,
                            'text': content,
                            'source': 'text',
                            'type': item_metadata.get('type', 'unknown'),
                            'title': item_metadata.get('title', ''),
                            'year': item_metadata.get('year', ''),
                            'url': item_metadata.get('url', ''),
                            'length': len(content)
                        })
                        
                except Exception as e:
                    logger.warning(f"Error processing text file {text_file}: {e}")
        
        # Process XML files for items without text files
        if xml_dir.exists():
            existing_ids = {item['id'] for item in raw_data}
            
            for xml_file in xml_dir.glob("*.xml"):
                item_id = xml_file.stem
                
                if item_id not in existing_ids:
                    try:
                        # Extract text content from XML
                        text_content = self._extract_text_from_xml(xml_file)
                        
                        if len(text_content) >= self.min_text_length:
                            item_metadata = metadata.get(item_id, {})
                            
                            raw_data.append({
                                'id': item_id,
                                'text': text_content,
                                'source': 'xml',
                                'type': item_metadata.get('type', 'unknown'),
                                'title': item_metadata.get('title', ''),
                                'year': item_metadata.get('year', ''),
                                'url': item_metadata.get('url', ''),
                                'length': len(text_content)
                            })
                            
                    except Exception as e:
                        logger.warning(f"Error processing XML file {xml_file}: {e}")
        
        logger.info(f"Created raw dataset with {len(raw_data)} items")
        
        # Create and save raw dataset
        dataset = Dataset.from_list(raw_data)
        raw_dataset_path = self.raw_datasets_dir / "legislation_raw"
        dataset.save_to_disk(str(raw_dataset_path))
        
        # Also save as parquet and JSON
        dataset.to_parquet(str(self.raw_datasets_dir / "legislation_raw.parquet"))
        dataset.to_json(str(self.raw_datasets_dir / "legislation_raw.json"))
        
        return dataset
    
    def _extract_text_from_xml(self, xml_file: Path) -> str:
        """Extract text content from XML file"""
        try:
            import xml.etree.ElementTree as ET
            
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Extract all text content, ignoring tags
            text_parts = []
            for elem in root.iter():
                if elem.text:
                    text_parts.append(elem.text.strip())
                if elem.tail:
                    text_parts.append(elem.tail.strip())
            
            # Join and clean text
            text = ' '.join(part for part in text_parts if part)
            
            # Basic cleanup
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize line breaks
            
            return text.strip()
            
        except Exception as e:
            logger.warning(f"Could not extract text from XML {xml_file}: {e}")
            return ""
    
    def create_instruction_dataset(self, raw_dataset: Dataset) -> Dataset:
        """Create instruction-following dataset for fine-tuning"""
        logger.info("Creating instruction-following dataset...")
        
        instruction_data = []
        
        for item in raw_dataset:
            # Generate various instruction formats for comprehensive understanding
            instructions = self._generate_instructions_for_item(item)
            instruction_data.extend(instructions)
        
        logger.info(f"Created instruction dataset with {len(instruction_data)} examples")
        
        dataset = Dataset.from_list(instruction_data)
        instruction_dataset_path = self.processed_datasets_dir / "legislation_instructions"
        dataset.save_to_disk(str(instruction_dataset_path))
        dataset.to_parquet(str(self.processed_datasets_dir / "legislation_instructions.parquet"))
        
        return dataset
    
    def _generate_instructions_for_item(self, item: Dict) -> List[Dict]:
        """Generate various instruction-response pairs for a legislation item"""
        instructions = []
        
        text = item['text']
        title = item['title']
        leg_type = item['type']
        year = item['year']
        
        # Question-answering format
        instructions.extend([
            {
                'instruction': f"What is {title}?",
                'input': "",
                'output': f"{title} is a {leg_type} from {year}. Here is its content:\n\n{text}",
                'legislation_id': item['id'],
                'legislation_type': leg_type,
                'legislation_year': year
            },
            {
                'instruction': f"Explain the content of {title}",
                'input': "",
                'output': text,
                'legislation_id': item['id'],
                'legislation_type': leg_type,
                'legislation_year': year
            },
            {
                'instruction': f"What does UK legislation {item['id']} contain?",
                'input': "",
                'output': f"UK legislation {item['id']} titled '{title}' contains:\n\n{text}",
                'legislation_id': item['id'],
                'legislation_type': leg_type,
                'legislation_year': year
            }
        ])
        
        # If text is long, create chunks with context
        if len(text) > self.max_length:
            chunks = self._create_text_chunks(text)
            for i, chunk in enumerate(chunks):
                instructions.append({
                    'instruction': f"Provide section {i+1} of {title}",
                    'input': "",
                    'output': chunk,
                    'legislation_id': item['id'],
                    'legislation_type': leg_type,
                    'legislation_year': year,
                    'section': i+1
                })
        
        # Legal context questions
        instructions.extend([
            {
                'instruction': f"What type of UK legislation is {title}?",
                'input': "",
                'output': f"{title} is a {leg_type} enacted in {year}.",
                'legislation_id': item['id'],
                'legislation_type': leg_type,
                'legislation_year': year
            },
            {
                'instruction': f"When was {title} enacted?",
                'input': "",
                'output': f"{title} was enacted in {year}.",
                'legislation_id': item['id'],
                'legislation_type': leg_type,
                'legislation_year': year
            }
        ])
        
        return instructions
    
    def _create_text_chunks(self, text: str) -> List[str]:
        """Create overlapping chunks of text for training"""
        if self.tokenizer is None:
            # Fallback to character-based chunking
            chunks = []
            chunk_size = self.max_length * 4  # Approximate character count
            overlap = self.chunk_overlap * 4
            
            for i in range(0, len(text), chunk_size - overlap):
                chunk = text[i:i + chunk_size]
                if chunk.strip():
                    chunks.append(chunk.strip())
            
            return chunks
        
        # Tokenizer-based chunking
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        for i in range(0, len(tokens), self.max_length - self.chunk_overlap):
            chunk_tokens = tokens[i:i + self.max_length]
            chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
        
        return chunks
    
    def create_completion_dataset(self, raw_dataset: Dataset) -> Dataset:
        """Create text completion dataset for pre-training style fine-tuning"""
        logger.info("Creating completion dataset...")
        
        completion_data = []
        
        for item in raw_dataset:
            text = item['text']
            
            # Create chunks for completion training
            chunks = self._create_text_chunks(text)
            
            for chunk in chunks:
                completion_data.append({
                    'text': chunk,
                    'legislation_id': item['id'],
                    'legislation_type': item['type'],
                    'legislation_year': item['year'],
                    'source': item['source']
                })
        
        logger.info(f"Created completion dataset with {len(completion_data)} examples")
        
        dataset = Dataset.from_list(completion_data)
        completion_dataset_path = self.processed_datasets_dir / "legislation_completion"
        dataset.save_to_disk(str(completion_dataset_path))
        dataset.to_parquet(str(self.processed_datasets_dir / "legislation_completion.parquet"))
        
        return dataset
    
    def create_qa_dataset(self, raw_dataset: Dataset) -> Dataset:
        """Create question-answer pairs from legislation content"""
        logger.info("Creating Q&A dataset...")
        
        qa_data = []
        
        for item in raw_dataset:
            qa_pairs = self._generate_qa_pairs(item)
            qa_data.extend(qa_pairs)
        
        logger.info(f"Created Q&A dataset with {len(qa_data)} examples")
        
        dataset = Dataset.from_list(qa_data)
        qa_dataset_path = self.processed_datasets_dir / "legislation_qa"
        dataset.save_to_disk(str(qa_dataset_path))
        dataset.to_parquet(str(self.processed_datasets_dir / "legislation_qa.parquet"))
        
        return dataset
    
    def _generate_qa_pairs(self, item: Dict) -> List[Dict]:
        """Generate question-answer pairs from legislation item"""
        qa_pairs = []
        
        text = item['text']
        title = item['title']
        leg_type = item['type']
        year = item['year']
        
        # Extract key information for questions
        sentences = text.split('.')
        paragraphs = text.split('\n\n')
        
        # General questions
        qa_pairs.extend([
            {
                'question': f"What is the title of this {leg_type}?",
                'answer': title,
                'context': text[:500] + "...",
                'legislation_id': item['id'],
                'legislation_type': leg_type,
                'legislation_year': year
            },
            {
                'question': f"What year was this {leg_type} enacted?",
                'answer': str(year),
                'context': text[:500] + "...",
                'legislation_id': item['id'],
                'legislation_type': leg_type,
                'legislation_year': year
            },
            {
                'question': f"What type of UK legislation is this?",
                'answer': leg_type,
                'context': text[:500] + "...",
                'legislation_id': item['id'],
                'legislation_type': leg_type,
                'legislation_year': year
            }
        ])
        
        # Content-based questions from paragraphs
        for i, paragraph in enumerate(paragraphs[:5]):  # Limit to first 5 paragraphs
            if len(paragraph.strip()) > 100:
                qa_pairs.append({
                    'question': f"What does section {i+1} of {title} state?",
                    'answer': paragraph.strip(),
                    'context': text,
                    'legislation_id': item['id'],
                    'legislation_type': leg_type,
                    'legislation_year': year,
                    'section': i+1
                })
        
        return qa_pairs
    
    def create_combined_training_dataset(
        self, 
        instruction_dataset: Dataset,
        completion_dataset: Dataset,
        qa_dataset: Dataset
    ) -> DatasetDict:
        """Combine all datasets into a comprehensive training dataset"""
        logger.info("Creating combined training dataset...")
        
        # Create different splits for various training scenarios
        datasets = {
            'instruction_following': instruction_dataset,
            'text_completion': completion_dataset,
            'question_answering': qa_dataset
        }
        
        # Create train/validation splits for each dataset type
        split_datasets = {}
        for name, dataset in datasets.items():
            # Split 90% train, 10% validation
            split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
            split_datasets[f"{name}_train"] = split_dataset['train']
            split_datasets[f"{name}_validation"] = split_dataset['test']
        
        # Create comprehensive training dataset
        combined_dataset = DatasetDict(split_datasets)
        
        # Save combined dataset
        combined_path = self.final_datasets_dir / "uk_legislation_complete"
        combined_dataset.save_to_disk(str(combined_path))
        
        # Save individual components as parquet
        for name, dataset in combined_dataset.items():
            dataset.to_parquet(str(self.final_datasets_dir / f"{name}.parquet"))
        
        return combined_dataset
    
    def validate_datasets(self, dataset_dict: DatasetDict) -> Dict[str, Dict]:
        """Validate created datasets and provide statistics"""
        logger.info("Validating datasets...")
        
        validation_stats = {}
        
        for name, dataset in dataset_dict.items():
            stats = {
                'total_examples': len(dataset),
                'columns': list(dataset.column_names),
                'avg_text_length': 0,
                'min_text_length': float('inf'),
                'max_text_length': 0
            }
            
            # Calculate text statistics
            text_lengths = []
            text_column = None
            
            # Find text column
            for col in ['text', 'output', 'answer']:
                if col in dataset.column_names:
                    text_column = col
                    break
            
            if text_column:
                for item in dataset:
                    length = len(item[text_column])
                    text_lengths.append(length)
                
                if text_lengths:
                    stats['avg_text_length'] = sum(text_lengths) / len(text_lengths)
                    stats['min_text_length'] = min(text_lengths)
                    stats['max_text_length'] = max(text_lengths)
            
            validation_stats[name] = stats
            logger.info(f"Dataset {name}: {stats['total_examples']} examples")
        
        # Save validation report
        with open(self.output_dir / "validation_report.json", 'w') as f:
            json.dump(validation_stats, f, indent=2)
        
        return validation_stats
    
    def create_all_datasets(self) -> DatasetDict:
        """Create all dataset types from downloaded legislation"""
        logger.info("Starting complete dataset creation process...")
        
        # Step 1: Create raw dataset from downloaded files
        raw_dataset = self.create_raw_text_dataset()
        
        # Step 2: Create specialised datasets
        instruction_dataset = self.create_instruction_dataset(raw_dataset)
        completion_dataset = self.create_completion_dataset(raw_dataset)
        qa_dataset = self.create_qa_dataset(raw_dataset)
        
        # Step 3: Combine into comprehensive training dataset
        combined_dataset = self.create_combined_training_dataset(
            instruction_dataset, completion_dataset, qa_dataset
        )
        
        # Step 4: Validate datasets
        validation_stats = self.validate_datasets(combined_dataset)
        
        logger.info("Dataset creation completed successfully!")
        logger.info(f"Created datasets with following statistics:")
        for name, stats in validation_stats.items():
            logger.info(f"  {name}: {stats['total_examples']} examples")
        
        return combined_dataset

def main():
    """Main function to create UK legislation datasets"""
    # Setup PyTorch before creating datasets
    if not setup_pytorch():
        logger.error("Failed to setup PyTorch. Exiting.")
        sys.exit(1)
    
    creator = UKLegislationDatasetCreator()
    
    try:
        # Create all datasets
        datasets = creator.create_all_datasets()
        
        print(f"\n=== DATASET CREATION SUMMARY ===")
        for name, dataset in datasets.items():
            print(f"{name}: {len(dataset)} examples")
        
        print(f"\nDatasets saved to: {creator.output_dir}")
        print("Available formats: HuggingFace Dataset, Parquet, JSON")
        
    except Exception as e:
        logger.error(f"Dataset creation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
ParaLlama Training Dataset Optimizer

Optimizes all collected legal and tax data for training Legal Llama 3.1 70B Instruct
using HuggingFace AutoTrain Advanced. Creates domain-specific datasets with
proper formatting, prompt engineering, and multi-round training support.
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random
from datetime import datetime

try:
    from datasets import Dataset, DatasetDict, load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    # Create stub classes for testing
    class Dataset:
        pass
    class DatasetDict:
        pass
    def load_dataset(*args, **kwargs):
        return None
    DATASETS_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ParaLlamaTrainingOptimizer:
    def __init__(self, input_dir: str, output_dir: str = "generated/llama_training"):
        if not DATASETS_AVAILABLE:
            logger.warning("datasets library not available - some functionality will be limited")
        if not PANDAS_AVAILABLE:
            logger.warning("pandas library not available - some functionality will be limited")
            
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Llama 3.1 specific prompt templates
        self.llama_templates = {
            "legal_specialist": {
                "system": "You are a highly experienced UK legal specialist with comprehensive knowledge of all UK legislation, case law, and legal procedures. You provide accurate, detailed legal analysis and can counter arguments effectively while maintaining professional standards.",
                "instruction_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>",
                "domains": ["legal", "case_law", "legislation", "procedures"]
            },
            "tax_specialist": {
                "system": "You are an expert UK tax advisor with complete knowledge of all HMRC rules, regulations, and tax optimization strategies. You ensure full tax compliance while maximizing legitimate tax savings and minimizing tax liabilities.",
                "instruction_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>",
                "domains": ["tax", "hmrc", "compliance", "optimization"]
            }
        }
        
        # Training phases for progressive complexity
        self.training_phases = {
            "phase_1_foundation": {
                "description": "Basic legal and tax knowledge",
                "max_length": 1024,
                "complexity": "basic",
                "focus": "foundational_knowledge"
            },
            "phase_2_reasoning": {
                "description": "Multi-step reasoning and analysis",
                "max_length": 2048,
                "complexity": "intermediate",
                "focus": "reasoning_chains"
            },
            "phase_3_expertise": {
                "description": "Expert-level problem solving",
                "max_length": 4096,
                "complexity": "advanced",
                "focus": "expert_application"
            },
            "phase_4_adversarial": {
                "description": "Adversarial and challenge scenarios",
                "max_length": 4096,
                "complexity": "adversarial",
                "focus": "challenge_response"
            }
        }
    
    def load_all_datasets(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load all available datasets from the enhanced pipelines"""
        datasets = {
            "base_legal": [],
            "base_tax": [],
            "enhanced_legal": [],
            "enhanced_tax": [],
            "advanced_qa": [],
            "reasoning_enhanced": [],
            "tax_scenarios": []
        }
        
        # Load base datasets
        base_dirs = [
            ("housing_legislation", "base_legal"),
            ("housing_case_law", "base_legal"),
            ("bailii_cases", "base_legal"),
            ("hmrc_documentation", "base_tax"),
            ("test_hmrc", "base_tax")
        ]
        
        for dir_name, dataset_key in base_dirs:
            dataset_path = self.input_dir / dir_name
            if dataset_path.exists():
                datasets[dataset_key].extend(self._load_directory_data(dataset_path))
        
        # Load enhanced datasets
        enhanced_files = [
            ("enhanced_legal/enhanced_legal_reasoning.json", "enhanced_legal"),
            ("enhanced_tax/enhanced_tax_scenarios.json", "enhanced_tax"),
            ("advanced_qa/advanced_qa_dataset.json", "advanced_qa"),
            ("enhanced_legal/enhanced_legal_reasoning.json", "reasoning_enhanced"),
            ("enhanced_tax/enhanced_tax_scenarios.json", "tax_scenarios")
        ]
        
        for file_path, dataset_key in enhanced_files:
            full_path = self.input_dir / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            datasets[dataset_key].extend(data)
                except Exception as e:
                    logger.warning(f"Could not load {full_path}: {e}")
        
        return datasets
    
    def _load_directory_data(self, directory: Path) -> List[Dict[str, Any]]:
        """Load data from a directory structure"""
        data = []
        
        metadata_dir = directory / "metadata"
        text_dir = directory / "text"
        
        if not metadata_dir.exists() or not text_dir.exists():
            return data
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                
                text_file = text_dir / f"{metadata_file.stem}.txt"
                if text_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    data.append({
                        "content": content,
                        "metadata": metadata,
                        "title": metadata.get("title", "Unknown"),
                        "source": str(directory.name)
                    })
                
            except Exception as e:
                logger.warning(f"Error loading {metadata_file}: {e}")
                continue
        
        return data
    
    def format_for_llama_training(self, data: List[Dict[str, Any]], specialist_type: str, phase: str) -> List[Dict[str, Any]]:
        """Format data specifically for Llama 3.1 training"""
        formatted_data = []
        
        template_config = self.llama_templates[specialist_type]
        phase_config = self.training_phases[phase]
        
        system_message = template_config["system"]
        
        for item in data:
            try:
                # Determine if this is Q&A format or content format
                if "question" in item and "answer" in item:
                    instruction = item["question"]
                    response = item["answer"]
                elif "instruction" in item and "output" in item:
                    instruction = item["instruction"]
                    response = item["output"]
                    if "input" in item and item["input"]:
                        instruction = f"{instruction}\n\nContext: {item['input']}"
                else:
                    # Convert raw content to instruction format
                    content = item.get("content", "")
                    title = item.get("title", "Document")
                    
                    if len(content) < 100:  # Skip very short content
                        continue
                    
                    # Create instruction-response pairs from content
                    instruction, response = self._create_instruction_from_content(content, title, specialist_type)
                
                # Apply length constraints based on phase
                if len(response) > phase_config["max_length"]:
                    response = response[:phase_config["max_length"]] + "..."
                
                # Filter by complexity if specified
                complexity = item.get("complexity", "basic")
                if phase_config["complexity"] != "basic" and complexity != phase_config["complexity"]:
                    continue
                
                # Format with Llama template
                formatted_text = template_config["instruction_template"].format(
                    system=system_message,
                    instruction=instruction,
                    response=response
                )
                
                formatted_item = {
                    "text": formatted_text,
                    "instruction": instruction,
                    "response": response,
                    "specialist_type": specialist_type,
                    "training_phase": phase,
                    "complexity": complexity,
                    "length": len(formatted_text),
                    "metadata": item.get("metadata", {})
                }
                
                formatted_data.append(formatted_item)
                
            except Exception as e:
                logger.warning(f"Error formatting item: {e}")
                continue
        
        return formatted_data
    
    def _create_instruction_from_content(self, content: str, title: str, specialist_type: str) -> Tuple[str, str]:
        """Create instruction-response pairs from raw content"""
        
        # Extract first paragraph or section as context
        content_parts = content.split('\n\n')
        if len(content_parts) > 1:
            context = content_parts[0]
            details = '\n\n'.join(content_parts[1:3])  # Next 2 paragraphs
        else:
            context = content[:500]
            details = content[500:1500]
        
        if specialist_type == "legal_specialist":
            instructions = [
                f"Explain the legal principles in '{title}' and their practical application.",
                f"Analyze the key legal requirements outlined in '{title}'.",
                f"What are the main legal implications of '{title}' for practitioners?",
                f"Summarize the essential legal points from '{title}' that legal professionals should know."
            ]
        else:  # tax_specialist
            instructions = [
                f"Explain the tax implications and requirements outlined in '{title}'.",
                f"What are the key compliance obligations detailed in '{title}'?",
                f"Analyze the tax planning opportunities discussed in '{title}'.",
                f"Summarize the essential tax guidance from '{title}' for practitioners."
            ]
        
        instruction = random.choice(instructions)
        
        # Create comprehensive response
        response = f"""Based on '{title}':

**Key Points:**
{context}

**Detailed Analysis:**
{details}

**Practical Application:**
[This would be supplemented with specific guidance based on the full document content]

**References:**
- {title}
- Relevant supporting authorities would be cited here
"""
        
        return instruction, response.strip()
    
    def create_multi_round_training_data(self, datasets: Dict[str, List[Dict[str, Any]]]) -> Dict[str, DatasetDict]:
        """Create multi-round training datasets for progressive learning"""
        
        training_datasets = {}
        
        for specialist_type in ["legal_specialist", "tax_specialist"]:
            logger.info(f"Creating training data for {specialist_type}")
            
            # Determine relevant datasets for this specialist
            if specialist_type == "legal_specialist":
                relevant_data = (
                    datasets["base_legal"] + 
                    datasets["enhanced_legal"] + 
                    [item for item in datasets["advanced_qa"] if item.get("domain") == "legal"]
                )
            else:
                relevant_data = (
                    datasets["base_tax"] + 
                    datasets["enhanced_tax"] + 
                    [item for item in datasets["advanced_qa"] if item.get("domain") == "tax"] +
                    datasets["tax_scenarios"]
                )
            
            # Create phase-specific datasets
            phase_datasets = {}
            
            for phase_name, phase_config in self.training_phases.items():
                logger.info(f"Creating {phase_name} dataset for {specialist_type}")
                
                formatted_data = self.format_for_llama_training(relevant_data, specialist_type, phase_name)
                
                if not formatted_data:
                    logger.warning(f"No data for {specialist_type} {phase_name}")
                    continue
                
                # Split into train/validation
                random.shuffle(formatted_data)
                split_idx = int(len(formatted_data) * 0.8)
                
                train_data = formatted_data[:split_idx]
                val_data = formatted_data[split_idx:]
                
                # Create HuggingFace datasets
                train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
                val_dataset = Dataset.from_pandas(pd.DataFrame(val_data))
                
                phase_datasets[phase_name] = DatasetDict({
                    "train": train_dataset,
                    "validation": val_dataset
                })
                
                # Save phase dataset
                phase_output_dir = self.output_dir / specialist_type / phase_name
                phase_output_dir.mkdir(parents=True, exist_ok=True)
                
                phase_datasets[phase_name].save_to_disk(str(phase_output_dir))
                
                # Also save as parquet for AutoTrain
                train_dataset.to_parquet(str(phase_output_dir / "train.parquet"))
                val_dataset.to_parquet(str(phase_output_dir / "validation.parquet"))
                
                logger.info(f"Phase {phase_name}: {len(train_data)} train, {len(val_data)} validation examples")
            
            training_datasets[specialist_type] = phase_datasets
        
        return training_datasets
    
    def create_autotrain_config(self, datasets: Dict[str, Dict[str, DatasetDict]]) -> Dict[str, Any]:
        """Create AutoTrain configuration for Llama 3.1 70B"""
        
        autotrain_configs = {}
        
        for specialist_type, phase_datasets in datasets.items():
            for phase_name, dataset_dict in phase_datasets.items():
                
                train_size = len(dataset_dict["train"])
                val_size = len(dataset_dict["validation"])
                
                # Calculate training parameters based on dataset size and phase
                if train_size < 100:
                    epochs = 5
                    batch_size = 2
                elif train_size < 1000:
                    epochs = 3
                    batch_size = 4
                else:
                    epochs = 2
                    batch_size = 8
                
                config = {
                    "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                    "task": "text-generation",
                    "backend": "autotrain",
                    "data": {
                        "train_data": f"{specialist_type}/{phase_name}/train.parquet",
                        "validation_data": f"{specialist_type}/{phase_name}/validation.parquet",
                        "text_column": "text"
                    },
                    "training": {
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "learning_rate": 2e-5,
                        "warmup_steps": 100,
                        "logging_steps": 10,
                        "save_steps": 500,
                        "eval_steps": 500,
                        "max_length": self.training_phases[phase_name]["max_length"],
                        "gradient_accumulation_steps": 4,
                        "fp16": True,
                        "dataloader_num_workers": 4
                    },
                    "lora": {
                        "r": 16,
                        "alpha": 32,
                        "dropout": 0.1,
                        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
                    },
                    "output": {
                        "model_name": f"llama-3.1-70b-{specialist_type}-{phase_name}",
                        "push_to_hub": False
                    },
                    "metadata": {
                        "specialist_type": specialist_type,
                        "training_phase": phase_name,
                        "train_examples": train_size,
                        "validation_examples": val_size,
                        "description": self.training_phases[phase_name]["description"]
                    }
                }
                
                autotrain_configs[f"{specialist_type}_{phase_name}"] = config
        
        return autotrain_configs
    
    def generate_training_readme(self, datasets: Dict[str, Dict[str, DatasetDict]], configs: Dict[str, Any]) -> str:
        """Generate comprehensive training documentation"""
        
        readme_content = f"""# Llama 3.1 70B Legal & Tax Specialist Training

This directory contains optimized datasets and configurations for training domain-specific LLMs using HuggingFace AutoTrain Advanced.

## Training Strategy

### Multi-Phase Progressive Training
The training follows a 4-phase approach for building expertise:

1. **Phase 1 - Foundation**: Basic legal and tax knowledge
2. **Phase 2 - Reasoning**: Multi-step analysis and reasoning
3. **Phase 3 - Expertise**: Expert-level problem solving
4. **Phase 4 - Adversarial**: Challenge scenarios and counter-arguments

### Specialist Types

#### Legal Specialist
- **Purpose**: Counter legal arguments for defendants, provide comprehensive legal analysis
- **Knowledge Base**: All UK legislation, case law, procedures
- **Capabilities**: Legal reasoning, precedent analysis, defense strategies

#### Tax Specialist  
- **Purpose**: Ensure tax compliance while maximizing legitimate savings
- **Knowledge Base**: Complete HMRC guidance, tax calculations, optimization strategies
- **Capabilities**: Tax planning, compliance checking, optimization advice

## Dataset Statistics

"""
        
        for specialist_type, phase_datasets in datasets.items():
            readme_content += f"\n### {specialist_type.replace('_', ' ').title()}\n"
            
            total_train = 0
            total_val = 0
            
            for phase_name, dataset_dict in phase_datasets.items():
                train_size = len(dataset_dict["train"])
                val_size = len(dataset_dict["validation"])
                total_train += train_size
                total_val += val_size
                
                readme_content += f"- **{phase_name}**: {train_size:,} train, {val_size:,} validation examples\n"
            
            readme_content += f"- **Total**: {total_train:,} train, {total_val:,} validation examples\n"
        
        readme_content += f"""

## Training with AutoTrain Advanced

### Prerequisites
```bash
pip install autotrain-advanced
autotrain setup
```

### Training Commands

"""
        
        for config_name, config in configs.items():
            readme_content += f"""
#### {config_name.replace('_', ' ').title()}
```bash
autotrain llm \\
    --train {config['data']['train_data']} \\
    --model {config['model']} \\
    --project-name {config['output']['model_name']} \\
    --epochs {config['training']['epochs']} \\
    --batch-size {config['training']['batch_size']} \\
    --lr {config['training']['learning_rate']} \\
    --block-size {config['training']['max_length']} \\
    --use-peft \\
    --lora-r {config['lora']['r']} \\
    --lora-alpha {config['lora']['alpha']} \\
    --lora-dropout {config['lora']['dropout']}
```
"""
        
        readme_content += f"""

## Recommended Training Order

1. **Start with Phase 1** for foundational knowledge
2. **Continue with Phase 2** for reasoning capabilities  
3. **Advance to Phase 3** for expert-level skills
4. **Finish with Phase 4** for adversarial robustness

Each phase builds on the previous, creating increasingly sophisticated domain specialists.

## Model Evaluation

Test the trained models with:
- Legal precedent analysis tasks
- Tax calculation and optimization scenarios
- Adversarial legal arguments
- Complex multi-step reasoning problems

## Expected Outcomes

### Legal Specialist Model
- Comprehensive UK legal knowledge
- Ability to counter prosecution arguments
- Strategic defense planning
- Precedent analysis and application

### Tax Specialist Model
- Complete HMRC compliance knowledge
- Tax optimization strategies
- Calculation accuracy
- Risk assessment and mitigation

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return readme_content
    
    def optimize_for_llama_training(self) -> Dict[str, Any]:
        """Complete optimization process for Llama 3.1 training"""
        
        logger.info("Starting Llama 3.1 training optimization...")
        
        # Load all datasets
        datasets = self.load_all_datasets()
        
        # Create multi-round training data
        training_datasets = self.create_multi_round_training_data(datasets)
        
        # Create AutoTrain configurations
        autotrain_configs = self.create_autotrain_config(training_datasets)
        
        # Save configurations
        config_file = self.output_dir / "autotrain_configs.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(autotrain_configs, f, indent=2)
        
        # Generate training documentation
        readme_content = self.generate_training_readme(training_datasets, autotrain_configs)
        readme_file = self.output_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(readme_content)
        
        # Generate summary statistics
        stats = {
            "specialists": list(training_datasets.keys()),
            "phases": list(self.training_phases.keys()),
            "total_configs": len(autotrain_configs),
            "output_directory": str(self.output_dir)
        }
        
        for specialist_type, phase_datasets in training_datasets.items():
            specialist_stats = {}
            for phase_name, dataset_dict in phase_datasets.items():
                specialist_stats[phase_name] = {
                    "train_examples": len(dataset_dict["train"]),
                    "validation_examples": len(dataset_dict["validation"])
                }
            stats[specialist_type] = specialist_stats
        
        stats_file = self.output_dir / "optimization_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        
        logger.info("Llama 3.1 training optimization complete!")
        return stats

def main():
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="ParaLlama: Optimize datasets for Legal Llama 3.1 70B training")
    parser.add_argument('--input-dir', default='generated',
                       help='Input directory containing all collected data')
    parser.add_argument('--output-dir', default='generated/llama_training',
                       help='Output directory for optimized training data')
    
    args = parser.parse_args()
    
    optimizer = ParaLlamaTrainingOptimizer(args.input_dir, args.output_dir)
    stats = optimizer.optimize_for_llama_training()
    
    print(f"\n=== ParaLlama Legal Llama 3.1 Training Optimization Complete ===")
    print(f"Specialist types: {len(stats['specialists'])}")
    print(f"Training phases: {len(stats['phases'])}")
    print(f"AutoTrain configurations: {stats['total_configs']}")
    
    for specialist in stats['specialists']:
        if specialist in stats:
            total_train = sum(phase['train_examples'] for phase in stats[specialist].values())
            total_val = sum(phase['validation_examples'] for phase in stats[specialist].values())
            print(f"{specialist}: {total_train:,} train, {total_val:,} validation examples")
    
    print(f"Output directory: {stats['output_directory']}")
    print(f"\nReady for HuggingFace AutoTrain Advanced!")

if __name__ == "__main__":
    main()
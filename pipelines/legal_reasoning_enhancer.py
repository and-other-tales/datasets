#!/usr/bin/env python3
"""
Legal Reasoning Dataset Enhancer

Enhances legal datasets with advanced reasoning patterns, procedural knowledge,
and adversarial examples for training sophisticated legal LLMs.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
import random

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.pipeline_controller import PipelineController, create_database_update_callback, create_dataset_creation_callback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalReasoningEnhancer:
    def __init__(self, input_dir: str, output_dir: str = "generated/enhanced_legal"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline controller for pause functionality
        self.controller = PipelineController()
        
        # Register pause functionality callbacks
        self.controller.register_callback('database_update', create_database_update_callback(self))
        self.controller.register_callback('dataset_creation', create_dataset_creation_callback(self))
        
        logger.info("🔶 Pipeline Control: Press P to pause/resume, A to update databases (when paused), D to create dataset (when paused), Q to quit")
        
        # Legal reasoning templates for enhanced training
        self.reasoning_templates = {
            "precedent_analysis": [
                "Analyze the precedent set by {case_name} and how it applies to {scenario}",
                "Compare the facts in {case_name} with {scenario} and determine if the precedent applies",
                "Explain why {case_name} is distinguishable from {scenario}",
            ],
            "statutory_interpretation": [
                "Interpret section {section} of {act} in the context of {scenario}",
                "Apply the golden rule of statutory interpretation to {provision} regarding {scenario}",
                "Using purposive interpretation, explain how {provision} should apply to {scenario}",
            ],
            "defense_strategies": [
                "Construct a defense argument for {scenario} using {legal_principle}",
                "Identify potential defenses available in {scenario} under {applicable_law}",
                "Counter the prosecution's argument in {scenario} by challenging {element}",
            ],
            "procedural_knowledge": [
                "Explain the procedural steps required for {legal_action} under {court_rules}",
                "Identify time limits and procedural requirements for {legal_process}",
                "Outline the evidence disclosure requirements in {case_type} proceedings",
            ],
            "burden_of_proof": [
                "Explain who bears the burden of proof in {scenario} and to what standard",
                "Analyze how the burden of proof shifts in {legal_context}",
                "Determine what evidence is required to meet the burden in {case_type}",
            ]
        }
        
        # Adversarial challenge templates
        self.adversarial_templates = [
            "Challenge this legal argument: {argument}",
            "Find weaknesses in this interpretation: {interpretation}",
            "Counter this defense strategy: {defense}",
            "Rebut this precedent application: {precedent_claim}",
        ]
    
    def generate_reasoning_examples(self, case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate enhanced reasoning examples from case data"""
        examples = []
        
        case_name = case_data.get('title', 'Unknown Case')
        case_facts = case_data.get('content', '')[:500]  # First 500 chars as summary
        
        # Generate different types of reasoning examples
        for reasoning_type, templates in self.reasoning_templates.items():
            for template in templates[:2]:  # Use first 2 templates per type
                try:
                    if reasoning_type == "precedent_analysis":
                        prompt = template.format(
                            case_name=case_name,
                            scenario="a similar factual situation"
                        )
                    elif reasoning_type == "statutory_interpretation":
                        prompt = template.format(
                            section="relevant section",
                            act="applicable statute",
                            provision="the statutory provision",
                            scenario="the current case"
                        )
                    else:
                        prompt = template.format(
                            scenario="the case facts",
                            legal_principle="established legal principles",
                            applicable_law="relevant legislation",
                            element="key elements of the case",
                            legal_action="the required legal action",
                            court_rules="applicable court rules",
                            legal_process="the legal process",
                            case_type="this type of case",
                            legal_context="this legal context"
                        )
                    
                    example = {
                        "instruction": prompt,
                        "input": f"Case: {case_name}\nFacts: {case_facts}",
                        "output": f"[Legal reasoning response would be generated here based on {case_name}]",
                        "reasoning_type": reasoning_type,
                        "source_case": case_name,
                        "metadata": case_data.get('metadata', {})
                    }
                    examples.append(example)
                    
                except KeyError as e:
                    logger.warning(f"Template formatting error: {e}")
                    continue
        
        return examples
    
    def generate_adversarial_examples(self, case_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate adversarial training examples"""
        examples = []
        
        case_name = case_data.get('title', 'Unknown Case')
        
        # Create adversarial challenges
        arguments = [
            f"The precedent in {case_name} clearly applies to this situation",
            f"The defendant in {case_name} had no valid defense",
            f"The court's reasoning in {case_name} was legally sound",
            f"The statutory interpretation in {case_name} should be followed"
        ]
        
        for i, template in enumerate(self.adversarial_templates):
            if i < len(arguments):
                challenge = template.format(
                    argument=arguments[i],
                    interpretation=arguments[i],
                    defense=arguments[i],
                    precedent_claim=arguments[i]
                )
                
                example = {
                    "instruction": challenge,
                    "input": f"Reference case: {case_name}",
                    "output": f"[Adversarial response challenging the given position]",
                    "example_type": "adversarial",
                    "source_case": case_name,
                    "metadata": case_data.get('metadata', {})
                }
                examples.append(example)
        
        return examples
    
    def enhance_dataset(self) -> Dict[str, int]:
        """Enhance the legal dataset with reasoning and adversarial examples"""
        logger.info("=== STARTING LEGAL REASONING ENHANCEMENT ===")
        
        stats = {"reasoning_examples": 0, "adversarial_examples": 0, "total_enhanced": 0}
        enhanced_examples = []
        
        try:
            # Phase 1: Process case law files
            logger.info("Phase 1: Processing case law files...")
            self.controller.set_current_phase('case_processing', {'step': 'reasoning_analysis'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            case_dirs = [
                self.input_dir / "housing_case_law",
                self.input_dir / "case_law",
                self.input_dir / "bailii_cases"
            ]
            
            for case_dir in case_dirs:
                if not case_dir.exists():
                    continue
                    
                metadata_dir = case_dir / "metadata"
                if not metadata_dir.exists():
                    continue
                
                logger.info(f"Processing {case_dir.name}...")
                
                for metadata_file in metadata_dir.glob("*.json"):
                    try:
                        # Check for pause commands periodically
                        self.controller.check_for_commands()
                        self.controller.wait_while_paused()
                        
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            case_data = json.load(f)
                        
                        # Read corresponding text file
                        text_file = case_dir / "text" / f"{metadata_file.stem}.txt"
                        if text_file.exists():
                            with open(text_file, 'r', encoding='utf-8') as f:
                                case_data['content'] = f.read()
                        
                        # Generate enhanced examples
                        reasoning_examples = self.generate_reasoning_examples(case_data)
                        adversarial_examples = self.generate_adversarial_examples(case_data)
                        
                        enhanced_examples.extend(reasoning_examples)
                        enhanced_examples.extend(adversarial_examples)
                        
                        stats["reasoning_examples"] += len(reasoning_examples)
                        stats["adversarial_examples"] += len(adversarial_examples)
                        
                    except Exception as e:
                        logger.error(f"Error processing {metadata_file}: {e}")
                        continue
            
            # Phase 2: Save enhanced dataset
            logger.info("Phase 2: Saving enhanced datasets...")
            self.controller.set_current_phase('dataset_saving', {'step': 'output_generation'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            # Save enhanced dataset
            output_file = self.output_dir / "enhanced_legal_reasoning.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_examples, f, indent=2, ensure_ascii=False)
            
            stats["total_enhanced"] = len(enhanced_examples)
            
            # Save statistics
            stats_file = self.output_dir / "enhancement_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Enhanced dataset created with {stats['total_enhanced']} examples")
            return stats
            
        except Exception as e:
            logger.error(f"Legal reasoning enhancement failed: {e}")
            raise
        finally:
            # Cleanup controller
            self.controller.cleanup()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhance legal datasets for LLM training")
    parser.add_argument('--input-dir', default='generated',
                       help='Input directory containing legal data')
    parser.add_argument('--output-dir', default='generated/enhanced_legal',
                       help='Output directory for enhanced dataset')
    
    args = parser.parse_args()
    
    enhancer = LegalReasoningEnhancer(args.input_dir, args.output_dir)
    stats = enhancer.enhance_dataset()
    
    print(f"\n=== Legal Reasoning Enhancement Complete ===")
    print(f"Reasoning examples generated: {stats['reasoning_examples']}")
    print(f"Adversarial examples generated: {stats['adversarial_examples']}")
    print(f"Total enhanced examples: {stats['total_enhanced']}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
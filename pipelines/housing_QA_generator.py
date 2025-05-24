#!/usr/bin/env python3
"""
Housing-Specific QA Generator
Generates housing, tenancy, and landlord-tenant focused Q&A pairs
"""

import os
import sys
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional
import anthropic

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.pipeline_controller import PipelineController, create_database_update_callback, create_dataset_creation_callback
from utils.QA_pairs import UKLegislationQAGenerator

logger = logging.getLogger(__name__)

class HousingQAGenerator(UKLegislationQAGenerator):
    def __init__(self, anthropic_api_key: Optional[str] = None):
        super().__init__(anthropic_api_key)
        
        # Initialize pipeline controller for pause functionality
        self.controller = PipelineController()
        
        # Register pause functionality callbacks
        self.controller.register_callback('database_update', create_database_update_callback(self))
        self.controller.register_callback('dataset_creation', create_dataset_creation_callback(self))
        
        
        # Housing-specific prompts and categories
        self.housing_categories = {
            'tenancy_types': [
                'assured shorthold tenancy', 'assured tenancy', 'secure tenancy',
                'protected tenancy', 'statutory tenancy', 'periodic tenancy'
            ],
            'eviction_possession': [
                'section 8', 'section 21', 'possession', 'eviction',
                'notice to quit', 'mandatory ground', 'discretionary ground'
            ],
            'tenant_rights': [
                'quiet enjoyment', 'repair obligations', 'deposit protection',
                'harassment', 'illegal eviction', 'retaliatory eviction'
            ],
            'landlord_obligations': [
                'gas safety', 'electrical safety', 'fire safety',
                'energy efficiency', 'licensing', 'deposit protection'
            ],
            'leasehold': [
                'service charge', 'ground rent', 'major works',
                'lease extension', 'enfranchisement', 'right to manage'
            ]
        }
    
    def identify_housing_category(self, text: str) -> str:
        """Identify the main housing category for the text"""
        text_lower = text.lower()
        
        category_scores = {}
        for category, terms in self.housing_categories.items():
            score = sum(1 for term in terms if term in text_lower)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        
        # Default categorization
        if any(term in text_lower for term in ['eviction', 'possession', 'section 8', 'section 21']):
            return 'eviction_possession'
        elif any(term in text_lower for term in ['tenancy', 'tenant', 'assured', 'shorthold']):
            return 'tenancy_types'
        elif any(term in text_lower for term in ['leasehold', 'service charge', 'ground rent']):
            return 'leasehold'
        elif any(term in text_lower for term in ['landlord', 'repair', 'safety']):
            return 'landlord_obligations'
        else:
            return 'tenant_rights'
    
    def generate_housing_qa_pairs(self, section_text: str, act_name: str, section_number: str,
                                model: str = "claude-3-5-sonnet-20241022") -> List[Dict[str, str]]:
        """Generate housing-specific Q&A pairs"""
        
        housing_category = self.identify_housing_category(section_text)
        
        # Category-specific prompts
        category_prompts = {
            'tenancy_types': """Focus on questions about:
- Types of tenancies and their characteristics
- Rights and obligations under different tenancy types
- How tenancies are created, transferred, and terminated
- Differences between assured, secure, and protected tenancies""",
            
            'eviction_possession': """Focus on questions about:
- Grounds for possession and eviction procedures
- Notice requirements (Section 8, Section 21)
- Mandatory vs discretionary grounds
- Court procedures and tenant defenses
- Unlawful eviction and remedies""",
            
            'tenant_rights': """Focus on questions about:
- Tenant rights to quiet enjoyment and peaceful occupation
- Protection from harassment and illegal eviction
- Rights regarding repairs and housing conditions
- Deposit protection and return procedures""",
            
            'landlord_obligations': """Focus on questions about:
- Landlord duties regarding property maintenance
- Safety obligations (gas, electrical, fire)
- Licensing requirements for rental properties
- Consequences of failing to meet obligations""",
            
            'leasehold': """Focus on questions about:
- Service charges and their reasonableness
- Ground rent and its calculation
- Major works consultation procedures
- Lease extension and enfranchisement rights
- Right to manage and its exercise"""
        }
        
        category_prompt = category_prompts.get(housing_category, "")
        
        prompt = f"""You are a housing law expert specializing in UK residential tenancy law. For the following section of the {act_name}, generate a set of question-answer pairs that cover housing-specific legal provisions.

{category_prompt}

Generate 4-6 question-answer pairs in this exact JSON format, ensuring questions are practical and relevant to tenants, landlords, or housing practitioners:
[
    {{"question": "What rights does this section give to tenants?", "answer": "This section gives tenants the right to..."}},
    {{"question": "What are a landlord's obligations under this provision?", "answer": "Under this provision, landlords must..."}}
]

Focus on:
- Practical implications for tenants and landlords
- Procedural requirements and time limits
- Remedies available for breaches
- Exceptions and special circumstances
- Real-world application of the law

Section {section_number} of {act_name}:
{section_text}

Return only the JSON array, no additional text."""

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=2500,
                temperature=0.2,
                system="You are a housing law expert. Generate precise, practical question-answer pairs about UK housing and tenancy law in JSON format.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.content[0].text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                try:
                    qa_pairs = json.loads(json_match.group())
                    
                    # Add housing category to each Q&A pair
                    for qa in qa_pairs:
                        qa['housing_category'] = housing_category
                    
                    return qa_pairs
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error for section {section_number}: {e}")
                    return []
            else:
                logger.warning(f"No JSON found in response for section {section_number}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating housing QA pairs for section {section_number}: {e}")
            return []
    
    def generate_case_law_qa_pairs(self, case_title: str, case_summary: str, case_content: str,
                                 model: str = "claude-3-5-sonnet-20241022") -> List[Dict[str, str]]:
        """Generate Q&A pairs from housing case law"""
        
        housing_category = self.identify_housing_category(case_title + " " + case_summary)
        
        prompt = f"""You are a housing law expert. For the following housing case, generate question-answer pairs that explain the legal principles, outcomes, and practical implications.

Generate 3-5 question-answer pairs in this exact JSON format:
[
    {{"question": "What was the main legal issue in this case?", "answer": "The main legal issue was..."}},
    {{"question": "How did the court decide this case?", "answer": "The court decided that..."}}
]

Focus on:
- Key legal principles established or applied
- Practical implications for landlords and tenants
- How this case affects housing law
- Precedent value and future applications

Case: {case_title}
Summary: {case_summary}
Content: {case_content[:2000]}...

Return only the JSON array, no additional text."""

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.2,
                system="You are a housing law expert. Generate precise question-answer pairs about housing case law in JSON format.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = response.content[0].text.strip()
            
            # Extract JSON from response
            json_match = re.search(r'\[.*?\]', response_text, re.DOTALL)
            if json_match:
                try:
                    qa_pairs = json.loads(json_match.group())
                    
                    # Add metadata
                    for qa in qa_pairs:
                        qa['housing_category'] = housing_category
                        qa['source_type'] = 'case_law'
                        qa['case_title'] = case_title
                    
                    return qa_pairs
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error for case {case_title}: {e}")
                    return []
            else:
                logger.warning(f"No JSON found in response for case {case_title}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating case law QA pairs for {case_title}: {e}")
            return []
    
    def process_housing_legislation_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Process a housing legislation file with housing-specific Q&A generation"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            act_name = self.extract_act_name(text)
            sections = self.parse_sections(text)
            
            all_qa_pairs = []
            
            logger.info(f"Processing housing legislation: {act_name} with {len(sections)} sections")
            
            for section in sections:
                qa_pairs = self.generate_housing_qa_pairs(
                    section['content'], 
                    act_name, 
                    section['section_number']
                )
                
                for qa in qa_pairs:
                    all_qa_pairs.append({
                        'act_name': act_name,
                        'section_number': section['section_number'],
                        'question': qa['question'],
                        'answer': qa['answer'],
                        'housing_category': qa.get('housing_category', ''),
                        'source_type': 'legislation',
                        'source_file': str(file_path.name)
                    })
            
            return all_qa_pairs
            
        except Exception as e:
            logger.error(f"Error processing housing legislation file {file_path}: {e}")
            return []
    
    def process_housing_case_law(self, case_law_file: str) -> List[Dict[str, str]]:
        """Process housing case law and generate Q&A pairs"""
        try:
            with open(case_law_file, 'r', encoding='utf-8') as f:
                cases = json.load(f)
            
            all_qa_pairs = []
            
            logger.info(f"Processing {len(cases)} housing cases for Q&A generation")
            
            for case in cases:
                qa_pairs = self.generate_case_law_qa_pairs(
                    case.get('title', ''),
                    case.get('summary', ''),
                    case.get('content', '')
                )
                
                all_qa_pairs.extend(qa_pairs)
            
            return all_qa_pairs
            
        except Exception as e:
            logger.error(f"Error processing case law file {case_law_file}: {e}")
            return []
    
    def process_all_housing_sources(self, legislation_dir: str, case_law_file: str = None, 
                                  output_file: str = "housing_qa_dataset.json"):
        """Process all housing sources and generate comprehensive Q&A dataset"""
        logger.info("=== STARTING HOUSING Q&A GENERATION ===")
        
        all_qa_pairs = []
        
        try:
            # Phase 1: Process legislation
            logger.info("Phase 1: Processing housing legislation...")
            self.controller.set_current_phase('legislation_processing', {'step': 'housing_acts'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            legislation_path = Path(legislation_dir)
            text_files = []
            if (legislation_path / "text").exists():
                text_files.extend(list((legislation_path / "text").glob("*.txt")))
            text_files.extend(list(legislation_path.glob("*.txt")))
            
            if text_files:
                logger.info(f"Processing {len(text_files)} housing legislation files")
                for file_path in text_files:
                    # Check for pause commands periodically
                    self.controller.check_for_commands()
                    self.controller.wait_while_paused()
                    
                    qa_pairs = self.process_housing_legislation_file(file_path)
                    all_qa_pairs.extend(qa_pairs)
            
            # Phase 2: Process case law if available
            if case_law_file and Path(case_law_file).exists():
                logger.info("Phase 2: Processing housing case law...")
                self.controller.set_current_phase('case_law_processing', {'step': 'housing_cases'})
                self.controller.check_for_commands()
                self.controller.wait_while_paused()
                
                case_qa_pairs = self.process_housing_case_law(case_law_file)
                all_qa_pairs.extend(case_qa_pairs)
            
            # Phase 3: Save results
            logger.info("Phase 3: Saving Q&A datasets...")
            self.controller.set_current_phase('dataset_saving', {'step': 'output_generation'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            # Save results
            output_path = Path(output_file)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated {len(all_qa_pairs)} housing Q&A pairs")
            logger.info(f"Results saved to {output_path.absolute()}")
            
            # Generate specialised housing summary
            self.generate_housing_qa_summary(all_qa_pairs, output_path.parent / "housing_qa_summary.json")
            
            return all_qa_pairs
            
        except Exception as e:
            logger.error(f"Housing Q&A generation failed: {e}")
            raise
        finally:
            # Cleanup controller
            self.controller.cleanup()
    
    def generate_housing_qa_summary(self, qa_pairs: List[Dict[str, str]], output_file: Path):
        """Generate summary statistics for housing Q&A dataset"""
        stats = {
            'total_qa_pairs': len(qa_pairs),
            'housing_categories': {},
            'source_types': {},
            'legislation_breakdown': {},
            'case_law_breakdown': {},
            'avg_question_length': sum(len(qa['question']) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            'avg_answer_length': sum(len(qa['answer']) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0
        }
        
        # Analyze by housing category
        for qa in qa_pairs:
            category = qa.get('housing_category', 'unknown')
            stats['housing_categories'][category] = stats['housing_categories'].get(category, 0) + 1
            
            source_type = qa.get('source_type', 'unknown')
            stats['source_types'][source_type] = stats['source_types'].get(source_type, 0) + 1
            
            if source_type == 'legislation':
                act_name = qa.get('act_name', 'unknown')
                stats['legislation_breakdown'][act_name] = stats['legislation_breakdown'].get(act_name, 0) + 1
            elif source_type == 'case_law':
                case_title = qa.get('case_title', 'unknown')
                stats['case_law_breakdown'][case_title] = stats['case_law_breakdown'].get(case_title, 0) + 1
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Housing Q&A summary saved to {output_file}")

def main():
    """Main function for housing Q&A generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate housing-specific Q&A pairs")
    parser.add_argument('--legislation-dir', default='housing_legislation',
                       help='Directory containing housing legislation')
    parser.add_argument('--case-law-file', default='housing_case_law/housing_cases.json',
                       help='JSON file containing housing case law')
    parser.add_argument('--output-file', default='housing_qa_dataset.json',
                       help='Output file for housing Q&A pairs')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    generator = HousingQAGenerator()
    generator.process_all_housing_sources(
        args.legislation_dir,
        args.case_law_file,
        args.output_file
    )

if __name__ == "__main__":
    main()
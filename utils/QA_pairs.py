#!/usr/bin/env python3
"""
QA Pairs Generator for UK Legislation
Generates question-answer pairs from UK legislation sections for LLM training
"""

import os
import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import anthropic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UKLegislationQAGenerator:
    def __init__(self, anthropic_api_key: Optional[str] = None):
        """Initialize the QA generator with Anthropic client"""
        self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        
    def extract_act_name(self, text: str) -> str:
        """Extract act name from legislation text"""
        lines = text.split('\n')[:10]  # Check first 10 lines
        for line in lines:
            if any(keyword in line.lower() for keyword in ['act', 'regulation', 'order']):
                return line.strip()
        return "UK Legislation"
    
    def parse_sections(self, text: str) -> List[Dict[str, str]]:
        """Parse legislation text into sections"""
        sections = []
        
        # Common section patterns in UK legislation
        section_patterns = [
            r'^\d+\.?\s+.*?(?=^\d+\.?\s+|\Z)',  # Numbered sections
            r'^Section\s+\d+.*?(?=^Section\s+\d+|\Z)',  # "Section X" format
            r'^\(\d+\).*?(?=^\(\d+\)|\Z)',  # Subsections (1), (2), etc.
            r'^[A-Z][^.]*\.\s*$.*?(?=^[A-Z][^.]*\.\s*$|\Z)'  # Titled sections
        ]
        
        # Try to split by sections
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            if matches and len(matches) > 1:
                for i, match in enumerate(matches):
                    if len(match.strip()) > 100:  # Only include substantial sections
                        sections.append({
                            'section_number': str(i + 1),
                            'content': match.strip()
                        })
                break
        
        # Fallback: split by paragraphs if no clear sections found
        if not sections:
            paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 100]
            for i, paragraph in enumerate(paragraphs[:20]):  # Limit to 20 paragraphs
                sections.append({
                    'section_number': f"Para {i + 1}",
                    'content': paragraph
                })
        
        return sections
    
    def generate_qa_pairs(self, section_text: str, act_name: str, section_number: str, 
                         model: str = "claude-3-5-sonnet-20241022") -> List[Dict[str, str]]:
        """Generate Q&A pairs for a specific section"""
        
        prompt = f"""You are a legal expert specializing in UK law. For the following section of the {act_name}, generate a set of question-answer pairs that cover the key provisions, scope, exceptions, procedures, and penalties. Ensure each question is specific and the answer is concise and accurate.

Generate 3-5 question-answer pairs in this exact JSON format:
[
    {{"question": "What does this section establish?", "answer": "This section establishes..."}},
    {{"question": "Who does this provision apply to?", "answer": "This provision applies to..."}}
]

Section {section_number}:
{section_text}

Return only the JSON array, no additional text."""

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=2000,
                temperature=0.3,
                system="You are a legal analyst specializing in UK legislation. Generate precise question-answer pairs in JSON format.",
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
                    return qa_pairs
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON decode error for section {section_number}: {e}")
                    return []
            else:
                logger.warning(f"No JSON found in response for section {section_number}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating QA pairs for section {section_number}: {e}")
            return []
    
    def process_legislation_file(self, file_path: Path) -> List[Dict[str, str]]:
        """Process a single legislation file and generate Q&A pairs"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            act_name = self.extract_act_name(text)
            sections = self.parse_sections(text)
            
            all_qa_pairs = []
            
            logger.info(f"Processing {act_name} with {len(sections)} sections")
            
            for section in sections:
                qa_pairs = self.generate_qa_pairs(
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
                        'source_file': str(file_path.name)
                    })
            
            return all_qa_pairs
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return []
    
    def process_all_legislation(self, source_dir: str, output_file: str = "qa_pairs_dataset.json"):
        """Process all legislation files and generate complete Q&A dataset"""
        source_path = Path(source_dir)
        
        # Look for text files
        text_files = []
        if (source_path / "text").exists():
            text_files.extend(list((source_path / "text").glob("*.txt")))
        
        # Also check root directory
        text_files.extend(list(source_path.glob("*.txt")))
        
        if not text_files:
            logger.error(f"No text files found in {source_dir}")
            return
        
        logger.info(f"Found {len(text_files)} legislation files to process")
        
        all_qa_pairs = []
        processed_count = 0
        
        for file_path in text_files:
            logger.info(f"Processing {file_path.name}...")
            qa_pairs = self.process_legislation_file(file_path)
            all_qa_pairs.extend(qa_pairs)
            processed_count += 1
            
            if processed_count % 10 == 0:
                logger.info(f"Processed {processed_count}/{len(text_files)} files, generated {len(all_qa_pairs)} Q&A pairs so far")
        
        # Save results
        output_path = Path(output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Generated {len(all_qa_pairs)} Q&A pairs from {processed_count} files")
        logger.info(f"Results saved to {output_path.absolute()}")
        
        # Generate summary statistics
        self.generate_summary_stats(all_qa_pairs, output_path.parent / "qa_summary.json")
        
        return all_qa_pairs
    
    def generate_summary_stats(self, qa_pairs: List[Dict[str, str]], output_file: Path):
        """Generate summary statistics for the Q&A dataset"""
        stats = {
            'total_qa_pairs': len(qa_pairs),
            'unique_acts': len(set(qa['act_name'] for qa in qa_pairs)),
            'unique_source_files': len(set(qa['source_file'] for qa in qa_pairs)),
            'avg_question_length': sum(len(qa['question']) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            'avg_answer_length': sum(len(qa['answer']) for qa in qa_pairs) / len(qa_pairs) if qa_pairs else 0,
            'acts_breakdown': {}
        }
        
        # Count Q&A pairs per act
        for qa in qa_pairs:
            act_name = qa['act_name']
            if act_name not in stats['acts_breakdown']:
                stats['acts_breakdown'][act_name] = 0
            stats['acts_breakdown'][act_name] += 1
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Summary statistics saved to {output_file}")

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate Q&A pairs from UK legislation")
    parser.add_argument('--source-dir', default='uk_legislation', 
                       help='Directory containing legislation files')
    parser.add_argument('--output-file', default='qa_pairs_dataset.json',
                       help='Output file for Q&A pairs')
    
    args = parser.parse_args()
    
    generator = UKLegislationQAGenerator()
    generator.process_all_legislation(args.source_dir, args.output_file)

if __name__ == "__main__":
    main()

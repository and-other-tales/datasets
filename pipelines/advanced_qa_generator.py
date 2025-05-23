#!/usr/bin/env python3
"""
Legal Llama Advanced Q&A Generator

Generates sophisticated multi-step reasoning Q&A pairs for legal and tax domains,
designed specifically for training Legal Llama models that can handle complex professional scenarios.
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
import re

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.pipeline_controller import PipelineController, create_database_update_callback, create_dataset_creation_callback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LegalLlamaAdvancedQAGenerator:
    def __init__(self, input_dir: str, output_dir: str = "generated/advanced_qa"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline controller for pause functionality
        self.controller = PipelineController()
        
        # Register pause functionality callbacks
        self.controller.register_callback('database_update', create_database_update_callback(self))
        self.controller.register_callback('dataset_creation', create_dataset_creation_callback(self))
        
        logger.info("🔶 Pipeline Control: Press P to pause/resume, A to update databases (when paused), D to create dataset (when paused), Q to quit")
        
        # Advanced Q&A patterns for different reasoning types
        self.qa_patterns = {
            "multi_step_legal": [
                {
                    "pattern": "chain_of_reasoning",
                    "template": "A client asks: '{question}'. Walk through your legal analysis step by step, citing relevant authorities and explaining your reasoning at each stage.",
                    "requires": ["legal_principle", "case_law", "statutory_provision"]
                },
                {
                    "pattern": "comparative_analysis",
                    "template": "Compare and contrast the legal positions in {case_a} and {case_b}. How would these precedents apply to {scenario}?",
                    "requires": ["multiple_cases", "factual_scenario"]
                },
                {
                    "pattern": "procedural_strategy",
                    "template": "Your client faces {legal_challenge}. Outline your procedural strategy, including timing, evidence gathering, and potential obstacles.",
                    "requires": ["court_procedure", "evidence_rules"]
                }
            ],
            "multi_step_tax": [
                {
                    "pattern": "calculation_with_alternatives",
                    "template": "Calculate the tax liability for {scenario} and explain three different strategies to legally minimise this liability.",
                    "requires": ["tax_calculation", "optimisation_strategies"]
                },
                {
                    "pattern": "compliance_assessment",
                    "template": "Assess the tax compliance requirements for {business_scenario} and identify potential risks and mitigation strategies.",
                    "requires": ["compliance_rules", "risk_assessment"]
                },
                {
                    "pattern": "cross_tax_analysis",
                    "template": "Analyze the interaction between {tax_type_1} and {tax_type_2} for {taxpayer_scenario} and recommend the optimal approach.",
                    "requires": ["multiple_tax_types", "optimisation"]
                }
            ],
            "adversarial_legal": [
                {
                    "pattern": "defense_construction",
                    "template": "The prosecution argues {prosecution_argument}. Construct a comprehensive defense strategy addressing each element of their case.",
                    "requires": ["prosecution_case", "defense_elements"]
                },
                {
                    "pattern": "precedent_challenge",
                    "template": "Opposing counsel cites {cited_case} to support their position. Distinguish this case or challenge its applicability to our facts.",
                    "requires": ["opposing_precedent", "distinguishing_factors"]
                }
            ],
            "practical_application": [
                {
                    "pattern": "client_interview",
                    "template": "During your initial client consultation for {case_type}, what are the key questions you would ask and why?",
                    "requires": ["case_type", "information_gathering"]
                },
                {
                    "pattern": "document_drafting",
                    "template": "Draft the key provisions for {document_type} that address {legal_issue}, explaining your drafting choices.",
                    "requires": ["document_type", "legal_provisions"]
                }
            ]
        }
        
        # Domain-specific question types
        self.legal_question_types = [
            "statutory_interpretation",
            "case_law_application",
            "procedural_compliance",
            "evidence_assessment",
            "defense_strategy",
            "civil_procedure",
            "criminal_procedure",
            "contract_analysis",
            "tort_liability",
            "constitutional_rights"
        ]
        
        self.tax_question_types = [
            "income_tax_calculation",
            "vat_compliance",
            "corporation_tax_planning",
            "capital_gains_optimization",
            "inheritance_tax_planning",
            "tax_investigation_response",
            "making_tax_digital_compliance",
            "international_tax_issues",
            "tax_efficient_structures",
            "penalty_mitigation"
        ]
    
    def extract_key_concepts(self, content: str) -> List[str]:
        """Extract key legal/tax concepts from content"""
        # Simple keyword extraction - could be enhanced with NLP
        legal_keywords = [
            "section", "act", "regulation", "case", "precedent", "court",
            "defendant", "plaintiff", "liability", "damages", "contract",
            "tort", "statutory", "common law", "jurisdiction", "appeal"
        ]
        
        tax_keywords = [
            "income tax", "vat", "corporation tax", "capital gains", "allowance",
            "relief", "deduction", "exemption", "rate", "threshold", "liability",
            "compliance", "filing", "payment", "penalty", "hmrc"
        ]
        
        found_concepts = []
        content_lower = content.lower()
        
        for keyword in legal_keywords + tax_keywords:
            if keyword in content_lower:
                found_concepts.append(keyword)
        
        return list(set(found_concepts))  # Remove duplicates
    
    def generate_multi_step_qa(self, source_data: Dict[str, Any], domain: str) -> List[Dict[str, Any]]:
        """Generate multi-step reasoning Q&A pairs"""
        qa_pairs = []
        
        content = source_data.get('content', '')
        title = source_data.get('title', 'Unknown')
        
        if len(content) < 100:  # Skip very short content
            return qa_pairs
        
        # Select appropriate patterns based on domain
        pattern_key = f"multi_step_{domain}"
        if pattern_key not in self.qa_patterns:
            return qa_pairs
        
        patterns = self.qa_patterns[pattern_key]
        
        for pattern in patterns[:2]:  # Use first 2 patterns
            try:
                # Generate context-specific question
                question = self._generate_contextual_question(pattern, source_data, domain)
                if not question:
                    continue
                
                # Generate comprehensive answer with reasoning steps
                answer = self._generate_multi_step_answer(question, source_data, domain, pattern)
                
                qa_pair = {
                    "question": question,
                    "answer": answer,
                    "reasoning_type": pattern["pattern"],
                    "domain": domain,
                    "source_title": title,
                    "complexity": "advanced",
                    "multi_step": True,
                    "concepts": self.extract_key_concepts(content),
                    "metadata": source_data.get('metadata', {})
                }
                
                qa_pairs.append(qa_pair)
                
            except Exception as e:
                logger.warning(f"Error generating QA pair: {e}")
                continue
        
        return qa_pairs
    
    def generate_adversarial_qa(self, source_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate adversarial Q&A pairs for challenge scenarios"""
        qa_pairs = []
        
        patterns = self.qa_patterns["adversarial_legal"]
        
        for pattern in patterns:
            try:
                question = self._generate_adversarial_question(pattern, source_data)
                if not question:
                    continue
                
                answer = self._generate_adversarial_answer(question, source_data, pattern)
                
                qa_pair = {
                    "question": question,
                    "answer": answer,
                    "reasoning_type": pattern["pattern"],
                    "domain": "legal",
                    "source_title": source_data.get('title', 'Unknown'),
                    "complexity": "adversarial",
                    "challenge_type": pattern["pattern"],
                    "metadata": source_data.get('metadata', {})
                }
                
                qa_pairs.append(qa_pair)
                
            except Exception as e:
                logger.warning(f"Error generating adversarial QA: {e}")
                continue
        
        return qa_pairs
    
    def generate_practical_qa(self, source_data: Dict[str, Any], domain: str) -> List[Dict[str, Any]]:
        """Generate practical application Q&A pairs"""
        qa_pairs = []
        
        patterns = self.qa_patterns["practical_application"]
        
        for pattern in patterns:
            try:
                question = self._generate_practical_question(pattern, source_data, domain)
                if not question:
                    continue
                
                answer = self._generate_practical_answer(question, source_data, domain, pattern)
                
                qa_pair = {
                    "question": question,
                    "answer": answer,
                    "reasoning_type": pattern["pattern"],
                    "domain": domain,
                    "source_title": source_data.get('title', 'Unknown'),
                    "complexity": "practical",
                    "application_type": pattern["pattern"],
                    "metadata": source_data.get('metadata', {})
                }
                
                qa_pairs.append(qa_pair)
                
            except Exception as e:
                logger.warning(f"Error generating practical QA: {e}")
                continue
        
        return qa_pairs
    
    def _generate_contextual_question(self, pattern: Dict[str, Any], source_data: Dict[str, Any], domain: str) -> str:
        """Generate a contextual question based on the pattern and source data"""
        title = source_data.get('title', 'Unknown')
        content = source_data.get('content', '')
        
        # Extract scenario from content (simplified)
        scenario = content[:200] + "..." if len(content) > 200 else content
        
        if pattern["pattern"] == "chain_of_reasoning":
            if domain == "legal":
                return f"A client's situation involves the principles discussed in '{title}'. They ask: 'How does this law apply to my case where {scenario}?' Provide a comprehensive legal analysis."
            else:  # tax
                return f"Based on the guidance in '{title}', explain how the tax rules apply to a situation where {scenario}"
        
        elif pattern["pattern"] == "comparative_analysis":
            return f"Compare the approach taken in '{title}' with similar cases and explain how this affects a client facing {scenario}"
        
        elif pattern["pattern"] == "calculation_with_alternatives":
            return f"Using the principles from '{title}', calculate the tax implications for {scenario} and provide three optimization strategies."
        
        # Default fallback
        return f"Analyze '{title}' and explain its application to {scenario}"
    
    def _generate_multi_step_answer(self, question: str, source_data: Dict[str, Any], domain: str, pattern: Dict[str, Any]) -> str:
        """Generate a comprehensive multi-step answer"""
        
        answer_template = f"""
**Step 1: Legal/Tax Framework Analysis**
[Identify the relevant legal principles or tax rules from {source_data.get('title', 'the source material')}]

**Step 2: Factual Analysis**
[Analyze the key facts and their legal/tax significance]

**Step 3: Application of Law/Rules**
[Apply the identified principles to the specific scenario]

**Step 4: Consideration of Alternatives**
[Explore alternative approaches or interpretations]

**Step 5: Conclusion and Recommendations**
[Provide clear conclusions and actionable recommendations]

**Supporting Authorities:**
- {source_data.get('title', 'Primary source')}
- [Additional relevant authorities would be cited here]

**Risk Assessment:**
[Identify potential risks and mitigation strategies]

**Next Steps:**
[Outline recommended actions and timing]
"""
        
        return answer_template.strip()
    
    def _generate_adversarial_question(self, pattern: Dict[str, Any], source_data: Dict[str, Any]) -> str:
        """Generate adversarial challenge questions"""
        title = source_data.get('title', 'Unknown Case')
        
        if pattern["pattern"] == "defense_construction":
            return f"The prosecution in a case similar to '{title}' argues that the defendant clearly violated the statute. Construct a comprehensive defense strategy."
        
        elif pattern["pattern"] == "precedent_challenge":
            return f"Opposing counsel cites '{title}' to support their position. How would you distinguish this case or challenge its applicability?"
        
        return f"Challenge the legal reasoning in '{title}' and present alternative interpretations."
    
    def _generate_adversarial_answer(self, question: str, source_data: Dict[str, Any], pattern: Dict[str, Any]) -> str:
        """Generate adversarial response strategies"""
        
        answer_template = f"""
**Adversarial Analysis:**

**1. Opposing Position Assessment:**
[Analyze the strength and weaknesses of the opposing argument]

**2. Counter-Arguments:**
[Develop specific counter-arguments addressing each element]

**3. Alternative Legal Interpretations:**
[Present alternative ways to interpret the law or facts]

**4. Distinguishing Factors:**
[Identify factors that distinguish this case from cited precedents]

**5. Strategic Response:**
[Outline the strategic approach to counter the opposition]

**6. Supporting Evidence:**
[Identify evidence needed to support the counter-position]

**Risk/Benefit Analysis:**
[Assess the risks and benefits of each counter-strategy]
"""
        
        return answer_template.strip()
    
    def _generate_practical_question(self, pattern: Dict[str, Any], source_data: Dict[str, Any], domain: str) -> str:
        """Generate practical application questions"""
        title = source_data.get('title', 'Unknown')
        
        if pattern["pattern"] == "client_interview":
            return f"You're meeting a new client whose case involves the issues discussed in '{title}'. What key questions would you ask during the initial consultation?"
        
        elif pattern["pattern"] == "document_drafting":
            return f"Draft key contract clauses that address the legal issues covered in '{title}', explaining your drafting choices."
        
        return f"How would you practically apply the principles from '{title}' in a real-world {domain} scenario?"
    
    def _generate_practical_answer(self, question: str, source_data: Dict[str, Any], domain: str, pattern: Dict[str, Any]) -> str:
        """Generate practical application answers"""
        
        answer_template = f"""
**Practical Application Guide:**

**1. Initial Assessment:**
[Key considerations for the practical scenario]

**2. Information Gathering:**
[What information is needed and how to obtain it]

**3. Legal/Tax Strategy:**
[Develop the appropriate professional strategy]

**4. Implementation Steps:**
[Specific steps to implement the strategy]

**5. Documentation Requirements:**
[What documents need to be prepared or obtained]

**6. Timeline and Deadlines:**
[Key dates and deadlines to consider]

**7. Risk Management:**
[Potential risks and how to mitigate them]

**8. Client Communication:**
[How to explain the approach and options to the client]
"""
        
        return answer_template.strip()
    
    def process_legal_sources(self) -> List[Dict[str, Any]]:
        """Process legal sources to generate advanced Q&A"""
        qa_pairs = []
        
        # Process case law
        case_dirs = [
            self.input_dir / "housing_case_law",
            self.input_dir / "case_law",
            self.input_dir / "bailii_cases"
        ]
        
        for case_dir in case_dirs:
            if not case_dir.exists():
                continue
            
            qa_pairs.extend(self._process_directory(case_dir, "legal"))
        
        # Process legislation
        legislation_dirs = [
            self.input_dir / "housing_legislation",
            self.input_dir / "uk_legislation"
        ]
        
        for leg_dir in legislation_dirs:
            if not leg_dir.exists():
                continue
            
            qa_pairs.extend(self._process_directory(leg_dir, "legal"))
        
        return qa_pairs
    
    def process_tax_sources(self) -> List[Dict[str, Any]]:
        """Process tax sources to generate advanced Q&A"""
        qa_pairs = []
        
        # Process HMRC documentation
        hmrc_dirs = [
            self.input_dir / "hmrc_documentation",
            self.input_dir / "test_hmrc"
        ]
        
        for hmrc_dir in hmrc_dirs:
            if not hmrc_dir.exists():
                continue
            
            qa_pairs.extend(self._process_directory(hmrc_dir, "tax"))
        
        return qa_pairs
    
    def _process_directory(self, directory: Path, domain: str) -> List[Dict[str, Any]]:
        """Process a directory of legal/tax documents"""
        qa_pairs = []
        
        metadata_dir = directory / "metadata"
        text_dir = directory / "text"
        
        if not metadata_dir.exists() or not text_dir.exists():
            return qa_pairs
        
        for metadata_file in metadata_dir.glob("*.json"):
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    source_data = json.load(f)
                
                # Read corresponding text file
                text_file = text_dir / f"{metadata_file.stem}.txt"
                if text_file.exists():
                    with open(text_file, 'r', encoding='utf-8') as f:
                        source_data['content'] = f.read()
                
                # Generate different types of Q&A
                multi_step_qa = self.generate_multi_step_qa(source_data, domain)
                practical_qa = self.generate_practical_qa(source_data, domain)
                
                qa_pairs.extend(multi_step_qa)
                qa_pairs.extend(practical_qa)
                
                # Generate adversarial Q&A for legal domain
                if domain == "legal":
                    adversarial_qa = self.generate_adversarial_qa(source_data)
                    qa_pairs.extend(adversarial_qa)
                
            except Exception as e:
                logger.error(f"Error processing {metadata_file}: {e}")
                continue
        
        return qa_pairs
    
    def generate_advanced_qa_dataset(self) -> Dict[str, int]:
        """Generate the complete advanced Q&A dataset"""
        logger.info("=== STARTING LEGAL LLAMA ADVANCED Q&A GENERATION ===")
        
        stats = {
            "legal_qa": 0,
            "tax_qa": 0,
            "multi_step": 0,
            "adversarial": 0,
            "practical": 0,
            "total_qa": 0
        }
        
        try:
            # Phase 1: Process legal sources
            logger.info("Phase 1: Processing legal sources...")
            self.controller.set_current_phase('legal_processing', {'step': 'case_law_analysis'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            legal_qa = self.process_legal_sources()
            
            # Phase 2: Process tax sources
            logger.info("Phase 2: Processing tax sources...")
            self.controller.set_current_phase('tax_processing', {'step': 'hmrc_analysis'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            tax_qa = self.process_tax_sources()
            
            # Phase 3: Combine and organize Q&A pairs
            logger.info("Phase 3: Combining and organizing Q&A pairs...")
            self.controller.set_current_phase('qa_organization', {'step': 'dataset_creation'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            # Combine all Q&A pairs
            all_qa = legal_qa + tax_qa
            
            # Calculate statistics
            stats["legal_qa"] = len(legal_qa)
            stats["tax_qa"] = len(tax_qa)
            stats["total_qa"] = len(all_qa)
            
            for qa in all_qa:
                if qa.get("multi_step"):
                    stats["multi_step"] += 1
                if qa.get("complexity") == "adversarial":
                    stats["adversarial"] += 1
                if qa.get("complexity") == "practical":
                    stats["practical"] += 1
            
            # Phase 4: Save datasets
            logger.info("Phase 4: Saving advanced Q&A datasets...")
            self.controller.set_current_phase('dataset_saving', {'step': 'file_output'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            # Save complete dataset
            output_file = self.output_dir / "advanced_qa_dataset.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_qa, f, indent=2, ensure_ascii=False)
            
            # Save by domain
            legal_file = self.output_dir / "legal_advanced_qa.json"
            with open(legal_file, 'w', encoding='utf-8') as f:
                json.dump(legal_qa, f, indent=2, ensure_ascii=False)
            
            tax_file = self.output_dir / "tax_advanced_qa.json"
            with open(tax_file, 'w', encoding='utf-8') as f:
                json.dump(tax_qa, f, indent=2, ensure_ascii=False)
            
            # Save statistics
            stats_file = self.output_dir / "advanced_qa_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Advanced Q&A dataset created with {stats['total_qa']} pairs")
            return stats
            
        except Exception as e:
            logger.error(f"Advanced Q&A generation failed: {e}")
            raise
        finally:
            # Cleanup controller
            self.controller.cleanup()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate advanced Q&A for Legal Llama training")
    parser.add_argument('--input-dir', default='generated',
                       help='Input directory containing legal and tax data')
    parser.add_argument('--output-dir', default='generated/advanced_qa',
                       help='Output directory for advanced Q&A dataset')
    
    args = parser.parse_args()
    
    generator = LegalLlamaAdvancedQAGenerator(args.input_dir, args.output_dir)
    stats = generator.generate_advanced_qa_dataset()
    
    print(f"\n=== Legal Llama Advanced Q&A Generation Complete ===")
    print(f"Legal Q&A pairs: {stats['legal_qa']}")
    print(f"Tax Q&A pairs: {stats['tax_qa']}")
    print(f"Multi-step reasoning: {stats['multi_step']}")
    print(f"Adversarial scenarios: {stats['adversarial']}")
    print(f"Practical applications: {stats['practical']}")
    print(f"Total Q&A pairs: {stats['total_qa']}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
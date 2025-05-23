#!/usr/bin/env python3
"""
Legal Training Framework
Production-ready legal-specific training formats, evaluation metrics,
and professional standards compliance for LLM fine-tuning.
"""

import json
import logging
import numpy as np
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from collections import defaultdict

try:
    from transformers import AutoTokenizer, Trainer, TrainingArguments
    from datasets import Dataset, DatasetDict
    from sklearn.metrics import accuracy_score, f1_score
except ImportError:
    logging.warning("Training dependencies not available. Install with: pip install transformers datasets scikit-learn")

from .legal_metadata import LegalMetadata, LegalCitation
from .hmrc_metadata import HMRCMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalTrainingExample:
    """Enhanced legal training example with professional standards"""
    input_context: str
    retrieved_authorities: List[Dict]
    expected_output: str
    legal_area: str
    difficulty_level: str  # "foundation", "reasoning", "expert", "adversarial"
    example_type: str  # "positive", "negative", "contrastive"
    citation_accuracy: Dict
    reasoning_chain: List[str] = field(default_factory=list)
    jurisdiction: str = "england_wales"
    authority_sources: List[str] = field(default_factory=list)
    professional_standards: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)

@dataclass
class LegalEvaluationMetrics:
    """Comprehensive legal evaluation metrics"""
    citation_accuracy: float
    legal_structure_score: float
    authority_hierarchy_score: float
    reasoning_coherence: float
    professional_language_score: float
    oscola_compliance: float
    factual_accuracy: float
    disclaimer_compliance: float
    overall_score: float

class LegalPromptTemplate:
    """
    Professional legal prompt templates following GUIDANCE.md specifications
    """
    
    def __init__(self):
        self.templates = {
            "legal_analysis": {
                "system": """You are a highly qualified UK legal specialist with comprehensive knowledge of all UK legislation, case law, and legal procedures. You provide accurate, detailed legal analysis following these professional standards:

1. Cite relevant statutory provisions and case law using OSCOLA format
2. Structure responses with clear legal reasoning and logical progression
3. Distinguish between different jurisdictions where relevant (England & Wales, Scotland, Northern Ireland)
4. Include confidence indicators for complex interpretations
5. Always include appropriate disclaimers about seeking professional legal advice
6. Maintain professional legal language and terminology
7. Consider precedent hierarchy and authority levels when citing cases""",
                
                "user_template": """Legal Context: {input_context}

Relevant Authorities:
{authorities_section}

Legal Question: {instruction}

Please provide a comprehensive legal analysis addressing:
- The key legal issues arising
- Applicable statutory provisions and case law
- Reasoning and interpretation
- Practical implications
- Any limitations or uncertainties in the analysis

Your analysis should follow professional legal writing standards with proper citations."""
            },
            
            "counter_argument": {
                "system": """You are a qualified UK legal professional specializing in defense advocacy and counter-argument development. Your role is to identify and develop strong legal counter-arguments while maintaining the highest professional standards.

Professional Requirements:
1. Use OSCOLA citation format for all legal authorities
2. Structure arguments with clear legal reasoning
3. Distinguish strong arguments from weak ones
4. Consider procedural and evidential challenges
5. Assess the strength of opposing positions objectively
6. Include appropriate professional disclaimers""",
                
                "user_template": """Case Bundle Summary: {input_context}

Relevant Legal Authorities:
{authorities_section}

Prosecution/Opposing Position: {opposing_argument}

Task: Develop comprehensive counter-arguments addressing:
1. Legal challenges to the opposing position
2. Alternative interpretations of relevant law
3. Procedural or evidential weaknesses
4. Precedent that supports the defense position
5. Strategic considerations for the defense

Provide a structured legal response with proper citations and professional reasoning."""
            },
            
            "tax_compliance": {
                "system": """You are an expert UK tax advisor with complete knowledge of all HMRC rules, regulations, and tax optimization strategies. You ensure full tax compliance while maximizing legitimate tax savings.

Professional Standards:
1. Reference specific HMRC guidance using manual codes (e.g., CG12345)
2. Cite relevant legislation (ITA 2007, TCGA 1992, etc.) accurately
3. Distinguish between tax avoidance and tax evasion clearly
4. Provide practical compliance guidance
5. Include appropriate tax disclaimers
6. Consider anti-avoidance legislation (GAAR, specific rules)""",
                
                "user_template": """Tax Scenario: {input_context}

Relevant Tax Authorities:
{authorities_section}

Tax Question: {instruction}

Please provide comprehensive tax advice covering:
- Applicable tax legislation and HMRC guidance
- Compliance obligations and deadlines
- Legitimate tax planning opportunities
- Risk assessment and mitigation
- Practical implementation steps
- Professional disclaimers regarding tax advice

Ensure all advice promotes compliance while optimizing the tax position."""
            }
        }
    
    def format_authorities_section(self, authorities: List[Dict]) -> str:
        """Format legal authorities using professional citation standards"""
        if not authorities:
            return "No specific authorities provided."
        
        formatted_authorities = []
        
        for i, auth in enumerate(authorities, 1):
            authority_text = f"{i}. "
            
            # Add citation if available
            if auth.get('citation'):
                authority_text += f"{auth['citation']} - "
            
            # Add title
            if auth.get('title'):
                authority_text += f"{auth['title']}"
            
            # Add authority level context
            if auth.get('authority_level'):
                authority_text += f" [Authority Level: {auth['authority_level']}]"
            
            # Add relevant text excerpt
            if auth.get('text'):
                excerpt = auth['text'][:300] + "..." if len(auth['text']) > 300 else auth['text']
                authority_text += f"\n   Relevant text: {excerpt}"
            
            formatted_authorities.append(authority_text)
        
        return "\n\n".join(formatted_authorities)
    
    def create_training_prompt(
        self, 
        template_type: str, 
        input_context: str, 
        instruction: str,
        authorities: List[Dict],
        opposing_argument: str = ""
    ) -> str:
        """Create a complete training prompt using professional templates"""
        
        if template_type not in self.templates:
            raise ValueError(f"Unknown template type: {template_type}")
        
        template = self.templates[template_type]
        authorities_section = self.format_authorities_section(authorities)
        
        # Format user message
        user_content = template["user_template"].format(
            input_context=input_context,
            authorities_section=authorities_section,
            instruction=instruction,
            opposing_argument=opposing_argument
        )
        
        # Create complete prompt for Llama format
        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{template["system"]}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        return full_prompt

class LegalTrainingDatasetCreator:
    """
    Create professional legal training datasets with enhanced examples
    """
    
    def __init__(self):
        self.prompt_template = LegalPromptTemplate()
        self.citation_patterns = {
            'neutral_citation': re.compile(r'\[(\d{4})\]\s+(UKSC|UKHL|UKPC|EWCA|EWHC|EWCOP|CSIH|CSOH|NICA|NIQB)\s+(\d+)'),
            'law_reports': re.compile(r'\[(\d{4})\]\s+(\d+)\s+(AC|WLR|All\s+ER|QB|Ch|Fam)'),
            'legislation': re.compile(r'\b((?:TCGA|ITA|ITTOIA|CTA|IHTA|FA|VATA|TMA)\s+\d{4})\b'),
            'hmrc_manual': re.compile(r'\b([A-Z]{2,4}\d{5}[A-Z]*)\b')
        }
    
    def create_foundation_examples(
        self, 
        legal_documents: List[Tuple[str, Union[LegalMetadata, HMRCMetadata], str]]
    ) -> List[LegalTrainingExample]:
        """Create foundation-level training examples"""
        examples = []
        
        for doc_id, metadata, content in legal_documents:
            # Extract key concepts and create basic Q&A
            foundation_examples = self._generate_foundation_qa(doc_id, metadata, content)
            examples.extend(foundation_examples)
        
        return examples
    
    def create_reasoning_examples(
        self,
        legal_documents: List[Tuple[str, Union[LegalMetadata, HMRCMetadata], str]]
    ) -> List[LegalTrainingExample]:
        """Create reasoning-chain training examples"""
        examples = []
        
        for doc_id, metadata, content in legal_documents:
            # Create multi-step reasoning scenarios
            reasoning_examples = self._generate_reasoning_scenarios(doc_id, metadata, content)
            examples.extend(reasoning_examples)
        
        return examples
    
    def create_expert_examples(
        self,
        legal_documents: List[Tuple[str, Union[LegalMetadata, HMRCMetadata], str]]
    ) -> List[LegalTrainingExample]:
        """Create expert-level application examples"""
        examples = []
        
        for doc_id, metadata, content in legal_documents:
            # Create complex professional scenarios
            expert_examples = self._generate_expert_scenarios(doc_id, metadata, content)
            examples.extend(expert_examples)
        
        return examples
    
    def create_adversarial_examples(
        self,
        legal_documents: List[Tuple[str, Union[LegalMetadata, HMRCMetadata], str]]
    ) -> List[LegalTrainingExample]:
        """Create adversarial challenge examples"""
        examples = []
        
        for doc_id, metadata, content in legal_documents:
            # Create challenging counter-argument scenarios
            adversarial_examples = self._generate_adversarial_scenarios(doc_id, metadata, content)
            examples.extend(adversarial_examples)
        
        return examples
    
    def _generate_foundation_qa(
        self, 
        doc_id: str, 
        metadata: Union[LegalMetadata, HMRCMetadata], 
        content: str
    ) -> List[LegalTrainingExample]:
        """Generate foundation-level Q&A pairs"""
        examples = []
        
        # Determine if this is legal or tax content
        is_tax = isinstance(metadata, HMRCMetadata)
        template_type = "tax_compliance" if is_tax else "legal_analysis"
        
        # Extract title and key sections
        title = metadata.title
        content_excerpt = content[:1000]  # First 1000 chars
        
        # Create basic understanding questions
        basic_questions = self._generate_basic_questions(title, metadata, is_tax)
        
        for question in basic_questions:
            authorities = [{'citation': getattr(metadata, 'citation', title), 'title': title, 'text': content_excerpt}]
            
            prompt = self.prompt_template.create_training_prompt(
                template_type=template_type,
                input_context=f"Document: {title}",
                instruction=question,
                authorities=authorities
            )
            
            # Generate expected response
            expected_output = self._generate_foundation_response(question, metadata, content_excerpt)
            
            example = LegalTrainingExample(
                input_context=prompt,
                retrieved_authorities=authorities,
                expected_output=expected_output,
                legal_area=self._get_legal_area(metadata),
                difficulty_level="foundation",
                example_type="positive",
                citation_accuracy=self._assess_citation_accuracy(expected_output),
                jurisdiction=self._get_jurisdiction(metadata),
                authority_sources=[title],
                professional_standards={'requires_disclaimer': True, 'oscola_format': True}
            )
            examples.append(example)
        
        return examples[:3]  # Limit to 3 examples per document
    
    def _generate_reasoning_scenarios(
        self, 
        doc_id: str, 
        metadata: Union[LegalMetadata, HMRCMetadata], 
        content: str
    ) -> List[LegalTrainingExample]:
        """Generate multi-step reasoning scenarios"""
        examples = []
        
        is_tax = isinstance(metadata, HMRCMetadata)
        template_type = "tax_compliance" if is_tax else "legal_analysis"
        
        # Create complex scenarios requiring multi-step analysis
        reasoning_scenarios = self._generate_reasoning_questions(metadata, content, is_tax)
        
        for scenario in reasoning_scenarios:
            authorities = [{'citation': getattr(metadata, 'citation', metadata.title), 'title': metadata.title, 'text': content[:1500]}]
            
            prompt = self.prompt_template.create_training_prompt(
                template_type=template_type,
                input_context=scenario['context'],
                instruction=scenario['question'],
                authorities=authorities
            )
            
            expected_output = self._generate_reasoning_response(scenario, metadata, content)
            
            example = LegalTrainingExample(
                input_context=prompt,
                retrieved_authorities=authorities,
                expected_output=expected_output,
                legal_area=self._get_legal_area(metadata),
                difficulty_level="reasoning",
                example_type="positive",
                citation_accuracy=self._assess_citation_accuracy(expected_output),
                reasoning_chain=scenario.get('reasoning_steps', []),
                jurisdiction=self._get_jurisdiction(metadata),
                authority_sources=[metadata.title],
                professional_standards={'requires_disclaimer': True, 'oscola_format': True, 'multi_step_reasoning': True}
            )
            examples.append(example)
        
        return examples[:2]  # Limit to 2 examples per document
    
    def _generate_expert_scenarios(
        self, 
        doc_id: str, 
        metadata: Union[LegalMetadata, HMRCMetadata], 
        content: str
    ) -> List[LegalTrainingExample]:
        """Generate expert-level professional scenarios"""
        examples = []
        
        is_tax = isinstance(metadata, HMRCMetadata)
        template_type = "tax_compliance" if is_tax else "legal_analysis"
        
        # Create professional consultation scenarios
        expert_scenarios = self._generate_expert_questions(metadata, content, is_tax)
        
        for scenario in expert_scenarios:
            authorities = [{'citation': getattr(metadata, 'citation', metadata.title), 'title': metadata.title, 'text': content[:2000]}]
            
            prompt = self.prompt_template.create_training_prompt(
                template_type=template_type,
                input_context=scenario['context'],
                instruction=scenario['question'],
                authorities=authorities
            )
            
            expected_output = self._generate_expert_response(scenario, metadata, content)
            
            example = LegalTrainingExample(
                input_context=prompt,
                retrieved_authorities=authorities,
                expected_output=expected_output,
                legal_area=self._get_legal_area(metadata),
                difficulty_level="expert",
                example_type="positive",
                citation_accuracy=self._assess_citation_accuracy(expected_output),
                jurisdiction=self._get_jurisdiction(metadata),
                authority_sources=[metadata.title],
                professional_standards={
                    'requires_disclaimer': True, 
                    'oscola_format': True, 
                    'professional_consultation': True,
                    'strategic_advice': True
                }
            )
            examples.append(example)
        
        return examples[:1]  # Limit to 1 example per document
    
    def _generate_adversarial_scenarios(
        self, 
        doc_id: str, 
        metadata: Union[LegalMetadata, HMRCMetadata], 
        content: str
    ) -> List[LegalTrainingExample]:
        """Generate adversarial challenge scenarios"""
        examples = []
        
        is_tax = isinstance(metadata, HMRCMetadata)
        template_type = "counter_argument" if not is_tax else "tax_compliance"
        
        # Create challenging scenarios
        adversarial_scenarios = self._generate_adversarial_questions(metadata, content, is_tax)
        
        for scenario in adversarial_scenarios:
            authorities = [{'citation': getattr(metadata, 'citation', metadata.title), 'title': metadata.title, 'text': content[:1500]}]
            
            prompt = self.prompt_template.create_training_prompt(
                template_type=template_type,
                input_context=scenario['context'],
                instruction=scenario['question'],
                authorities=authorities,
                opposing_argument=scenario.get('opposing_argument', '')
            )
            
            expected_output = self._generate_adversarial_response(scenario, metadata, content)
            
            example = LegalTrainingExample(
                input_context=prompt,
                retrieved_authorities=authorities,
                expected_output=expected_output,
                legal_area=self._get_legal_area(metadata),
                difficulty_level="adversarial",
                example_type="adversarial",
                citation_accuracy=self._assess_citation_accuracy(expected_output),
                jurisdiction=self._get_jurisdiction(metadata),
                authority_sources=[metadata.title],
                professional_standards={
                    'requires_disclaimer': True, 
                    'oscola_format': True, 
                    'counter_argument': True,
                    'challenge_response': True
                }
            )
            examples.append(example)
        
        return examples[:1]  # Limit to 1 example per document
    
    def _generate_basic_questions(self, title: str, metadata: Union[LegalMetadata, HMRCMetadata], is_tax: bool) -> List[str]:
        """Generate basic understanding questions"""
        if is_tax:
            return [
                f"What are the key tax implications outlined in '{title}'?",
                f"What compliance obligations are established by '{title}'?",
                f"Explain the practical application of the guidance in '{title}'"
            ]
        else:
            return [
                f"What are the main legal principles established in '{title}'?",
                f"Explain the key legal requirements outlined in '{title}'",
                f"What are the practical implications of '{title}' for legal practitioners?"
            ]
    
    def _generate_reasoning_questions(self, metadata: Union[LegalMetadata, HMRCMetadata], content: str, is_tax: bool) -> List[Dict]:
        """Generate multi-step reasoning questions"""
        if is_tax:
            return [{
                'context': f"A client has a complex tax situation involving the principles outlined in '{metadata.title}'",
                'question': "Analyze the tax implications and provide step-by-step compliance guidance",
                'reasoning_steps': ['Identify applicable tax provisions', 'Analyze client facts', 'Apply legal tests', 'Recommend compliance strategy']
            }]
        else:
            return [{
                'context': f"A legal dispute has arisen involving the principles in '{metadata.title}'",
                'question': "Provide a comprehensive legal analysis with step-by-step reasoning",
                'reasoning_steps': ['Identify legal issues', 'Analyze applicable law', 'Apply law to facts', 'Reach legal conclusion']
            }]
    
    def _generate_expert_questions(self, metadata: Union[LegalMetadata, HMRCMetadata], content: str, is_tax: bool) -> List[Dict]:
        """Generate expert-level consultation questions"""
        if is_tax:
            return [{
                'context': f"A high-net-worth client seeks sophisticated tax planning advice relating to '{metadata.title}'",
                'question': "Provide expert tax planning strategy with risk assessment and implementation guidance"
            }]
        else:
            return [{
                'context': f"A complex commercial dispute requires expert legal strategy based on '{metadata.title}'",
                'question': "Develop a comprehensive legal strategy with risk analysis and strategic recommendations"
            }]
    
    def _generate_adversarial_questions(self, metadata: Union[LegalMetadata, HMRCMetadata], content: str, is_tax: bool) -> List[Dict]:
        """Generate adversarial challenge questions"""
        if is_tax:
            return [{
                'context': f"HMRC is challenging a tax position based on '{metadata.title}'",
                'question': "Develop counter-arguments and defensive strategies",
                'opposing_argument': f"HMRC contends that the guidance in '{metadata.title}' clearly prohibits this tax position"
            }]
        else:
            return [{
                'context': f"Opposing counsel argues that '{metadata.title}' supports their client's position",
                'question': "Counter this argument with alternative legal analysis",
                'opposing_argument': f"The precedent in '{metadata.title}' clearly establishes liability in this case"
            }]
    
    def _generate_foundation_response(self, question: str, metadata: Union[LegalMetadata, HMRCMetadata], content: str) -> str:
        """Generate foundation-level response"""
        title = metadata.title
        citation = getattr(metadata, 'citation', title)
        
        return f"""Based on '{title}', the key points are:

**Legal Framework:**
{content[:300]}...

**Practical Application:**
[This analysis would be expanded with specific guidance from the full document]

**References:**
- {citation}

**Important Note:** This analysis is provided for educational purposes. For specific legal or tax advice, please consult with a qualified professional who can consider your individual circumstances.

**Confidence Level:** Foundation - based on primary source material."""
    
    def _generate_reasoning_response(self, scenario: Dict, metadata: Union[LegalMetadata, HMRCMetadata], content: str) -> str:
        """Generate multi-step reasoning response"""
        title = metadata.title
        citation = getattr(metadata, 'citation', title)
        
        return f"""**Multi-Step Legal Analysis:**

**Step 1: Issue Identification**
The key legal issues arising from this scenario relate to the principles established in '{title}'.

**Step 2: Legal Framework Analysis**
{content[:400]}...

**Step 3: Application to Facts**
Applying these principles to the specific circumstances requires consideration of [detailed analysis would follow].

**Step 4: Conclusion and Recommendations**
Based on this analysis, the recommended approach is [specific guidance would be provided].

**Legal Authorities:**
- {citation}

**Professional Disclaimer:** This analysis represents a legal interpretation based on available authorities. The application to specific circumstances may vary, and professional legal advice should be sought for definitive guidance.

**Confidence Level:** Intermediate - based on established legal principles with professional interpretation."""
    
    def _generate_expert_response(self, scenario: Dict, metadata: Union[LegalMetadata, HMRCMetadata], content: str) -> str:
        """Generate expert-level professional response"""
        title = metadata.title
        citation = getattr(metadata, 'citation', title)
        
        return f"""**Expert Professional Analysis:**

**Strategic Overview:**
This matter requires careful consideration of the established principles in '{title}' within the broader legal/regulatory context.

**Detailed Legal Analysis:**
{content[:500]}...

**Risk Assessment:**
- Primary risks: [detailed risk analysis]
- Mitigation strategies: [specific recommendations]
- Alternative approaches: [strategic options]

**Implementation Strategy:**
[Detailed step-by-step implementation guidance]

**Ongoing Considerations:**
[Future compliance and monitoring requirements]

**Legal Authorities:**
- {citation}
- [Additional supporting authorities would be cited]

**Professional Standards Disclaimer:** This advice is provided based on current legal understanding and established precedents. The implementation should be carefully monitored, and regular review is recommended given the evolving nature of legal interpretation.

**Confidence Level:** High - based on comprehensive analysis of established authorities and professional experience."""
    
    def _generate_adversarial_response(self, scenario: Dict, metadata: Union[LegalMetadata, HMRCMetadata], content: str) -> str:
        """Generate adversarial challenge response"""
        title = metadata.title
        citation = getattr(metadata, 'citation', title)
        opposing = scenario.get('opposing_argument', '')
        
        return f"""**Counter-Argument Analysis:**

**Opposing Position Assessment:**
{opposing}

**Challenge to Opposing Interpretation:**
The opposing interpretation fails to consider [detailed counter-analysis based on '{title}'].

**Alternative Legal Analysis:**
{content[:400]}...

**Distinguishing Factors:**
[Specific factors that differentiate this case]

**Supporting Precedents:**
[Additional authorities that support the alternative position]

**Strategic Considerations:**
[Tactical and procedural considerations for advancing this argument]

**Strength Assessment:**
This counter-argument has [strong/moderate/limited] prospects based on established precedent and legal principle.

**Legal Authorities:**
- {citation}
- [Supporting authorities]

**Professional Disclaimer:** This counter-argument analysis is provided for strategic planning purposes. The outcome of legal disputes depends on specific facts and judicial interpretation, and professional legal representation is essential.

**Confidence Level:** Strategic - based on established legal principles with adversarial context consideration."""
    
    def _get_legal_area(self, metadata: Union[LegalMetadata, HMRCMetadata]) -> str:
        """Extract legal area from metadata"""
        if isinstance(metadata, LegalMetadata):
            return metadata.legal_area
        elif isinstance(metadata, HMRCMetadata):
            return metadata.tax_domain.value
        return "general"
    
    def _get_jurisdiction(self, metadata: Union[LegalMetadata, HMRCMetadata]) -> str:
        """Extract jurisdiction from metadata"""
        if isinstance(metadata, LegalMetadata):
            return metadata.jurisdiction.value
        elif isinstance(metadata, HMRCMetadata):
            return "uk_wide"  # HMRC guidance applies UK-wide
        return "england_wales"
    
    def _assess_citation_accuracy(self, response: str) -> Dict:
        """Assess citation accuracy in the response"""
        citations_found = []
        
        # Check for different citation types
        for pattern_name, pattern in self.citation_patterns.items():
            matches = pattern.findall(response)
            citations_found.extend([(pattern_name, match) for match in matches])
        
        return {
            'total_citations': len(citations_found),
            'citation_types': [cite[0] for cite in citations_found],
            'accuracy_score': 0.8 if citations_found else 0.5  # Placeholder scoring
        }

class LegalTrainer:
    """
    Enhanced trainer with legal-specific evaluation metrics
    """
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.legal_evaluator = LegalEvaluator()
    
    def compute_metrics(self, eval_pred):
        """Compute legal-specific evaluation metrics"""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Legal-specific metrics
        metrics = {}
        
        for pred, label in zip(decoded_preds, decoded_labels):
            eval_result = self.legal_evaluator.evaluate_response(pred, label)
            
            # Aggregate metrics
            for metric_name, value in asdict(eval_result).items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(value)
        
        # Average all metrics
        final_metrics = {}
        for metric_name, values in metrics.items():
            final_metrics[metric_name] = np.mean(values) if values else 0.0
        
        return final_metrics

class LegalEvaluator:
    """
    Comprehensive legal response evaluation
    """
    
    def __init__(self):
        self.citation_patterns = {
            'neutral_citation': re.compile(r'\[(\d{4})\]\s+(UKSC|UKHL|UKPC|EWCA|EWHC|EWCOP|CSIH|CSOH|NICA|NIQB)\s+(\d+)'),
            'law_reports': re.compile(r'\[(\d{4})\]\s+(\d+)\s+(AC|WLR|All\s+ER|QB|Ch|Fam)'),
            'legislation': re.compile(r'\b((?:TCGA|ITA|ITTOIA|CTA|IHTA|FA|VATA|TMA)\s+\d{4})\b'),
            'sections': re.compile(r'\b(?:section|s\.|§)\s*(\d+)', re.IGNORECASE)
        }
        
        self.legal_structure_patterns = [
            r'(?:Introduction|Background|Legal Framework|Analysis|Conclusion)',
            r'(?:pursuant to|in accordance with|as provided by)',
            r'(?:s\.|section)\s*\d+',
            r'(?:para\.|paragraph)\s*\d+'
        ]
        
        self.professional_language_indicators = [
            'notwithstanding', 'pursuant', 'whereas', 'hereby', 'therein', 'aforementioned',
            'inter alia', 'prima facie', 'ultra vires', 'obiter dicta', 'ratio decidendi'
        ]
        
        self.disclaimer_patterns = [
            r'professional legal advice',
            r'qualified legal professional',
            r'specific circumstances',
            r'disclaimer'
        ]
    
    def evaluate_response(self, prediction: str, reference: str) -> LegalEvaluationMetrics:
        """Evaluate a legal response comprehensively"""
        
        # Citation accuracy
        citation_accuracy = self._evaluate_citation_accuracy(prediction, reference)
        
        # Legal structure
        legal_structure_score = self._evaluate_legal_structure(prediction)
        
        # Authority hierarchy
        authority_hierarchy_score = self._evaluate_authority_hierarchy(prediction)
        
        # Reasoning coherence
        reasoning_coherence = self._evaluate_reasoning_coherence(prediction)
        
        # Professional language
        professional_language_score = self._evaluate_professional_language(prediction)
        
        # OSCOLA compliance
        oscola_compliance = self._evaluate_oscola_compliance(prediction)
        
        # Disclaimer compliance
        disclaimer_compliance = self._evaluate_disclaimer_compliance(prediction)
        
        # Factual accuracy (basic check)
        factual_accuracy = self._evaluate_factual_accuracy(prediction, reference)
        
        # Overall score
        overall_score = np.mean([
            citation_accuracy, legal_structure_score, authority_hierarchy_score,
            reasoning_coherence, professional_language_score, oscola_compliance,
            factual_accuracy, disclaimer_compliance
        ])
        
        return LegalEvaluationMetrics(
            citation_accuracy=citation_accuracy,
            legal_structure_score=legal_structure_score,
            authority_hierarchy_score=authority_hierarchy_score,
            reasoning_coherence=reasoning_coherence,
            professional_language_score=professional_language_score,
            oscola_compliance=oscola_compliance,
            factual_accuracy=factual_accuracy,
            disclaimer_compliance=disclaimer_compliance,
            overall_score=overall_score
        )
    
    def _evaluate_citation_accuracy(self, prediction: str, reference: str) -> float:
        """Evaluate accuracy of legal citations"""
        pred_citations = set()
        ref_citations = set()
        
        for pattern in self.citation_patterns.values():
            pred_citations.update(pattern.findall(prediction))
            ref_citations.update(pattern.findall(reference))
        
        if not ref_citations:
            return 1.0 if not pred_citations else 0.8
        
        intersection = pred_citations.intersection(ref_citations)
        return len(intersection) / len(ref_citations) if ref_citations else 0.0
    
    def _evaluate_legal_structure(self, prediction: str) -> float:
        """Evaluate proper legal document structure"""
        score = 0.0
        total_patterns = len(self.legal_structure_patterns)
        
        for pattern in self.legal_structure_patterns:
            if re.search(pattern, prediction, re.IGNORECASE):
                score += 1.0
        
        return score / total_patterns
    
    def _evaluate_authority_hierarchy(self, prediction: str) -> float:
        """Evaluate proper authority hierarchy consideration"""
        # Check for references to different court levels
        hierarchy_indicators = [
            r'Supreme Court|UKSC',
            r'Court of Appeal|EWCA',
            r'High Court|EWHC',
            r'authority|precedent|binding'
        ]
        
        score = 0.0
        for indicator in hierarchy_indicators:
            if re.search(indicator, prediction, re.IGNORECASE):
                score += 0.25
        
        return min(score, 1.0)
    
    def _evaluate_reasoning_coherence(self, prediction: str) -> float:
        """Evaluate logical reasoning coherence"""
        # Check for logical connectors and structured reasoning
        reasoning_indicators = [
            r'therefore|thus|consequently|accordingly',
            r'however|nevertheless|notwithstanding',
            r'furthermore|moreover|additionally',
            r'step \d+|firstly|secondly|finally'
        ]
        
        score = 0.0
        for indicator in reasoning_indicators:
            if re.search(indicator, prediction, re.IGNORECASE):
                score += 0.25
        
        return min(score, 1.0)
    
    def _evaluate_professional_language(self, prediction: str) -> float:
        """Evaluate use of professional legal language"""
        score = 0.0
        text_lower = prediction.lower()
        
        for term in self.professional_language_indicators:
            if term in text_lower:
                score += 0.1
        
        return min(score, 1.0)
    
    def _evaluate_oscola_compliance(self, prediction: str) -> float:
        """Evaluate OSCOLA citation format compliance"""
        # Basic OSCOLA format checking
        oscola_patterns = [
            r'\[(\d{4})\]\s+[A-Z]+\s+\d+',  # Neutral citations
            r'[A-Z][a-z]+\s+v\s+[A-Z][a-z]+',  # Case names
            r'\(\d{4}\)\s+\d+\s+[A-Z]+',  # Law reports
        ]
        
        score = 0.0
        for pattern in oscola_patterns:
            if re.search(pattern, prediction):
                score += 0.33
        
        return min(score, 1.0)
    
    def _evaluate_disclaimer_compliance(self, prediction: str) -> float:
        """Evaluate presence of appropriate disclaimers"""
        score = 0.0
        
        for pattern in self.disclaimer_patterns:
            if re.search(pattern, prediction, re.IGNORECASE):
                score += 0.25
        
        return min(score, 1.0)
    
    def _evaluate_factual_accuracy(self, prediction: str, reference: str) -> float:
        """Basic factual accuracy evaluation"""
        # This is a simplified check - in practice would need more sophisticated NLP
        pred_words = set(prediction.lower().split())
        ref_words = set(reference.lower().split())
        
        if not ref_words:
            return 0.8  # Default score when no reference
        
        common_words = pred_words.intersection(ref_words)
        return len(common_words) / len(ref_words) if ref_words else 0.0

def create_legal_training_datasets(
    legal_documents: List[Tuple[str, Union[LegalMetadata, HMRCMetadata], str]],
    output_dir: Path
) -> Dict[str, Dataset]:
    """Create comprehensive legal training datasets"""
    
    creator = LegalTrainingDatasetCreator()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create different difficulty levels
    foundation_examples = creator.create_foundation_examples(legal_documents)
    reasoning_examples = creator.create_reasoning_examples(legal_documents)
    expert_examples = creator.create_expert_examples(legal_documents)
    adversarial_examples = creator.create_adversarial_examples(legal_documents)
    
    # Convert to HuggingFace datasets
    datasets = {}
    
    for level, examples in [
        ("foundation", foundation_examples),
        ("reasoning", reasoning_examples),
        ("expert", expert_examples),
        ("adversarial", adversarial_examples)
    ]:
        if examples:
            # Convert to format suitable for training
            training_data = []
            for example in examples:
                training_data.append({
                    'text': example.input_context + example.expected_output + "<|eot_id|>",
                    'input_context': example.input_context,
                    'expected_output': example.expected_output,
                    'legal_area': example.legal_area,
                    'difficulty_level': example.difficulty_level,
                    'jurisdiction': example.jurisdiction,
                    'metadata': json.dumps(example.metadata)
                })
            
            dataset = Dataset.from_list(training_data)
            datasets[level] = dataset
            
            # Save dataset
            dataset.save_to_disk(str(output_dir / f"legal_{level}_dataset"))
            dataset.to_json(str(output_dir / f"legal_{level}_dataset.json"))
    
    logger.info(f"Created {len(datasets)} legal training datasets with total {sum(len(d) for d in datasets.values())} examples")
    
    return datasets
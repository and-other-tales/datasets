#!/usr/bin/env python3
"""
Legal Compliance and Validation Framework
Production-ready legal AI compliance system with professional standards,
disclaimer validation, and ethical safeguards for legal AI systems.
"""

import re
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from enum import Enum

from .legal_metadata import LegalMetadata
from .hmrc_metadata import HMRCMetadata

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplianceLevel(Enum):
    """Compliance assessment levels"""
    COMPLIANT = "compliant"
    WARNING = "warning"
    NON_COMPLIANT = "non_compliant"
    REQUIRES_REVIEW = "requires_review"

class ConfidenceLevel(Enum):
    """AI confidence levels for legal responses"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNCERTAIN = "uncertain"

@dataclass
class ComplianceViolation:
    """Represents a compliance violation or warning"""
    violation_type: str
    severity: str  # "critical", "high", "medium", "low"
    description: str
    location: str  # Where in the text the violation occurs
    recommendation: str
    auto_fixable: bool = False

@dataclass
class LegalValidationResult:
    """Comprehensive legal validation result"""
    compliance_level: ComplianceLevel
    confidence_level: ConfidenceLevel
    overall_score: float
    
    # Compliance checks
    disclaimer_compliance: bool
    citation_accuracy: float
    professional_standards: bool
    ethical_compliance: bool
    jurisdiction_compliance: bool
    
    # Detailed analysis
    violations: List[ComplianceViolation]
    warnings: List[str]
    required_disclaimers: List[str]
    citation_issues: List[str]
    
    # Recommendations
    improvement_suggestions: List[str]
    auto_fixes: List[str]
    
    # Metadata
    validation_timestamp: datetime
    validator_version: str

class LegalDisclaimerValidator:
    """
    Validates legal disclaimers and professional standards compliance
    """
    
    def __init__(self):
        # Standard legal disclaimers
        self.required_disclaimers = {
            'general_legal': """This analysis is generated by an AI system and should not be considered as formal legal advice. 
It is intended for informational purposes only. Any legal decisions should be made in consultation with a qualified 
legal professional who can consider the full circumstances of your case and provide advice tailored to your specific 
situation. The analysis may not reflect the most current legal developments.""",
            
            'tax_advice': """This tax guidance is provided by an AI system for informational purposes only and should not be 
considered as professional tax advice. Tax law is complex and constantly evolving. You should consult with a qualified 
tax advisor or HMRC directly for advice specific to your circumstances. The guidance may not reflect the most recent 
tax changes or HMRC practice.""",
            
            'litigation_advice': """This litigation analysis is generated by an AI system for informational purposes only. 
Legal proceedings involve complex procedural and substantive considerations that require professional judgment. You should 
consult with a qualified solicitor or barrister for advice specific to your case. The analysis should not be relied upon 
for any legal proceedings without professional legal review."""
        }
        
        # Disclaimer detection patterns
        self.disclaimer_patterns = [
            r'not\s+(?:formal\s+)?legal\s+advice',
            r'informational\s+purposes?\s+only',
            r'consult\s+(?:with\s+)?(?:a\s+)?qualified\s+(?:legal\s+professional|solicitor|barrister)',
            r'professional\s+legal\s+advice',
            r'specific\s+(?:to\s+your\s+)?circumstances',
            r'may\s+not\s+reflect\s+(?:the\s+)?(?:most\s+)?(?:current|recent)',
            r'ai\s+system',
            r'disclaimer'
        ]
        
        # Professional standards requirements
        self.professional_standards = {
            'accuracy': 'Must provide accurate legal information based on current law',
            'clarity': 'Must use clear, precise legal language appropriate to the audience',
            'competence': 'Must demonstrate competent understanding of legal principles',
            'integrity': 'Must not mislead or provide false legal information',
            'confidentiality': 'Must respect client confidentiality principles',
            'independence': 'Must provide objective legal analysis',
            'accountability': 'Must acknowledge limitations and uncertainties'
        }
    
    def validate_disclaimer_presence(self, text: str, response_type: str = "general_legal") -> Tuple[bool, List[str]]:
        """Validate presence and adequacy of legal disclaimers"""
        issues = []
        has_disclaimer = False
        
        text_lower = text.lower()
        
        # Check for disclaimer patterns
        disclaimer_matches = 0
        for pattern in self.disclaimer_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                disclaimer_matches += 1
        
        # Require at least 3 disclaimer elements
        if disclaimer_matches >= 3:
            has_disclaimer = True
        else:
            issues.append(f"Insufficient disclaimer content (found {disclaimer_matches} elements, need 3+)")
        
        # Check for specific required elements based on response type
        if response_type == "tax_advice":
            if 'hmrc' not in text_lower and 'tax advisor' not in text_lower:
                issues.append("Tax advice disclaimer should mention HMRC or qualified tax advisor")
        
        elif response_type == "litigation_advice":
            if 'solicitor' not in text_lower and 'barrister' not in text_lower:
                issues.append("Litigation advice should mention qualified solicitor or barrister")
        
        # Check for overconfident language that contradicts disclaimers
        overconfident_patterns = [
            r'definitely\s+will',
            r'guaranteed\s+(?:to\s+)?(?:win|succeed)',
            r'certain\s+(?:to\s+)?(?:win|succeed)',
            r'no\s+chance\s+of\s+(?:losing|failing)',
            r'absolutely\s+certain'
        ]
        
        for pattern in overconfident_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                issues.append(f"Overconfident language detected: '{pattern}' contradicts disclaimer")
        
        return has_disclaimer, issues
    
    def get_required_disclaimer(self, response_type: str, legal_area: str = None) -> str:
        """Get the appropriate disclaimer for the response type"""
        disclaimer_key = response_type
        
        if response_type not in self.required_disclaimers:
            disclaimer_key = "general_legal"
        
        base_disclaimer = self.required_disclaimers[disclaimer_key]
        
        # Add area-specific disclaimers
        if legal_area:
            area_specific = self._get_area_specific_disclaimer(legal_area)
            if area_specific:
                base_disclaimer += f"\n\n{area_specific}"
        
        return base_disclaimer
    
    def _get_area_specific_disclaimer(self, legal_area: str) -> Optional[str]:
        """Get area-specific disclaimer additions"""
        area_disclaimers = {
            'criminal': "Criminal law matters require immediate professional legal representation. Do not rely on this analysis for any criminal proceedings.",
            'immigration': "Immigration law is highly complex and changes frequently. Professional immigration advice is essential for any immigration matters.",
            'family': "Family law matters involve sensitive personal circumstances requiring professional legal guidance and court representation.",
            'employment': "Employment law disputes should be addressed with qualified employment lawyers and may involve time-sensitive procedural requirements."
        }
        
        return area_disclaimers.get(legal_area.lower())

class LegalCitationValidator:
    """
    Validates legal citations for accuracy and proper format
    """
    
    def __init__(self):
        # Citation format patterns
        self.citation_patterns = {
            'neutral_citation': {
                'pattern': re.compile(r'\[(\d{4})\]\s+(UKSC|UKHL|UKPC|EWCA|EWHC|EWCOP|CSIH|CSOH|NICA|NIQB)\s+(\d+)', re.IGNORECASE),
                'description': 'UK Neutral Citation',
                'format': '[YYYY] COURT N'
            },
            'law_reports': {
                'pattern': re.compile(r'\[(\d{4})\]\s+(\d+)\s+(AC|WLR|All\s+ER|QB|Ch|Fam|P)\s+(\d+)', re.IGNORECASE),
                'description': 'Law Reports Citation',
                'format': '[YYYY] Volume Report Page'
            },
            'legislation': {
                'pattern': re.compile(r'\b((?:TCGA|ITA|ITTOIA|CTA|IHTA|FA|VATA|TMA)\s+\d{4})\b', re.IGNORECASE),
                'description': 'UK Legislation',
                'format': 'Act YYYY'
            },
            'section_references': {
                'pattern': re.compile(r'\b(?:section|s\.|§)\s*(\d+(?:\([^)]+\))*)', re.IGNORECASE),
                'description': 'Section Reference',
                'format': 's.N or section N'
            },
            'hmrc_manual': {
                'pattern': re.compile(r'\b([A-Z]{2,4}\d{5}[A-Z]*)\b'),
                'description': 'HMRC Manual Reference',
                'format': 'MANUAL12345'
            }
        }
        
        # Court hierarchy for authority validation
        self.court_hierarchy = {
            'UKSC': 1, 'UKHL': 1, 'UKPC': 1,  # Supreme Court level
            'EWCA': 2, 'CSIH': 2, 'NICA': 2,  # Court of Appeal level
            'EWHC': 3, 'CSOH': 3, 'NIQB': 3,  # High Court level
            'EWCOP': 3,  # Court of Protection
        }
    
    def validate_citations(self, text: str) -> Tuple[float, List[str]]:
        """Validate all citations in text"""
        issues = []
        total_citations = 0
        valid_citations = 0
        
        for citation_type, pattern_info in self.citation_patterns.items():
            matches = pattern_info['pattern'].findall(text)
            total_citations += len(matches)
            
            for match in matches:
                is_valid, citation_issues = self._validate_individual_citation(match, citation_type)
                if is_valid:
                    valid_citations += 1
                else:
                    issues.extend(citation_issues)
        
        # Calculate accuracy score
        accuracy = valid_citations / total_citations if total_citations > 0 else 1.0
        
        # Additional validation checks
        issues.extend(self._validate_citation_context(text))
        issues.extend(self._validate_authority_hierarchy(text))
        
        return accuracy, issues
    
    def _validate_individual_citation(self, match: Union[str, Tuple], citation_type: str) -> Tuple[bool, List[str]]:
        """Validate an individual citation"""
        issues = []
        
        if citation_type == 'neutral_citation':
            year, court, number = match
            
            # Validate year range
            year_int = int(year)
            if year_int < 1900 or year_int > datetime.now().year:
                issues.append(f"Invalid year in citation: {year}")
            
            # Validate court abbreviation
            if court.upper() not in self.court_hierarchy:
                issues.append(f"Unknown court abbreviation: {court}")
            
        elif citation_type == 'legislation':
            # Basic validation for UK legislation format
            if not re.match(r'^[A-Z]+\s+\d{4}$', match):
                issues.append(f"Invalid legislation format: {match}")
        
        return len(issues) == 0, issues
    
    def _validate_citation_context(self, text: str) -> List[str]:
        """Validate citations are used in proper context"""
        issues = []
        
        # Check for dangling citations (citations without context)
        citation_pattern = re.compile(r'\[(\d{4})\]\s+[A-Z]+\s+\d+')
        
        for match in citation_pattern.finditer(text):
            start = max(0, match.start() - 50)
            end = min(len(text), match.end() + 50)
            context = text[start:end]
            
            # Check if citation has proper context
            context_indicators = ['held', 'decided', 'established', 'case', 'judgment', 'ruling']
            if not any(indicator in context.lower() for indicator in context_indicators):
                issues.append(f"Citation may lack proper context: {match.group()}")
        
        return issues
    
    def _validate_authority_hierarchy(self, text: str) -> List[str]:
        """Validate proper use of authority hierarchy"""
        issues = []
        
        # Find all court citations
        neutral_pattern = re.compile(r'\[(\d{4})\]\s+(UKSC|UKHL|UKPC|EWCA|EWHC|EWCOP|CSIH|CSOH|NICA|NIQB)\s+(\d+)', re.IGNORECASE)
        citations = neutral_pattern.findall(text)
        
        # Check if higher authority is cited before lower authority for the same point
        court_levels = []
        for year, court, number in citations:
            level = self.court_hierarchy.get(court.upper(), 999)
            court_levels.append((level, f"{court} {year}"))
        
        # Simple check: if multiple courts cited, ensure higher authority comes first or is emphasized
        if len(court_levels) > 1:
            levels = [level for level, _ in court_levels]
            if not all(levels[i] <= levels[i+1] for i in range(len(levels)-1)):
                issues.append("Consider citing higher authority cases before lower authority cases")
        
        return issues

class LegalContentValidator:
    """
    Validates legal content for professional standards and accuracy
    """
    
    def __init__(self):
        # Professional language indicators
        self.professional_indicators = [
            'pursuant to', 'notwithstanding', 'whereas', 'hereby', 'therein',
            'aforementioned', 'inter alia', 'prima facie', 'ultra vires'
        ]
        
        # Problematic language patterns
        self.problematic_patterns = [
            (r'always\s+win', "Avoid absolute guarantees about legal outcomes"),
            (r'never\s+(?:fail|lose)', "Avoid absolute certainty about legal outcomes"),
            (r'guaranteed\s+success', "Legal outcomes cannot be guaranteed"),
            (r'impossible\s+to\s+lose', "Avoid overconfident language"),
            (r'easy\s+case', "Legal matters should not be characterized as 'easy'"),
            (r'slam\s+dunk', "Avoid colloquial language in legal analysis"),
            (r'obvious(?:ly)?\s+(?:win|guilty|liable)', "Avoid overconfident conclusions")
        ]
        
        # Required legal reasoning elements
        self.reasoning_elements = [
            'legal framework', 'analysis', 'application', 'conclusion',
            'relevant law', 'applicable', 'principle', 'precedent'
        ]
    
    def validate_professional_standards(self, text: str) -> Tuple[bool, List[str]]:
        """Validate text meets professional legal writing standards"""
        issues = []
        
        # Check for problematic language
        for pattern, issue_desc in self.problematic_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                issues.append(issue_desc)
        
        # Check for reasoning structure
        reasoning_score = 0
        text_lower = text.lower()
        for element in self.reasoning_elements:
            if element in text_lower:
                reasoning_score += 1
        
        if reasoning_score < 3:
            issues.append("Response should include more structured legal reasoning elements")
        
        # Check for appropriate legal language
        professional_score = 0
        for indicator in self.professional_indicators:
            if indicator in text_lower:
                professional_score += 1
        
        # Don't require professional language for all responses, but note if completely absent
        if len(text) > 500 and professional_score == 0:
            issues.append("Consider using more formal legal language for comprehensive analysis")
        
        return len(issues) == 0, issues
    
    def validate_factual_consistency(self, text: str, source_metadata: Optional[Union[LegalMetadata, HMRCMetadata]] = None) -> List[str]:
        """Validate factual consistency with source material"""
        issues = []
        
        if source_metadata:
            # Check jurisdiction consistency
            if hasattr(source_metadata, 'jurisdiction'):
                jurisdiction_mentioned = self._extract_mentioned_jurisdictions(text)
                source_jurisdiction = source_metadata.jurisdiction.value if hasattr(source_metadata.jurisdiction, 'value') else str(source_metadata.jurisdiction)
                
                if jurisdiction_mentioned and source_jurisdiction not in jurisdiction_mentioned:
                    issues.append(f"Jurisdiction inconsistency: source is {source_jurisdiction} but text mentions {jurisdiction_mentioned}")
            
            # Check legal area consistency
            legal_area = getattr(source_metadata, 'legal_area', None) or getattr(source_metadata, 'tax_domain', None)
            if legal_area:
                area_value = legal_area.value if hasattr(legal_area, 'value') else str(legal_area)
                if not self._check_area_consistency(text, area_value):
                    issues.append(f"Content may not be consistent with {area_value} legal area")
        
        return issues
    
    def _extract_mentioned_jurisdictions(self, text: str) -> Set[str]:
        """Extract mentioned jurisdictions from text"""
        jurisdictions = set()
        text_lower = text.lower()
        
        jurisdiction_indicators = {
            'england_wales': ['england', 'wales', 'english', 'welsh'],
            'scotland': ['scotland', 'scottish'],
            'northern_ireland': ['northern ireland', 'ni'],
            'uk_wide': ['uk', 'united kingdom', 'britain', 'british']
        }
        
        for jurisdiction, indicators in jurisdiction_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                jurisdictions.add(jurisdiction)
        
        return jurisdictions
    
    def _check_area_consistency(self, text: str, legal_area: str) -> bool:
        """Check if text content is consistent with declared legal area"""
        text_lower = text.lower()
        area_lower = legal_area.lower()
        
        # Basic keyword matching for area consistency
        area_keywords = {
            'criminal': ['crime', 'criminal', 'prosecution', 'defendant', 'guilty'],
            'tax': ['tax', 'hmrc', 'revenue', 'vat', 'duty'],
            'employment': ['employment', 'worker', 'employer', 'dismissal'],
            'property': ['property', 'land', 'conveyancing', 'lease'],
            'contract': ['contract', 'agreement', 'breach', 'terms']
        }
        
        if area_lower in area_keywords:
            keywords = area_keywords[area_lower]
            return any(keyword in text_lower for keyword in keywords)
        
        return True  # Default to consistent if no specific check available

class LegalComplianceFramework:
    """
    Comprehensive legal compliance framework integrating all validation components
    """
    
    def __init__(self):
        self.disclaimer_validator = LegalDisclaimerValidator()
        self.citation_validator = LegalCitationValidator()
        self.content_validator = LegalContentValidator()
        
        # Confidence thresholds
        self.confidence_thresholds = {
            "high": 0.8,
            "medium": 0.6,
            "low": 0.4
        }
        
        # Compliance scoring weights
        self.scoring_weights = {
            'disclaimer_compliance': 0.25,
            'citation_accuracy': 0.20,
            'professional_standards': 0.20,
            'ethical_compliance': 0.15,
            'factual_consistency': 0.20
        }
    
    def validate_legal_response(
        self, 
        response: str, 
        response_type: str = "general_legal",
        legal_area: str = None,
        source_metadata: Union[LegalMetadata, HMRCMetadata] = None,
        context: Dict = None
    ) -> LegalValidationResult:
        """Comprehensive legal response validation"""
        
        violations = []
        warnings = []
        required_disclaimers = []
        citation_issues = []
        improvement_suggestions = []
        auto_fixes = []
        
        # 1. Disclaimer compliance validation
        has_disclaimer, disclaimer_issues = self.disclaimer_validator.validate_disclaimer_presence(response, response_type)
        if not has_disclaimer:
            violations.append(ComplianceViolation(
                violation_type="missing_disclaimer",
                severity="critical",
                description="Required legal disclaimer is missing or insufficient",
                location="document_end",
                recommendation="Add comprehensive legal disclaimer",
                auto_fixable=True
            ))
            required_disclaimers.append(self.disclaimer_validator.get_required_disclaimer(response_type, legal_area))
        
        # 2. Citation accuracy validation
        citation_accuracy, citation_validation_issues = self.citation_validator.validate_citations(response)
        citation_issues.extend(citation_validation_issues)
        
        if citation_accuracy < 0.8:
            violations.append(ComplianceViolation(
                violation_type="citation_accuracy",
                severity="high",
                description=f"Citation accuracy below standard ({citation_accuracy:.2f})",
                location="throughout_document",
                recommendation="Review and correct legal citations",
                auto_fixable=False
            ))
        
        # 3. Professional standards validation
        meets_standards, professional_issues = self.content_validator.validate_professional_standards(response)
        if not meets_standards:
            for issue in professional_issues:
                violations.append(ComplianceViolation(
                    violation_type="professional_standards",
                    severity="medium",
                    description=issue,
                    location="content_analysis",
                    recommendation="Revise language to meet professional standards",
                    auto_fixable=False
                ))
        
        # 4. Factual consistency validation
        factual_issues = self.content_validator.validate_factual_consistency(response, source_metadata)
        for issue in factual_issues:
            violations.append(ComplianceViolation(
                violation_type="factual_consistency",
                severity="high",
                description=issue,
                location="content_analysis",
                recommendation="Review and correct factual inconsistencies",
                auto_fixable=False
            ))
        
        # 5. Ethical compliance check
        ethical_compliant = self._check_ethical_compliance(response)
        
        # Calculate overall compliance score
        scores = {
            'disclaimer_compliance': 1.0 if has_disclaimer else 0.0,
            'citation_accuracy': citation_accuracy,
            'professional_standards': 1.0 if meets_standards else 0.5,
            'ethical_compliance': 1.0 if ethical_compliant else 0.0,
            'factual_consistency': 1.0 if not factual_issues else 0.5
        }
        
        overall_score = sum(
            scores[metric] * self.scoring_weights[metric] 
            for metric in scores
        )
        
        # Determine compliance level
        compliance_level = self._determine_compliance_level(violations, overall_score)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(overall_score, len(violations))
        
        # Generate improvement suggestions
        improvement_suggestions = self._generate_improvement_suggestions(violations, scores)
        
        # Generate auto-fixes
        auto_fixes = self._generate_auto_fixes(violations, required_disclaimers)
        
        return LegalValidationResult(
            compliance_level=compliance_level,
            confidence_level=confidence_level,
            overall_score=overall_score,
            disclaimer_compliance=has_disclaimer,
            citation_accuracy=citation_accuracy,
            professional_standards=meets_standards,
            ethical_compliance=ethical_compliant,
            jurisdiction_compliance=len(factual_issues) == 0,
            violations=violations,
            warnings=warnings,
            required_disclaimers=required_disclaimers,
            citation_issues=citation_issues,
            improvement_suggestions=improvement_suggestions,
            auto_fixes=auto_fixes,
            validation_timestamp=datetime.now(),
            validator_version="1.0.0"
        )
    
    def _check_ethical_compliance(self, response: str) -> bool:
        """Check for ethical compliance issues"""
        # Check for problematic content
        ethical_violations = [
            r'encourage\s+(?:illegal|unlawful)',
            r'evade\s+(?:tax|legal)',
            r'false\s+(?:statement|claim)',
            r'mislead\s+(?:court|authority)',
            r'conceal\s+(?:evidence|information)'
        ]
        
        response_lower = response.lower()
        for violation in ethical_violations:
            if re.search(violation, response_lower):
                return False
        
        return True
    
    def _determine_compliance_level(self, violations: List[ComplianceViolation], overall_score: float) -> ComplianceLevel:
        """Determine overall compliance level"""
        critical_violations = [v for v in violations if v.severity == "critical"]
        high_violations = [v for v in violations if v.severity == "high"]
        
        if critical_violations:
            return ComplianceLevel.NON_COMPLIANT
        elif high_violations or overall_score < 0.6:
            return ComplianceLevel.WARNING
        elif overall_score < 0.8:
            return ComplianceLevel.REQUIRES_REVIEW
        else:
            return ComplianceLevel.COMPLIANT
    
    def _determine_confidence_level(self, overall_score: float, violation_count: int) -> ConfidenceLevel:
        """Determine confidence level in the validation"""
        if overall_score >= self.confidence_thresholds["high"] and violation_count == 0:
            return ConfidenceLevel.HIGH
        elif overall_score >= self.confidence_thresholds["medium"] and violation_count <= 2:
            return ConfidenceLevel.MEDIUM
        elif overall_score >= self.confidence_thresholds["low"]:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNCERTAIN
    
    def _generate_improvement_suggestions(self, violations: List[ComplianceViolation], scores: Dict[str, float]) -> List[str]:
        """Generate specific improvement suggestions"""
        suggestions = []
        
        # Priority suggestions based on violations
        for violation in violations:
            suggestions.append(violation.recommendation)
        
        # Score-based suggestions
        if scores['citation_accuracy'] < 0.8:
            suggestions.append("Review legal citations for proper format and accuracy")
        
        if scores['professional_standards'] < 0.8:
            suggestions.append("Enhance professional legal language and structure")
        
        if scores['disclaimer_compliance'] < 1.0:
            suggestions.append("Add comprehensive legal disclaimers")
        
        return list(set(suggestions))  # Remove duplicates
    
    def _generate_auto_fixes(self, violations: List[ComplianceViolation], required_disclaimers: List[str]) -> List[str]:
        """Generate automatic fixes for fixable violations"""
        auto_fixes = []
        
        for violation in violations:
            if violation.auto_fixable:
                if violation.violation_type == "missing_disclaimer":
                    auto_fixes.append(f"Add disclaimer: {required_disclaimers[0] if required_disclaimers else 'Standard legal disclaimer'}")
        
        return auto_fixes
    
    def add_compliance_wrapper(self, response: str, validation: LegalValidationResult) -> str:
        """Add compliance wrapper with disclaimers and confidence indicators"""
        confidence_indicator = f"[CONFIDENCE: {validation.confidence_level.value.upper()}]"
        compliance_indicator = f"[COMPLIANCE: {validation.compliance_level.value.upper()}]"
        
        wrapped_response = f"""{confidence_indicator} {compliance_indicator}

{response}"""
        
        # Add required disclaimers
        if validation.required_disclaimers:
            wrapped_response += f"\n\n{'-'*50}\nIMPORTANT LEGAL DISCLAIMER:\n{validation.required_disclaimers[0]}"
        
        # Add compliance notes if needed
        if validation.compliance_level != ComplianceLevel.COMPLIANT:
            compliance_notes = []
            if validation.violations:
                compliance_notes.append(f"Compliance Issues: {len(validation.violations)} violations detected")
            if validation.warnings:
                compliance_notes.append(f"Warnings: {'; '.join(validation.warnings[:3])}")
            
            if compliance_notes:
                wrapped_response += f"\n\nCOMPLIANCE NOTES: {'; '.join(compliance_notes)}"
        
        # Add confidence assessment
        wrapped_response += f"\n\nCONFIDENCE ASSESSMENT: This analysis is provided with {validation.confidence_level.value} confidence (overall score: {validation.overall_score:.2f})."
        
        return wrapped_response

def validate_legal_dataset(
    dataset_path: Path,
    output_path: Path,
    compliance_framework: LegalComplianceFramework
) -> Dict:
    """Validate an entire legal training dataset for compliance"""
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Load dataset
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    validation_results = []
    summary_stats = {
        'total_examples': len(dataset),
        'compliant': 0,
        'warnings': 0,
        'non_compliant': 0,
        'requires_review': 0,
        'high_confidence': 0,
        'medium_confidence': 0,
        'low_confidence': 0,
        'uncertain': 0
    }
    
    # Validate each example
    for i, example in enumerate(dataset):
        response = example.get('expected_output', '')
        legal_area = example.get('legal_area', 'general')
        
        validation = compliance_framework.validate_legal_response(
            response=response,
            legal_area=legal_area
        )
        
        # Update statistics
        summary_stats[validation.compliance_level.value] += 1
        summary_stats[f"{validation.confidence_level.value}_confidence"] += 1
        
        validation_results.append({
            'example_index': i,
            'compliance_level': validation.compliance_level.value,
            'confidence_level': validation.confidence_level.value,
            'overall_score': validation.overall_score,
            'violations': [asdict(v) for v in validation.violations],
            'improvement_suggestions': validation.improvement_suggestions
        })
    
    # Save validation results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        'dataset_path': str(dataset_path),
        'validation_timestamp': datetime.now().isoformat(),
        'summary_statistics': summary_stats,
        'detailed_results': validation_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Dataset validation complete: {summary_stats['compliant']}/{summary_stats['total_examples']} examples compliant")
    
    return summary_stats
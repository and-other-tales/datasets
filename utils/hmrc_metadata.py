#!/usr/bin/env python3
"""
HMRC-Specific Metadata Framework
Production-ready HMRC tax guidance metadata handling with tax-specific 
authority hierarchies, cross-referencing, and legislative citation tracking.
"""

import re
import json
import logging
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Set, Union, Tuple
from pathlib import Path
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HMRCDocumentType(Enum):
    """Types of HMRC documentation"""
    INTERNAL_MANUAL = "internal_manual"          # CG Manual, BIM Manual, etc.
    GUIDANCE = "guidance"                        # Public guidance documents
    REVENUE_AND_CUSTOMS_BRIEF = "revenue_brief"  # Official briefs
    TECHNICAL_NOTE = "technical_note"           # Technical explanations
    POLICY_PAPER = "policy_paper"               # Policy documents
    CONSULTATION = "consultation"               # Public consultations
    FORM = "form"                               # Tax forms and returns
    LEGISLATION_COMMENTARY = "legislation_commentary"  # HMRC view on legislation
    PRACTICE_NOTE = "practice_note"             # Practical guidance
    BULLETIN = "bulletin"                       # Regular updates

class TaxAuthority(Enum):
    """Tax authority hierarchy (1=highest precedence)"""
    PRIMARY_LEGISLATION = 1      # Acts of Parliament (TCGA 1992, ITA 2007, etc.)
    SECONDARY_LEGISLATION = 2    # Statutory Instruments, Regulations
    CASE_LAW = 3                # Tax tribunal and court decisions
    HMRC_INTERNAL_MANUAL = 4    # Official HMRC interpretation
    HMRC_GUIDANCE = 5           # Public guidance documents
    HMRC_BRIEF = 6              # Revenue and Customs Briefs
    HMRC_TECHNICAL_NOTE = 7     # Technical explanations
    HMRC_PRACTICE_NOTE = 8      # Practice guidance
    COMMENTARY = 9              # External commentary

class TaxDomain(Enum):
    """Tax domains and subject areas"""
    INCOME_TAX = "income_tax"
    CORPORATION_TAX = "corporation_tax"
    CAPITAL_GAINS_TAX = "capital_gains_tax"
    VALUE_ADDED_TAX = "vat"
    INHERITANCE_TAX = "inheritance_tax"
    STAMP_DUTY = "stamp_duty"
    NATIONAL_INSURANCE = "national_insurance"
    PAYE = "paye"
    CUSTOMS_DUTIES = "customs_duties"
    EXCISE_DUTIES = "excise_duties"
    ANTI_AVOIDANCE = "anti_avoidance"
    COMPLIANCE = "compliance"
    PENALTIES = "penalties"
    APPEALS = "appeals"
    
    # New categories identified from API analysis
    MAKING_TAX_DIGITAL = "making_tax_digital"
    EMPLOYEE_BENEFITS = "employee_benefits"
    FUEL_DUTIES = "fuel_duties"
    ALCOHOL_TOBACCO_DUTIES = "alcohol_tobacco_duties"
    CUSTOMS_IMPORT_DUTIES = "customs_import_duties"
    MONEY_LAUNDERING_SUPERVISION = "money_laundering_supervision"
    TAX_AVOIDANCE_SCHEMES = "tax_avoidance_schemes"
    RESEARCH_DEVELOPMENT_RELIEFS = "research_development_reliefs"
    SERVICE_AVAILABILITY = "service_availability"
    DIGITAL_SERVICES = "digital_services"
    TRADE_PROCEDURES = "trade_procedures"
    ECONOMIC_CRIME = "economic_crime"
    TAX_CREDITS = "tax_credits"
    BUSINESS_RATES = "business_rates"
    ENVIRONMENTAL_TAXES = "environmental_taxes"
    
    GENERAL = "general"

@dataclass
class TaxLegislationReference:
    """Structured tax legislation reference"""
    act_name: str                    # e.g., "TCGA 1992", "ITA 2007"
    section: Optional[str] = None    # e.g., "s.1", "s.123(4)"
    full_name: Optional[str] = None  # e.g., "Taxation of Chargeable Gains Act 1992"
    reference_type: str = "section"  # section, subsection, paragraph, schedule
    context: str = ""                # Surrounding text context

@dataclass
class HMRCReference:
    """Cross-reference to other HMRC materials"""
    manual_code: str                 # e.g., "CG12345", "BIM10000"
    manual_name: str                 # e.g., "Capital Gains Manual"
    section_title: Optional[str] = None
    url: Optional[str] = None

@dataclass
class HMRCMetadata:
    """
    Comprehensive HMRC document metadata for tax guidance
    """
    # Core identification
    document_type: HMRCDocumentType
    title: str
    manual_code: Optional[str] = None        # e.g., "CG12345", "BIM20000"
    manual_name: Optional[str] = None        # e.g., "Capital Gains Manual"
    content_id: str = field(default_factory=lambda: "")
    
    # Tax authority and hierarchy
    authority_level: TaxAuthority = TaxAuthority.COMMENTARY
    tax_domain: TaxDomain = TaxDomain.GENERAL
    subject_areas: List[str] = field(default_factory=list)
    
    # Publication and versioning
    published_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    version: Optional[str] = None
    supersedes: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    
    # Legislative and case references
    legislation_references: List[TaxLegislationReference] = field(default_factory=list)
    case_references: List[str] = field(default_factory=list)
    hmrc_cross_references: List[HMRCReference] = field(default_factory=list)
    
    # Content characteristics
    document_length: int = 0
    section_count: int = 0
    has_examples: bool = False
    has_calculations: bool = False
    language: str = "en-GB"
    
    # Technical metadata
    source_url: Optional[str] = None
    content_hash: Optional[str] = None
    extraction_method: str = "hmrc_scraper"
    
    # Tax-specific classifications
    affects_individuals: bool = False
    affects_companies: bool = False
    affects_trusts: bool = False
    affects_partnerships: bool = False
    
    # Keywords and topics
    keywords: List[str] = field(default_factory=list)
    tax_concepts: List[str] = field(default_factory=list)
    
    # Quality metrics
    completeness_score: float = 0.0
    reference_accuracy_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with enum handling"""
        result = asdict(self)
        # Convert enums to values
        result['document_type'] = self.document_type.value
        result['authority_level'] = self.authority_level.value
        result['tax_domain'] = self.tax_domain.value
        
        # Handle datetime serialization
        for field_name in ['published_date', 'last_updated']:
            if result[field_name]:
                result[field_name] = result[field_name].isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'HMRCMetadata':
        """Create from dictionary with enum handling"""
        # Convert string enums back to enum objects
        if 'document_type' in data and isinstance(data['document_type'], str):
            data['document_type'] = HMRCDocumentType(data['document_type'])
        if 'authority_level' in data and isinstance(data['authority_level'], str):
            data['authority_level'] = TaxAuthority(data['authority_level'])
        if 'tax_domain' in data and isinstance(data['tax_domain'], str):
            data['tax_domain'] = TaxDomain(data['tax_domain'])
        
        # Handle datetime deserialization
        for field_name in ['published_date', 'last_updated']:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)

class HMRCReferenceExtractor:
    """
    Production-ready HMRC reference extraction with tax-specific patterns
    """
    
    def __init__(self):
        # HMRC manual reference patterns
        self.hmrc_patterns = {
            'manual_sections': re.compile(
                r'\b([A-Z]{2,4})(\d{5}[A-Z]*)\b',  # e.g., CG12345, BIM10000A
                re.IGNORECASE
            ),
            'cross_references': re.compile(
                r'(?:see|refer to|guidance at)\s+([A-Z]{2,4}\d{5}[A-Z]*)',
                re.IGNORECASE
            )
        }
        
        # Tax legislation patterns
        self.legislation_patterns = {
            'acts': re.compile(
                r'\b((?:TCGA|ITA|ITTOIA|CTA|IHTA|FA|VATA|TMA)\s+\d{4})\b',
                re.IGNORECASE
            ),
            'sections': re.compile(
                r'\b(?:section|s\.|sec\.|§)\s*(\d+(?:\([A-Za-z0-9]+\))*(?:[A-Z]*)?)',
                re.IGNORECASE
            ),
            'schedules': re.compile(
                r'\b(?:schedule|sch\.|para\.)\s*(\d+(?:\([A-Za-z0-9]+\))*)',
                re.IGNORECASE
            ),
            'statutory_instruments': re.compile(
                r'\b(?:SI|S\.I\.)\s*(\d{4}/\d+)',
                re.IGNORECASE
            )
        }
        
        # Manual code mappings
        self.manual_mappings = {
            'CG': 'Capital Gains Manual',
            'BIM': 'Business Income Manual',
            'EIM': 'Employment Income Manual',
            'SAIM': 'Savings and Investment Manual',
            'PCTM': 'Pension Contributions Tax Manual',
            'RPSM': 'Registered Pension Schemes Manual',
            'IHTM': 'Inheritance Tax Manual',
            'VATFIN': 'VAT Finance Manual',
            'ARTG': 'Anti-Avoidance Reporting of Tax Avoidance Schemes',
            'INTM': 'International Manual',
            'CTM': 'Corporation Tax Manual',
            'PIM': 'Property Income Manual',
            'TSEM': 'Tax Specialist Investigations Manual',
            'CH': 'Compliance Handbook'
        }
        
        # Tax legislation full names
        self.legislation_names = {
            'TCGA 1992': 'Taxation of Chargeable Gains Act 1992',
            'ITA 2007': 'Income Tax Act 2007',
            'ITTOIA 2005': 'Income Tax (Trading and Other Income) Act 2005',
            'CTA 2009': 'Corporation Tax Act 2009',
            'CTA 2010': 'Corporation Tax Act 2010',
            'IHTA 1984': 'Inheritance Tax Act 1984',
            'VATA 1994': 'Value Added Tax Act 1994',
            'TMA 1970': 'Taxes Management Act 1970'
        }
    
    def extract_hmrc_references(self, text: str) -> List[HMRCReference]:
        """Extract HMRC manual references from text"""
        references = []
        
        # Find manual section references
        for match in self.hmrc_patterns['manual_sections'].finditer(text):
            prefix = match.group(1).upper()
            full_code = match.group(0).upper()
            
            manual_name = self.manual_mappings.get(prefix, f"{prefix} Manual")
            
            reference = HMRCReference(
                manual_code=full_code,
                manual_name=manual_name
            )
            references.append(reference)
        
        return references
    
    def extract_legislation_references(self, text: str) -> List[TaxLegislationReference]:
        """Extract tax legislation references from text"""
        references = []
        
        # Find act references
        for match in self.legislation_patterns['acts'].finditer(text):
            act_ref = match.group(1).upper()
            full_name = self.legislation_names.get(act_ref, act_ref)
            
            # Look for associated section references nearby
            surrounding_text = text[max(0, match.start()-100):match.end()+100]
            section_matches = self.legislation_patterns['sections'].finditer(surrounding_text)
            
            for section_match in section_matches:
                section_ref = section_match.group(1)
                
                reference = TaxLegislationReference(
                    act_name=act_ref,
                    section=f"s.{section_ref}",
                    full_name=full_name,
                    reference_type="section",
                    context=surrounding_text.strip()
                )
                references.append(reference)
            
            # If no specific sections found, add general act reference
            if not any(ref.act_name == act_ref for ref in references):
                reference = TaxLegislationReference(
                    act_name=act_ref,
                    full_name=full_name,
                    reference_type="act"
                )
                references.append(reference)
        
        return references
    
    def extract_case_references(self, text: str) -> List[str]:
        """Extract tax case references (basic pattern matching)"""
        # Tax cases often follow standard legal citation patterns
        case_patterns = [
            re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+\[(\d{4})\]', re.IGNORECASE),
            re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+v\.?\s+(HMRC|Inspector of Taxes)\s+\[(\d{4})\]', re.IGNORECASE)
        ]
        
        cases = []
        for pattern in case_patterns:
            for match in pattern.finditer(text):
                case_ref = match.group(0)
                cases.append(case_ref)
        
        return list(set(cases))  # Remove duplicates

class HMRCDocumentProcessor:
    """
    Production-ready HMRC document processing with tax-specific classification
    """
    
    def __init__(self):
        self.reference_extractor = HMRCReferenceExtractor()
        
        # Tax domain classification keywords
        self.tax_domain_keywords = {
            TaxDomain.INCOME_TAX: [
                'income tax', 'ita 2007', 'ittoia', 'employment income', 'trading income',
                'property income', 'savings income', 'dividend income'
            ],
            TaxDomain.CORPORATION_TAX: [
                'corporation tax', 'cta 2009', 'cta 2010', 'corporate', 'company tax',
                'business profits', 'chargeable gains', 'loan relationships'
            ],
            TaxDomain.CAPITAL_GAINS_TAX: [
                'capital gains', 'tcga 1992', 'chargeable gains', 'disposal',
                'acquisition', 'base cost', 'enhancement expenditure'
            ],
            TaxDomain.VALUE_ADDED_TAX: [
                'vat', 'value added tax', 'vata 1994', 'input tax', 'output tax',
                'exempt', 'zero-rated', 'standard rate'
            ],
            TaxDomain.INHERITANCE_TAX: [
                'inheritance tax', 'ihta 1984', 'potentially exempt transfer',
                'chargeable lifetime transfer', 'death estate'
            ],
            TaxDomain.NATIONAL_INSURANCE: [
                'national insurance', 'ni contributions', 'class 1', 'class 2',
                'class 3', 'class 4', 'earnings threshold'
            ],
            TaxDomain.PAYE: [
                'paye', 'pay as you earn', 'payroll', 'p45', 'p46', 'p60',
                'p11d', 'benefits in kind', 'real time information'
            ],
            
            # New categories with keywords
            TaxDomain.MAKING_TAX_DIGITAL: [
                'making tax digital', 'mtd', 'digital reporting', 'digital tax',
                'quarterly reporting', 'software', 'api'
            ],
            TaxDomain.EMPLOYEE_BENEFITS: [
                'benefits in kind', 'company car', 'fuel benefit', 'advisory fuel rates',
                'taxable benefits', 'p11d', 'benefit charge', 'employee benefit'
            ],
            TaxDomain.FUEL_DUTIES: [
                'fuel duty', 'advisory fuel rates', 'vehicle excise duty',
                'company car fuel', 'fuel benefit'
            ],
            TaxDomain.ALCOHOL_TOBACCO_DUTIES: [
                'alcohol duty', 'tobacco duty', 'spirits duty', 'beer duty',
                'wine duty', 'cigarette duty', 'alcoholic products'
            ],
            TaxDomain.CUSTOMS_IMPORT_DUTIES: [
                'customs duty', 'import duty', 'customs clearance', 'cds',
                'customs declaration service', 'tariff', 'import'
            ],
            TaxDomain.MONEY_LAUNDERING_SUPERVISION: [
                'money laundering', 'aml supervision', 'economic crime',
                'supervision handbook', 'aml', 'terrorist financing'
            ],
            TaxDomain.TAX_AVOIDANCE_SCHEMES: [
                'tax avoidance schemes', 'dotas', 'disclosure', 'promoter',
                'notifiable arrangements', 'named schemes'
            ],
            TaxDomain.RESEARCH_DEVELOPMENT_RELIEFS: [
                'research and development', 'r&d relief', 'r&d tax credits',
                'corporate intangibles', 'innovation', 'sme r&d'
            ],
            TaxDomain.SERVICE_AVAILABILITY: [
                'service availability', 'online service', 'system issues',
                'service disruption', 'planned maintenance'
            ],
            TaxDomain.DIGITAL_SERVICES: [
                'digital services tax', 'online services', 'digital platform',
                'digital economy', 'digital transformation'
            ],
            TaxDomain.TRADE_PROCEDURES: [
                'trade procedures', 'export', 'import procedures',
                'customs procedures', 'trade facilitation'
            ],
            TaxDomain.ECONOMIC_CRIME: [
                'economic crime', 'financial crime', 'fraud', 'sanctions',
                'terrorist financing', 'proceeds of crime'
            ],
            TaxDomain.TAX_CREDITS: [
                'tax credits', 'working tax credit', 'child tax credit',
                'universal credit', 'credits system'
            ],
            TaxDomain.BUSINESS_RATES: [
                'business rates', 'non-domestic rates', 'rateable value',
                'rating list', 'rate relief'
            ],
            TaxDomain.ENVIRONMENTAL_TAXES: [
                'climate change levy', 'landfill tax', 'aggregates levy',
                'carbon price', 'environmental tax', 'green tax'
            ]
        }
        
        # Entity type keywords
        self.entity_keywords = {
            'individuals': ['individual', 'person', 'taxpayer', 'employee', 'self-employed'],
            'companies': ['company', 'corporation', 'limited', 'plc', 'ltd', 'corporate'],
            'trusts': ['trust', 'trustee', 'beneficiary', 'settlement'],
            'partnerships': ['partnership', 'partner', 'llp', 'limited liability partnership']
        }
    
    def process_hmrc_document(
        self, 
        text: str, 
        title: str = "",
        url: str = "",
        manual_code: str = ""
    ) -> HMRCMetadata:
        """
        Process HMRC document and extract comprehensive metadata
        """
        # Determine document type
        document_type = self._determine_document_type(text, title, url)
        
        # Determine tax authority level
        authority_level = self._determine_authority_level(document_type, text)
        
        # Classify tax domain
        tax_domain = self._classify_tax_domain(text, title)
        
        # Extract references
        hmrc_references = self.reference_extractor.extract_hmrc_references(text)
        legislation_references = self.reference_extractor.extract_legislation_references(text)
        case_references = self.reference_extractor.extract_case_references(text)
        
        # Determine manual information
        manual_name = None
        if manual_code:
            prefix = re.match(r'^([A-Z]+)', manual_code.upper())
            if prefix:
                manual_name = self.reference_extractor.manual_mappings.get(
                    prefix.group(1), f"{prefix.group(1)} Manual"
                )
        
        # Analyze entity applicability
        entity_analysis = self._analyze_entity_applicability(text)
        
        # Extract tax-specific concepts
        tax_concepts = self._extract_tax_concepts(text, title)
        
        # Generate content hash
        import hashlib
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        
        # Create metadata
        metadata = HMRCMetadata(
            document_type=document_type,
            title=title,
            manual_code=manual_code or None,
            manual_name=manual_name,
            content_id=content_hash,
            authority_level=authority_level,
            tax_domain=tax_domain,
            legislation_references=legislation_references,
            case_references=case_references,
            hmrc_cross_references=hmrc_references,
            document_length=len(text),
            has_examples=self._has_examples(text),
            has_calculations=self._has_calculations(text),
            source_url=url,
            last_updated=datetime.now(),
            affects_individuals=entity_analysis['individuals'],
            affects_companies=entity_analysis['companies'],
            affects_trusts=entity_analysis['trusts'],
            affects_partnerships=entity_analysis['partnerships'],
            keywords=self._extract_keywords(text, title),
            tax_concepts=tax_concepts
        )
        
        # Calculate quality scores
        metadata.completeness_score = self._calculate_completeness(metadata)
        metadata.reference_accuracy_score = self._calculate_reference_accuracy(
            legislation_references, hmrc_references
        )
        
        return metadata
    
    def _determine_document_type(self, text: str, title: str, url: str) -> HMRCDocumentType:
        """Determine HMRC document type"""
        title_lower = title.lower()
        url_lower = url.lower()
        text_sample = text[:1000].lower()
        
        if 'manual' in title_lower or '/hmrc-internal-manuals/' in url_lower:
            return HMRCDocumentType.INTERNAL_MANUAL
        elif 'revenue and customs brief' in title_lower or 'brief' in url_lower:
            return HMRCDocumentType.REVENUE_AND_CUSTOMS_BRIEF
        elif 'technical note' in title_lower:
            return HMRCDocumentType.TECHNICAL_NOTE
        elif 'consultation' in title_lower or 'consultation' in url_lower:
            return HMRCDocumentType.CONSULTATION
        elif 'policy paper' in title_lower:
            return HMRCDocumentType.POLICY_PAPER
        elif any(term in title_lower for term in ['form', 'return', 'p45', 'p46', 'p60']):
            return HMRCDocumentType.FORM
        elif 'bulletin' in title_lower:
            return HMRCDocumentType.BULLETIN
        else:
            return HMRCDocumentType.GUIDANCE
    
    def _determine_authority_level(self, document_type: HMRCDocumentType, text: str) -> TaxAuthority:
        """Determine tax authority level"""
        if document_type == HMRCDocumentType.INTERNAL_MANUAL:
            return TaxAuthority.HMRC_INTERNAL_MANUAL
        elif document_type == HMRCDocumentType.REVENUE_AND_CUSTOMS_BRIEF:
            return TaxAuthority.HMRC_BRIEF
        elif document_type == HMRCDocumentType.TECHNICAL_NOTE:
            return TaxAuthority.HMRC_TECHNICAL_NOTE
        elif document_type in [HMRCDocumentType.GUIDANCE, HMRCDocumentType.PRACTICE_NOTE]:
            return TaxAuthority.HMRC_GUIDANCE
        else:
            return TaxAuthority.COMMENTARY
    
    def _classify_tax_domain(self, text: str, title: str) -> TaxDomain:
        """Classify the tax domain"""
        combined_text = (title + " " + text[:2000]).lower()
        
        # Score each domain
        domain_scores = {}
        for domain, keywords in self.tax_domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                domain_scores[domain] = score
        
        # Return highest scoring domain, or GENERAL if none match
        if domain_scores:
            return max(domain_scores.keys(), key=lambda k: domain_scores[k])
        else:
            return TaxDomain.GENERAL
    
    def _analyze_entity_applicability(self, text: str) -> Dict[str, bool]:
        """Analyze which entities this guidance applies to"""
        text_lower = text.lower()
        
        analysis = {}
        for entity_type, keywords in self.entity_keywords.items():
            analysis[entity_type] = any(keyword in text_lower for keyword in keywords)
        
        return analysis
    
    def _extract_tax_concepts(self, text: str, title: str) -> List[str]:
        """Extract tax-specific concepts"""
        concepts = []
        combined_text = (title + " " + text[:3000]).lower()
        
        concept_patterns = {
            'allowances': ['allowance', 'relief', 'exemption', 'deduction'],
            'charges': ['charge', 'liability', 'assessment', 'computation'],
            'procedures': ['return', 'filing', 'deadline', 'procedure', 'process'],
            'penalties': ['penalty', 'interest', 'surcharge', 'default'],
            'appeals': ['appeal', 'tribunal', 'review', 'dispute'],
            'anti_avoidance': ['avoidance', 'artificial', 'scheme', 'arrangement']
        }
        
        for concept_category, patterns in concept_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                concepts.append(concept_category)
        
        return concepts
    
    def _has_examples(self, text: str) -> bool:
        """Check if document contains examples"""
        example_indicators = ['example', 'for instance', 'case study', 'illustration']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in example_indicators)
    
    def _has_calculations(self, text: str) -> bool:
        """Check if document contains calculations"""
        calculation_indicators = ['£', '%', 'calculate', 'computation', '=', '+', '-', '×', '÷']
        return any(indicator in text for indicator in calculation_indicators)
    
    def _extract_keywords(self, text: str, title: str) -> List[str]:
        """Extract relevant tax keywords"""
        tax_terms = [
            'tax', 'duty', 'rate', 'threshold', 'band', 'relief', 'allowance',
            'liability', 'assessment', 'return', 'filing', 'deadline', 'penalty',
            'interest', 'appeal', 'tribunal', 'compliance', 'avoidance', 'planning'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for term in tax_terms:
            if term in text_lower:
                found_keywords.append(term)
        
        return found_keywords[:15]  # Limit to 15 keywords
    
    def _calculate_completeness(self, metadata: HMRCMetadata) -> float:
        """Calculate metadata completeness score"""
        total_fields = 0
        complete_fields = 0
        
        # Essential fields
        essential_fields = [
            'title', 'document_type', 'tax_domain', 'authority_level'
        ]
        
        for field in essential_fields:
            total_fields += 1
            value = getattr(metadata, field)
            if value and str(value) != "TaxDomain.GENERAL":
                complete_fields += 1
        
        # Important optional fields
        optional_fields = [
            'manual_code', 'legislation_references', 'hmrc_cross_references',
            'keywords', 'tax_concepts'
        ]
        
        for field in optional_fields:
            total_fields += 1
            value = getattr(metadata, field)
            if value:
                complete_fields += 1
        
        return complete_fields / total_fields if total_fields > 0 else 0.0
    
    def _calculate_reference_accuracy(
        self, 
        legislation_refs: List[TaxLegislationReference],
        hmrc_refs: List[HMRCReference]
    ) -> float:
        """Calculate reference accuracy score"""
        total_refs = len(legislation_refs) + len(hmrc_refs)
        if total_refs == 0:
            return 0.0
        
        # Score based on reference completeness
        score = 0.0
        
        for leg_ref in legislation_refs:
            ref_score = 0.0
            if leg_ref.act_name:
                ref_score += 0.5
            if leg_ref.section:
                ref_score += 0.3
            if leg_ref.full_name:
                ref_score += 0.2
            score += ref_score
        
        for hmrc_ref in hmrc_refs:
            ref_score = 0.0
            if hmrc_ref.manual_code:
                ref_score += 0.6
            if hmrc_ref.manual_name:
                ref_score += 0.4
            score += ref_score
        
        return min(score / total_refs, 1.0)

def save_hmrc_metadata(metadata: HMRCMetadata, output_path: Path) -> bool:
    """Save HMRC metadata to JSON file with error handling"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save HMRC metadata to {output_path}: {e}")
        return False

def load_hmrc_metadata(metadata_path: Path) -> Optional[HMRCMetadata]:
    """Load HMRC metadata from JSON file with error handling"""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return HMRCMetadata.from_dict(data)
    except Exception as e:
        logger.error(f"Failed to load HMRC metadata from {metadata_path}: {e}")
        return None
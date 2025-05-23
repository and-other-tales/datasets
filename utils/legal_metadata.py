#!/usr/bin/env python3
"""
Legal Metadata Framework
Production-ready legal document metadata handling with comprehensive 
legal domain awareness, citation tracking, and authority hierarchies.
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

class LegalJurisdiction(Enum):
    """UK Legal Jurisdictions"""
    ENGLAND_WALES = "england_wales"
    SCOTLAND = "scotland"
    NORTHERN_IRELAND = "northern_ireland"
    UK_WIDE = "uk_wide"
    EU_RETAINED = "eu_retained"

class LegalSourceType(Enum):
    """Types of legal sources"""
    STATUTE = "statute"
    CASE = "case"
    REGULATION = "regulation"
    STATUTORY_INSTRUMENT = "statutory_instrument"
    COMMENTARY = "commentary"
    HMRC_GUIDANCE = "hmrc_guidance"
    COURT_RULE = "court_rule"
    PRACTICE_DIRECTION = "practice_direction"

class CourtHierarchy(Enum):
    """UK Court Hierarchy (1=highest authority)"""
    SUPREME_COURT = 1
    HOUSE_OF_LORDS = 1  # Historical
    COURT_OF_APPEAL = 2
    HIGH_COURT = 3
    CROWN_COURT = 4
    COUNTY_COURT = 4
    MAGISTRATES_COURT = 5
    TRIBUNAL = 6
    COMMENTARY = 7
    UNKNOWN = 8

@dataclass
class LegalCitation:
    """Structured legal citation"""
    citation_string: str
    court: Optional[str] = None
    year: Optional[int] = None
    case_number: Optional[str] = None
    neutral_citation: Optional[str] = None
    law_report_citation: Optional[str] = None
    is_neutral: bool = False
    authority_level: Optional[int] = None

@dataclass
class LegalMetadata:
    """
    Comprehensive legal document metadata following GUIDANCE.md specifications
    """
    # Core identification
    source_type: LegalSourceType
    jurisdiction: LegalJurisdiction
    title: str
    citation: str
    content_id: str = field(default_factory=lambda: "")
    
    # Authority and hierarchy
    authority_level: int = 8  # Default to lowest authority
    effective_date: Optional[datetime] = None
    amendment_history: List[str] = field(default_factory=list)
    superseded_by: Optional[str] = None
    supersedes: List[str] = field(default_factory=list)
    
    # Legal domain classification
    legal_area: str = "general"
    subject_matter: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # Citation and precedent tracking
    citations_found: List[LegalCitation] = field(default_factory=list)
    cited_by: List[str] = field(default_factory=list)
    cross_references: List[str] = field(default_factory=list)
    
    # Content characteristics
    document_length: int = 0
    section_count: int = 0
    has_amendments: bool = False
    language: str = "en-GB"
    
    # Technical metadata
    source_url: Optional[str] = None
    last_updated: Optional[datetime] = None
    first_published: Optional[datetime] = None
    content_hash: Optional[str] = None
    extraction_method: str = "unknown"
    
    # Validation and quality
    citation_accuracy_score: float = 0.0
    content_quality_score: float = 0.0
    completeness_score: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary with enum handling"""
        result = asdict(self)
        # Convert enums to values
        result['source_type'] = self.source_type.value
        result['jurisdiction'] = self.jurisdiction.value
        # Handle datetime serialization
        for field_name in ['effective_date', 'last_updated', 'first_published']:
            if result[field_name]:
                result[field_name] = result[field_name].isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LegalMetadata':
        """Create from dictionary with enum handling"""
        # Convert string enums back to enum objects
        if 'source_type' in data and isinstance(data['source_type'], str):
            data['source_type'] = LegalSourceType(data['source_type'])
        if 'jurisdiction' in data and isinstance(data['jurisdiction'], str):
            data['jurisdiction'] = LegalJurisdiction(data['jurisdiction'])
        
        # Handle datetime deserialization
        for field_name in ['effective_date', 'last_updated', 'first_published']:
            if data.get(field_name) and isinstance(data[field_name], str):
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)

class LegalCitationExtractor:
    """
    Production-ready legal citation extraction with comprehensive UK patterns
    """
    
    def __init__(self):
        # UK Citation patterns from GUIDANCE.md
        self.citation_patterns = {
            'neutral_citation': re.compile(
                r'\[(\d{4})\]\s+(UKSC|UKHL|UKPC|EWCA|EWHC|EWCOP|CSIH|CSOH|NICA|NIQB)\s+(\d+)',
                re.IGNORECASE
            ),
            'law_reports': re.compile(
                r'\[(\d{4})\]\s+(\d+)\s+(AC|WLR|All\s+ER|QB|Ch|Fam|P|Lloyd\'s\s+Rep)',
                re.IGNORECASE
            ),
            'weekly_law_reports': re.compile(
                r'\[(\d{4})\]\s+(\d+)\s+W\.?L\.?R\.?\s+(\d+)',
                re.IGNORECASE
            ),
            'section_references': re.compile(
                r'(?:section|s\.|§)\s*(\d+(?:\(\d+\))*(?:[A-Z]*)?)',
                re.IGNORECASE
            ),
            'act_references': re.compile(
                r'([A-Z][A-Za-z\s]+Act\s+\d{4})',
                re.IGNORECASE
            ),
            'statutory_instruments': re.compile(
                r'S\.?I\.?\s+(\d{4}/\d+)',
                re.IGNORECASE
            ),
            'hmrc_references': re.compile(
                r'((?:HMRC|Revenue)\s+(?:Manual|Guidance|Brief)\s+[A-Z]+\d+)',
                re.IGNORECASE
            )
        }
        
        # Court abbreviation mapping
        self.court_mapping = {
            'UKSC': ('Supreme Court', CourtHierarchy.SUPREME_COURT),
            'UKHL': ('House of Lords', CourtHierarchy.HOUSE_OF_LORDS),
            'UKPC': ('Privy Council', CourtHierarchy.SUPREME_COURT),
            'EWCA': ('Court of Appeal', CourtHierarchy.COURT_OF_APPEAL),
            'EWHC': ('High Court', CourtHierarchy.HIGH_COURT),
            'EWCOP': ('Court of Protection', CourtHierarchy.HIGH_COURT),
            'CSIH': ('Court of Session (Inner House)', CourtHierarchy.COURT_OF_APPEAL),
            'CSOH': ('Court of Session (Outer House)', CourtHierarchy.HIGH_COURT),
            'NICA': ('Northern Ireland Court of Appeal', CourtHierarchy.COURT_OF_APPEAL),
            'NIQB': ('Northern Ireland High Court', CourtHierarchy.HIGH_COURT)
        }
    
    def extract_citations(self, text: str) -> List[LegalCitation]:
        """Extract all legal citations from text"""
        citations = []
        
        # Extract neutral citations
        for match in self.citation_patterns['neutral_citation'].finditer(text):
            year, court_abbr, number = match.groups()
            court_name, authority_level = self.court_mapping.get(
                court_abbr.upper(), 
                (court_abbr, CourtHierarchy.UNKNOWN)
            )
            
            citation = LegalCitation(
                citation_string=match.group(0),
                court=court_name,
                year=int(year),
                case_number=number,
                neutral_citation=match.group(0),
                is_neutral=True,
                authority_level=authority_level.value
            )
            citations.append(citation)
        
        # Extract law report citations
        for match in self.citation_patterns['law_reports'].finditer(text):
            year, volume, report = match.groups()
            citation = LegalCitation(
                citation_string=match.group(0),
                year=int(year),
                law_report_citation=match.group(0),
                is_neutral=False
            )
            citations.append(citation)
        
        return citations
    
    def extract_statutory_references(self, text: str) -> List[str]:
        """Extract statutory and regulatory references"""
        references = []
        
        # Section references
        for match in self.citation_patterns['section_references'].finditer(text):
            references.append(f"Section {match.group(1)}")
        
        # Act references
        for match in self.citation_patterns['act_references'].finditer(text):
            references.append(match.group(1))
        
        # Statutory instruments
        for match in self.citation_patterns['statutory_instruments'].finditer(text):
            references.append(f"SI {match.group(1)}")
        
        # HMRC references
        for match in self.citation_patterns['hmrc_references'].finditer(text):
            references.append(match.group(1))
        
        return list(set(references))  # Remove duplicates

class LegalDocumentProcessor:
    """
    Production-ready legal document processing with structure preservation
    """
    
    def __init__(self):
        self.citation_extractor = LegalCitationExtractor()
        
        # Legal document structure patterns
        self.structure_patterns = {
            'sections': re.compile(
                r'(?:^|\n)\s*(?:Section|SECTION)\s+(\d+(?:\.\d+)*)\s*[:\-]?\s*(.*?)(?=\n(?:Section|SECTION)|\Z)',
                re.MULTILINE | re.DOTALL
            ),
            'subsections': re.compile(
                r'(?:^|\n)\s*\((\d+)\)\s*(.*?)(?=\n\s*\(\d+\)|\Z)',
                re.MULTILINE | re.DOTALL
            ),
            'paragraphs': re.compile(
                r'(?:^|\n)\s*\(([a-z])\)\s*(.*?)(?=\n\s*\([a-z]\)|\Z)',
                re.MULTILINE | re.DOTALL
            ),
            'amendments': re.compile(
                r'\[.*?amended\s+by.*?\]',
                re.IGNORECASE
            )
        }
    
    def process_document(
        self, 
        text: str, 
        title: str = "",
        source_url: str = "",
        document_type: str = "unknown"
    ) -> LegalMetadata:
        """
        Process a legal document and extract comprehensive metadata
        """
        # Determine source type
        source_type = self._determine_source_type(text, title, document_type)
        
        # Determine jurisdiction
        jurisdiction = self._determine_jurisdiction(text, title)
        
        # Extract citations
        citations = self.citation_extractor.extract_citations(text)
        
        # Extract statutory references
        cross_references = self.citation_extractor.extract_statutory_references(text)
        
        # Determine authority level
        authority_level = self._determine_authority_level(source_type, citations)
        
        # Extract structural information
        sections = self.structure_patterns['sections'].findall(text)
        amendments = self.structure_patterns['amendments'].findall(text)
        
        # Classify legal area
        legal_area = self._classify_legal_area(text, title)
        
        # Generate content hash for deduplication
        import hashlib
        content_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
        
        # Create comprehensive metadata
        metadata = LegalMetadata(
            source_type=source_type,
            jurisdiction=jurisdiction,
            title=title,
            citation=self._extract_primary_citation(text, title),
            content_id=content_hash,
            authority_level=authority_level,
            legal_area=legal_area,
            citations_found=citations,
            cross_references=cross_references,
            document_length=len(text),
            section_count=len(sections),
            has_amendments=len(amendments) > 0,
            amendment_history=amendments,
            source_url=source_url,
            last_updated=datetime.now(),
            extraction_method="legal_document_processor",
            keywords=self._extract_keywords(text, title),
            subject_matter=self._extract_subject_matter(text, title)
        )
        
        # Calculate quality scores
        metadata.citation_accuracy_score = self._calculate_citation_accuracy(citations)
        metadata.content_quality_score = self._calculate_content_quality(text)
        metadata.completeness_score = self._calculate_completeness(metadata)
        
        return metadata
    
    def _determine_source_type(self, text: str, title: str, document_type: str) -> LegalSourceType:
        """Determine the type of legal source"""
        title_lower = title.lower()
        text_lower = text.lower()
        
        if any(term in title_lower for term in ['act', 'statute']):
            return LegalSourceType.STATUTE
        elif any(term in title_lower for term in ['v.', 'v ', 'versus', '[20', 'case']):
            return LegalSourceType.CASE
        elif any(term in title_lower for term in ['regulation', 'rules']):
            return LegalSourceType.REGULATION
        elif 's.i.' in text_lower or 'statutory instrument' in text_lower:
            return LegalSourceType.STATUTORY_INSTRUMENT
        elif any(term in title_lower for term in ['hmrc', 'revenue', 'tax']):
            return LegalSourceType.HMRC_GUIDANCE
        elif 'court rule' in title_lower or 'cpr' in title_lower:
            return LegalSourceType.COURT_RULE
        elif 'practice direction' in title_lower:
            return LegalSourceType.PRACTICE_DIRECTION
        else:
            return LegalSourceType.COMMENTARY
    
    def _determine_jurisdiction(self, text: str, title: str) -> LegalJurisdiction:
        """Determine the legal jurisdiction"""
        combined_text = (title + " " + text[:1000]).lower()
        
        if any(term in combined_text for term in ['scotland', 'scottish', 'csih', 'csoh']):
            return LegalJurisdiction.SCOTLAND
        elif any(term in combined_text for term in ['northern ireland', 'nica', 'niqb']):
            return LegalJurisdiction.NORTHERN_IRELAND
        elif any(term in combined_text for term in ['england', 'wales', 'ewca', 'ewhc']):
            return LegalJurisdiction.ENGLAND_WALES
        elif any(term in combined_text for term in ['uk', 'united kingdom', 'uksc', 'ukhl']):
            return LegalJurisdiction.UK_WIDE
        else:
            return LegalJurisdiction.ENGLAND_WALES  # Default
    
    def _determine_authority_level(self, source_type: LegalSourceType, citations: List[LegalCitation]) -> int:
        """Determine authority level of the document"""
        if source_type == LegalSourceType.STATUTE:
            return 1  # Primary legislation is highest authority
        elif source_type in [LegalSourceType.REGULATION, LegalSourceType.STATUTORY_INSTRUMENT]:
            return 2  # Secondary legislation
        elif source_type == LegalSourceType.CASE:
            # Determine by court level
            if citations:
                return min(citation.authority_level for citation in citations if citation.authority_level)
            return CourtHierarchy.UNKNOWN.value
        elif source_type == LegalSourceType.HMRC_GUIDANCE:
            return 3  # HMRC guidance has high authority in tax matters
        else:
            return CourtHierarchy.COMMENTARY.value
    
    def _classify_legal_area(self, text: str, title: str) -> str:
        """Classify the legal area/domain"""
        combined_text = (title + " " + text[:2000]).lower()
        
        area_keywords = {
            'criminal': ['criminal', 'crime', 'prosecution', 'defendant', 'guilty', 'sentence'],
            'civil': ['civil', 'tort', 'negligence', 'contract', 'damages'],
            'tax': ['tax', 'vat', 'hmrc', 'revenue', 'duty', 'allowance', 'relief'],
            'employment': ['employment', 'worker', 'dismissal', 'redundancy', 'discrimination'],
            'housing': ['housing', 'landlord', 'tenant', 'rent', 'eviction', 'lease'],
            'family': ['family', 'marriage', 'divorce', 'custody', 'child', 'matrimonial'],
            'commercial': ['commercial', 'business', 'company', 'corporate', 'insolvency'],
            'property': ['property', 'land', 'conveyancing', 'real estate', 'mortgage'],
            'immigration': ['immigration', 'visa', 'asylum', 'deportation', 'nationality'],
            'administrative': ['administrative', 'judicial review', 'public law', 'tribunal']
        }
        
        for area, keywords in area_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                return area
        
        return "general"
    
    def _extract_primary_citation(self, text: str, title: str) -> str:
        """Extract or generate primary citation for the document"""
        citations = self.citation_extractor.extract_citations(text)
        
        if citations and citations[0].neutral_citation:
            return citations[0].neutral_citation
        elif citations and citations[0].law_report_citation:
            return citations[0].law_report_citation
        else:
            # Generate citation from title
            return title or "Unknown Citation"
    
    def _extract_keywords(self, text: str, title: str) -> List[str]:
        """Extract relevant legal keywords"""
        # Simple keyword extraction - could be enhanced with NLP
        legal_terms = [
            'statute', 'regulation', 'case law', 'precedent', 'jurisdiction',
            'liability', 'damages', 'injunction', 'appeal', 'tribunal',
            'legislative', 'judicial', 'executive', 'constitutional'
        ]
        
        found_keywords = []
        text_lower = text.lower()
        
        for term in legal_terms:
            if term in text_lower:
                found_keywords.append(term)
        
        return found_keywords[:10]  # Limit to 10 keywords
    
    def _extract_subject_matter(self, text: str, title: str) -> List[str]:
        """Extract subject matter tags"""
        # Extract specific legal topics
        subjects = []
        combined_text = (title + " " + text[:1000]).lower()
        
        subject_patterns = {
            'contracts': ['contract', 'agreement', 'terms', 'breach'],
            'torts': ['negligence', 'duty of care', 'tort', 'damages'],
            'constitutional': ['constitutional', 'human rights', 'judicial review'],
            'procedure': ['procedure', 'process', 'court rules', 'evidence'],
            'remedies': ['remedy', 'injunction', 'damages', 'specific performance']
        }
        
        for subject, patterns in subject_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                subjects.append(subject)
        
        return subjects
    
    def _calculate_citation_accuracy(self, citations: List[LegalCitation]) -> float:
        """Calculate citation accuracy score"""
        if not citations:
            return 0.0
        
        # Score based on citation completeness and format
        total_score = 0
        for citation in citations:
            score = 0
            if citation.year:
                score += 0.3
            if citation.court:
                score += 0.3
            if citation.is_neutral:
                score += 0.4
            total_score += score
        
        return min(total_score / len(citations), 1.0)
    
    def _calculate_content_quality(self, text: str) -> float:
        """Calculate content quality score"""
        score = 0.0
        
        # Length scoring
        if len(text) > 1000:
            score += 0.3
        elif len(text) > 500:
            score += 0.2
        
        # Structure scoring
        if 'section' in text.lower():
            score += 0.2
        if re.search(r'\(\d+\)', text):  # Has subsections
            score += 0.2
        
        # Legal language scoring
        legal_indicators = ['pursuant', 'whereas', 'hereby', 'therein', 'aforementioned']
        found_indicators = sum(1 for indicator in legal_indicators if indicator in text.lower())
        score += min(found_indicators * 0.1, 0.3)
        
        return min(score, 1.0)
    
    def _calculate_completeness(self, metadata: LegalMetadata) -> float:
        """Calculate metadata completeness score"""
        total_fields = 0
        complete_fields = 0
        
        # Check essential fields
        essential_fields = [
            'title', 'citation', 'source_type', 'jurisdiction', 
            'legal_area', 'authority_level'
        ]
        
        for field in essential_fields:
            total_fields += 1
            value = getattr(metadata, field)
            if value and value != "unknown" and value != "general":
                complete_fields += 1
        
        # Check optional but valuable fields
        optional_fields = [
            'effective_date', 'citations_found', 'cross_references',
            'keywords', 'subject_matter'
        ]
        
        for field in optional_fields:
            total_fields += 1
            value = getattr(metadata, field)
            if value:
                complete_fields += 1
        
        return complete_fields / total_fields if total_fields > 0 else 0.0

def save_legal_metadata(metadata: LegalMetadata, output_path: Path) -> bool:
    """Save legal metadata to JSON file with error handling"""
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata.to_dict(), f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Failed to save metadata to {output_path}: {e}")
        return False

def load_legal_metadata(metadata_path: Path) -> Optional[LegalMetadata]:
    """Load legal metadata from JSON file with error handling"""
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return LegalMetadata.from_dict(data)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_path}: {e}")
        return None
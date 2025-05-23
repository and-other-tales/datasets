#!/usr/bin/env python3
"""
Legal Citation Graph and Authority System
Production-ready citation relationship tracking, precedent analysis,
and authority-based legal document ranking.
"""

import json
import logging
import networkx as nx
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Tuple, Union
from collections import defaultdict, deque
from dataclasses import dataclass

from .legal_metadata import LegalMetadata, LegalCitation, CourtHierarchy, LegalSourceType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CitationRelationship:
    """Represents a citation relationship between legal documents"""
    citing_doc: str
    cited_doc: str
    citation_text: str
    relationship_type: str  # 'follows', 'distinguishes', 'overrules', 'applies', 'considers'
    context: str = ""
    strength: float = 1.0  # Relationship strength (0-1)
    extraction_confidence: float = 1.0

@dataclass
class PrecedentAuthority:
    """Authority ranking for legal precedents"""
    document_id: str
    authority_score: float
    court_hierarchy_score: float
    temporal_relevance_score: float
    citation_count: int
    positive_citations: int
    negative_citations: int
    judicial_consideration_score: float

class LegalCitationGraph:
    """
    Production-ready legal citation graph with precedent analysis
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.citation_relationships: Dict[str, List[CitationRelationship]] = defaultdict(list)
        self.authority_scores: Dict[str, PrecedentAuthority] = {}
        self.document_metadata: Dict[str, LegalMetadata] = {}
        
        # Citation relationship patterns
        self.relationship_patterns = {
            'follows': [
                r'follow(?:s|ing)?\s+(?:the\s+)?(?:decision|ruling|principle)\s+in\s+(.+?)(?:\s|;|\.)',
                r'as\s+(?:held|decided|established)\s+in\s+(.+?)(?:\s|;|\.)',
                r'in\s+accordance\s+with\s+(.+?)(?:\s|;|\.)'
            ],
            'distinguishes': [
                r'distinguish(?:es|able|ed)?\s+(?:from\s+)?(.+?)(?:\s|;|\.)',
                r'(?:can|may)\s+be\s+distinguished\s+from\s+(.+?)(?:\s|;|\.)',
                r'unlike\s+(?:in\s+)?(.+?)(?:\s|;|\.)'
            ],
            'overrules': [
                r'overrul(?:es?|ed|ing)\s+(.+?)(?:\s|;|\.)',
                r'reverses?\s+(?:the\s+decision\s+in\s+)?(.+?)(?:\s|;|\.)',
                r'no\s+longer\s+follows?\s+(.+?)(?:\s|;|\.)'
            ],
            'applies': [
                r'appl(?:ies|ying|ied)\s+(?:the\s+principle\s+from\s+)?(.+?)(?:\s|;|\.)',
                r'using\s+(?:the\s+test\s+from\s+)?(.+?)(?:\s|;|\.)',
                r'pursuant\s+to\s+(.+?)(?:\s|;|\.)'
            ],
            'considers': [
                r'consider(?:s|ed|ing)\s+(.+?)(?:\s|;|\.)',
                r'having\s+regard\s+to\s+(.+?)(?:\s|;|\.)',
                r'with\s+reference\s+to\s+(.+?)(?:\s|;|\.)'
            ]
        }
    
    def add_document(self, doc_id: str, metadata: LegalMetadata) -> None:
        """Add a document to the citation graph"""
        self.document_metadata[doc_id] = metadata
        
        # Add node with metadata
        self.graph.add_node(doc_id, **{
            'title': metadata.title,
            'court': getattr(metadata, 'court', 'Unknown'),
            'year': getattr(metadata, 'year', None),
            'authority_level': metadata.authority_level,
            'legal_area': metadata.legal_area,
            'source_type': metadata.source_type.value,
            'jurisdiction': metadata.jurisdiction.value,
            'effective_date': metadata.effective_date
        })
        
        logger.debug(f"Added document {doc_id} to citation graph")
    
    def extract_citation_relationships(self, doc_id: str, content: str) -> List[CitationRelationship]:
        """Extract citation relationships from document content"""
        relationships = []
        
        # Import here to avoid circular imports
        from .legal_metadata import LegalCitationExtractor
        extractor = LegalCitationExtractor()
        
        # Extract citations from content
        citations = extractor.extract_citations(content)
        
        # For each citation, analyze the surrounding context
        for citation in citations:
            # Find the context around the citation
            citation_start = content.find(citation.citation_string)
            if citation_start == -1:
                continue
            
            # Extract context (100 chars before and after)
            context_start = max(0, citation_start - 100)
            context_end = min(len(content), citation_start + len(citation.citation_string) + 100)
            context = content[context_start:context_end]
            
            # Determine relationship type from context
            relationship_type = self._determine_relationship_type(context)
            
            # Create relationship
            relationship = CitationRelationship(
                citing_doc=doc_id,
                cited_doc=citation.citation_string,
                citation_text=citation.citation_string,
                relationship_type=relationship_type,
                context=context,
                strength=self._calculate_relationship_strength(context, relationship_type),
                extraction_confidence=0.8  # Base confidence
            )
            
            relationships.append(relationship)
        
        # Store relationships
        self.citation_relationships[doc_id].extend(relationships)
        
        return relationships
    
    def _determine_relationship_type(self, context: str) -> str:
        """Determine the type of citation relationship from context"""
        context_lower = context.lower()
        
        # Check each relationship type
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                if any(keyword in context_lower for keyword in pattern.split()):
                    return rel_type
        
        return 'considers'  # Default relationship type
    
    def _calculate_relationship_strength(self, context: str, relationship_type: str) -> float:
        """Calculate the strength of a citation relationship"""
        base_strength = {
            'follows': 1.0,
            'applies': 0.9,
            'considers': 0.6,
            'distinguishes': 0.4,
            'overrules': 1.0
        }.get(relationship_type, 0.5)
        
        # Adjust based on context strength indicators
        strength_indicators = [
            'clearly', 'explicitly', 'directly', 'specifically',
            'definitively', 'unambiguously', 'precisely'
        ]
        
        weakness_indicators = [
            'arguably', 'possibly', 'potentially', 'might',
            'could', 'perhaps', 'seemingly'
        ]
        
        context_lower = context.lower()
        
        # Boost for strength indicators
        strength_boost = sum(0.1 for indicator in strength_indicators if indicator in context_lower)
        
        # Reduce for weakness indicators
        strength_reduction = sum(0.1 for indicator in weakness_indicators if indicator in context_lower)
        
        final_strength = base_strength + strength_boost - strength_reduction
        return max(0.1, min(1.0, final_strength))
    
    def build_citation_network(self) -> None:
        """Build the complete citation network from relationships"""
        logger.info("Building citation network...")
        
        # Add edges for all relationships
        for doc_id, relationships in self.citation_relationships.items():
            for rel in relationships:
                if rel.cited_doc in self.document_metadata:
                    self.graph.add_edge(
                        rel.citing_doc,
                        rel.cited_doc,
                        relationship_type=rel.relationship_type,
                        strength=rel.strength,
                        context=rel.context[:100],  # Truncate context
                        confidence=rel.extraction_confidence
                    )
        
        logger.info(f"Citation network built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def calculate_authority_scores(self) -> Dict[str, PrecedentAuthority]:
        """Calculate comprehensive authority scores for all documents"""
        logger.info("Calculating authority scores...")
        
        authority_scores = {}
        
        for doc_id in self.graph.nodes():
            metadata = self.document_metadata.get(doc_id)
            if not metadata:
                continue
            
            # Court hierarchy score (higher court = higher score)
            court_score = self._calculate_court_hierarchy_score(metadata.authority_level)
            
            # Citation count analysis
            in_degree = self.graph.in_degree(doc_id)  # How many times cited
            out_degree = self.graph.out_degree(doc_id)  # How many cases it cites
            
            # Analyze positive vs negative citations
            positive_citations, negative_citations = self._analyze_citation_sentiment(doc_id)
            
            # Temporal relevance (more recent = more relevant, but not always)
            temporal_score = self._calculate_temporal_relevance(metadata)
            
            # Judicial consideration score (how seriously considered)
            judicial_score = self._calculate_judicial_consideration_score(doc_id)
            
            # Combined authority score
            authority_score = self._calculate_combined_authority_score(
                court_score, in_degree, positive_citations, negative_citations,
                temporal_score, judicial_score
            )
            
            authority = PrecedentAuthority(
                document_id=doc_id,
                authority_score=authority_score,
                court_hierarchy_score=court_score,
                temporal_relevance_score=temporal_score,
                citation_count=in_degree,
                positive_citations=positive_citations,
                negative_citations=negative_citations,
                judicial_consideration_score=judicial_score
            )
            
            authority_scores[doc_id] = authority
        
        self.authority_scores = authority_scores
        logger.info(f"Authority scores calculated for {len(authority_scores)} documents")
        
        return authority_scores
    
    def _calculate_court_hierarchy_score(self, authority_level: int) -> float:
        """Calculate score based on court hierarchy"""
        # Invert authority level (lower number = higher court)
        max_level = max(level.value for level in CourtHierarchy)
        return (max_level - authority_level + 1) / max_level
    
    def _analyze_citation_sentiment(self, doc_id: str) -> Tuple[int, int]:
        """Analyze whether citations are positive or negative"""
        positive = 0
        negative = 0
        
        # Check incoming edges (how this case is cited)
        for citing_doc in self.graph.predecessors(doc_id):
            edge_data = self.graph.get_edge_data(citing_doc, doc_id)
            if edge_data:
                rel_type = edge_data.get('relationship_type', 'considers')
                
                if rel_type in ['follows', 'applies']:
                    positive += 1
                elif rel_type in ['distinguishes', 'overrules']:
                    negative += 1
                else:
                    positive += 0.5  # Neutral consideration
        
        return positive, negative
    
    def _calculate_temporal_relevance(self, metadata: LegalMetadata) -> float:
        """Calculate temporal relevance score"""
        if not metadata.effective_date:
            return 0.5  # Default for unknown dates
        
        # Calculate years since decision
        years_ago = (datetime.now() - metadata.effective_date).days / 365.25
        
        # Legal precedents don't lose authority quickly, but very old cases may be less relevant
        if years_ago < 5:
            return 1.0  # Very recent
        elif years_ago < 15:
            return 0.9  # Recent
        elif years_ago < 30:
            return 0.8  # Moderately recent
        elif years_ago < 50:
            return 0.7  # Older but still relevant
        else:
            return 0.6  # Historical but potentially important
    
    def _calculate_judicial_consideration_score(self, doc_id: str) -> float:
        """Calculate how seriously this case is considered judicially"""
        # Based on relationship strengths and types
        total_strength = 0.0
        consideration_count = 0
        
        for citing_doc in self.graph.predecessors(doc_id):
            edge_data = self.graph.get_edge_data(citing_doc, doc_id)
            if edge_data:
                strength = edge_data.get('strength', 0.5)
                total_strength += strength
                consideration_count += 1
        
        if consideration_count == 0:
            return 0.0
        
        return total_strength / consideration_count
    
    def _calculate_combined_authority_score(
        self,
        court_score: float,
        citation_count: int,
        positive_citations: int,
        negative_citations: int,
        temporal_score: float,
        judicial_score: float
    ) -> float:
        """Calculate combined authority score"""
        
        # Base authority from court hierarchy (40% weight)
        base_authority = court_score * 0.4
        
        # Citation impact (30% weight)
        citation_impact = 0.0
        if citation_count > 0:
            positive_ratio = positive_citations / citation_count
            citation_frequency = min(citation_count / 10.0, 1.0)  # Normalize to max 10 citations
            citation_impact = (positive_ratio * 0.7 + citation_frequency * 0.3) * 0.3
        
        # Temporal relevance (15% weight)
        temporal_component = temporal_score * 0.15
        
        # Judicial consideration quality (15% weight)
        judicial_component = judicial_score * 0.15
        
        return base_authority + citation_impact + temporal_component + judicial_component
    
    def find_relevant_precedents(
        self,
        query_legal_area: str,
        query_jurisdiction: str,
        max_results: int = 10,
        min_authority_score: float = 0.3
    ) -> List[Tuple[str, PrecedentAuthority]]:
        """Find most relevant precedents for a legal query"""
        
        relevant_docs = []
        
        for doc_id, authority in self.authority_scores.items():
            metadata = self.document_metadata.get(doc_id)
            if not metadata:
                continue
            
            # Filter by legal area
            if query_legal_area != "general" and metadata.legal_area != query_legal_area:
                continue
            
            # Filter by jurisdiction (allow UK-wide to match any jurisdiction)
            if (query_jurisdiction != "uk_wide" and 
                metadata.jurisdiction.value != query_jurisdiction and 
                metadata.jurisdiction.value != "uk_wide"):
                continue
            
            # Filter by minimum authority score
            if authority.authority_score < min_authority_score:
                continue
            
            relevant_docs.append((doc_id, authority))
        
        # Sort by authority score (descending)
        relevant_docs.sort(key=lambda x: x[1].authority_score, reverse=True)
        
        return relevant_docs[:max_results]
    
    def get_citation_path(self, source_doc: str, target_doc: str) -> Optional[List[str]]:
        """Find citation path between two documents"""
        try:
            path = nx.shortest_path(self.graph, source_doc, target_doc)
            return path
        except nx.NetworkXNoPath:
            return None
    
    def find_landmark_cases(self, legal_area: str = None) -> List[Tuple[str, PrecedentAuthority]]:
        """Identify landmark cases (highest authority + most cited)"""
        
        candidates = []
        
        for doc_id, authority in self.authority_scores.items():
            metadata = self.document_metadata.get(doc_id)
            if not metadata:
                continue
            
            # Filter by legal area if specified
            if legal_area and metadata.legal_area != legal_area:
                continue
            
            # Landmark criteria: high authority + significant citations
            if (authority.authority_score > 0.7 and 
                authority.citation_count > 3 and
                authority.positive_citations > authority.negative_citations):
                
                candidates.append((doc_id, authority))
        
        # Sort by combined score: authority + citation impact
        candidates.sort(
            key=lambda x: x[1].authority_score + (x[1].citation_count * 0.1),
            reverse=True
        )
        
        return candidates[:20]  # Top 20 landmark cases
    
    def export_graph_data(self, output_path: Path) -> bool:
        """Export citation graph data for external analysis"""
        try:
            graph_data = {
                'nodes': [],
                'edges': [],
                'metadata': {
                    'total_nodes': self.graph.number_of_nodes(),
                    'total_edges': self.graph.number_of_edges(),
                    'created': datetime.now().isoformat()
                }
            }
            
            # Export nodes
            for node_id in self.graph.nodes():
                node_data = self.graph.nodes[node_id]
                authority = self.authority_scores.get(node_id)
                
                node_export = {
                    'id': node_id,
                    'title': node_data.get('title', ''),
                    'authority_level': node_data.get('authority_level', 8),
                    'legal_area': node_data.get('legal_area', ''),
                    'authority_score': authority.authority_score if authority else 0.0,
                    'citation_count': authority.citation_count if authority else 0
                }
                graph_data['nodes'].append(node_export)
            
            # Export edges
            for source, target in self.graph.edges():
                edge_data = self.graph.get_edge_data(source, target)
                edge_export = {
                    'source': source,
                    'target': target,
                    'relationship_type': edge_data.get('relationship_type', 'considers'),
                    'strength': edge_data.get('strength', 0.5)
                }
                graph_data['edges'].append(edge_export)
            
            # Save to file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Citation graph exported to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export citation graph: {e}")
            return False
    
    def generate_precedent_report(self, doc_id: str) -> Dict:
        """Generate comprehensive precedent analysis report"""
        metadata = self.document_metadata.get(doc_id)
        authority = self.authority_scores.get(doc_id)
        
        if not metadata or not authority:
            return {}
        
        # Find cases this document cites
        cited_cases = list(self.graph.successors(doc_id))
        
        # Find cases that cite this document
        citing_cases = list(self.graph.predecessors(doc_id))
        
        # Find related cases (same legal area)
        related_cases = self.find_relevant_precedents(
            metadata.legal_area,
            metadata.jurisdiction.value,
            max_results=5
        )
        
        report = {
            'document_id': doc_id,
            'title': metadata.title,
            'authority_analysis': {
                'overall_score': authority.authority_score,
                'court_hierarchy_score': authority.court_hierarchy_score,
                'citation_impact': authority.citation_count,
                'positive_citations': authority.positive_citations,
                'negative_citations': authority.negative_citations,
                'temporal_relevance': authority.temporal_relevance_score
            },
            'citation_network': {
                'cites': len(cited_cases),
                'cited_by': len(citing_cases),
                'citing_cases': citing_cases[:10]  # Limit for readability
            },
            'related_precedents': [
                {
                    'id': case_id,
                    'title': self.document_metadata[case_id].title,
                    'authority_score': auth.authority_score
                }
                for case_id, auth in related_cases[:5]
                if case_id != doc_id
            ],
            'legal_significance': self._assess_legal_significance(authority),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def _assess_legal_significance(self, authority: PrecedentAuthority) -> str:
        """Assess the legal significance of a case"""
        if authority.authority_score > 0.8:
            return "Landmark case - highest authority"
        elif authority.authority_score > 0.7:
            return "Highly authoritative precedent"
        elif authority.authority_score > 0.6:
            return "Important precedent"
        elif authority.authority_score > 0.4:
            return "Relevant precedent"
        else:
            return "Limited precedential value"

class LegalPrecedentSearchEngine:
    """
    Advanced search engine for legal precedents using citation graph
    """
    
    def __init__(self, citation_graph: LegalCitationGraph):
        self.citation_graph = citation_graph
    
    def search_precedents(
        self,
        query_terms: List[str],
        legal_area: str = None,
        jurisdiction: str = None,
        authority_threshold: float = 0.3,
        max_results: int = 10
    ) -> List[Dict]:
        """Search for relevant precedents"""
        
        results = []
        
        # Get base set of relevant documents
        relevant_docs = self.citation_graph.find_relevant_precedents(
            legal_area or "general",
            jurisdiction or "uk_wide",
            max_results=max_results * 2,  # Get more to filter
            min_authority_score=authority_threshold
        )
        
        # Score documents based on query terms
        for doc_id, authority in relevant_docs:
            metadata = self.citation_graph.document_metadata.get(doc_id)
            if not metadata:
                continue
            
            # Calculate relevance score
            relevance_score = self._calculate_query_relevance(
                query_terms, metadata.title, metadata.keywords
            )
            
            if relevance_score > 0.1:  # Minimum relevance threshold
                result = {
                    'document_id': doc_id,
                    'title': metadata.title,
                    'citation': metadata.citation,
                    'authority_score': authority.authority_score,
                    'relevance_score': relevance_score,
                    'combined_score': authority.authority_score * 0.7 + relevance_score * 0.3,
                    'legal_area': metadata.legal_area,
                    'year': getattr(metadata, 'year', None),
                    'court': getattr(metadata, 'court', 'Unknown')
                }
                results.append(result)
        
        # Sort by combined score
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return results[:max_results]
    
    def _calculate_query_relevance(
        self, 
        query_terms: List[str], 
        title: str, 
        keywords: List[str]
    ) -> float:
        """Calculate relevance score for query terms"""
        
        title_lower = title.lower()
        keywords_lower = [kw.lower() for kw in keywords]
        query_lower = [term.lower() for term in query_terms]
        
        score = 0.0
        
        # Title matches (higher weight)
        for term in query_lower:
            if term in title_lower:
                score += 0.3
        
        # Keyword matches
        for term in query_lower:
            for keyword in keywords_lower:
                if term in keyword or keyword in term:
                    score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
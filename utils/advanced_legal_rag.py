#!/usr/bin/env python3
"""
Advanced Legal RAG Pipeline
Production-ready legal RAG system integrating citation graphs, authority-aware retrieval,
temporal filtering, and professional compliance standards.
"""

import json
import logging
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict

from .legal_metadata import LegalMetadata, LegalJurisdiction
from .hmrc_metadata import HMRCMetadata, TaxDomain
from .legal_citation_graph import LegalCitationGraph, PrecedentAuthority
from .legal_faiss_index import LegalFAISSIndex, LegalEmbeddingModel, LegalSearchResult
from .legal_compliance import LegalComplianceFramework, LegalValidationResult
from .legal_training import LegalPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalQuery:
    """Structured legal query with context"""
    query_text: str
    legal_area: Optional[str] = None
    jurisdiction: Optional[str] = None
    case_date: Optional[datetime] = None
    query_type: str = "general_analysis"  # general_analysis, counter_argument, compliance_advice
    client_context: str = ""
    opposing_arguments: List[str] = None
    specific_authorities: List[str] = None
    urgency_level: str = "normal"  # urgent, normal, research
    confidence_required: str = "medium"  # high, medium, low

@dataclass
class LegalRAGResult:
    """Comprehensive legal RAG result"""
    query: LegalQuery
    response: str
    retrieved_authorities: List[LegalSearchResult]
    citation_analysis: Dict
    confidence_assessment: Dict
    compliance_validation: LegalValidationResult
    reasoning_chain: List[str]
    alternative_interpretations: List[str]
    procedural_considerations: List[str]
    risk_assessment: Dict
    generated_at: datetime
    processing_time: float

class AdvancedLegalRetriever:
    """
    Advanced legal retrieval with temporal, authority, and context awareness
    """
    
    def __init__(self, faiss_index: LegalFAISSIndex, citation_graph: Optional[LegalCitationGraph] = None):
        self.faiss_index = faiss_index
        self.citation_graph = citation_graph
        
        # Set citation graph in FAISS index if provided
        if citation_graph:
            self.faiss_index.set_citation_graph(citation_graph)
        
        # Authority weights for different scenarios
        self.scenario_weights = {
            'litigation_urgent': {'authority': 1.5, 'temporal': 1.2, 'citation': 1.3},
            'compliance_advice': {'authority': 1.2, 'temporal': 1.4, 'citation': 1.0},
            'academic_research': {'authority': 1.0, 'temporal': 0.8, 'citation': 1.4},
            'general_analysis': {'authority': 1.1, 'temporal': 1.0, 'citation': 1.1}
        }
    
    def retrieve_with_legal_context(
        self,
        query: LegalQuery,
        k: int = 10,
        authority_threshold: float = 0.0,
        include_related_precedents: bool = True
    ) -> List[LegalSearchResult]:
        """Advanced retrieval with comprehensive legal context"""
        
        # Determine retrieval strategy based on query type
        scenario_key = f"{query.query_type}_{query.urgency_level}" if f"{query.query_type}_{query.urgency_level}" in self.scenario_weights else "general_analysis"
        weights = self.scenario_weights[scenario_key]
        
        # Primary retrieval
        primary_results = self.faiss_index.search(
            query=query.query_text,
            k=k * 2,  # Get more candidates for filtering
            legal_area=query.legal_area,
            jurisdiction=query.jurisdiction,
            authority_threshold=authority_threshold,
            temporal_filter=query.case_date,
            boost_authority=True,
            boost_citations=True
        )
        
        # Apply scenario-specific re-ranking
        for result in primary_results:
            result.final_score = (
                result.score * 
                (result.authority_boost * weights['authority']) *
                (result.temporal_boost * weights['temporal']) *
                (result.citation_boost * weights['citation'])
            )
        
        # Sort by enhanced score
        primary_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Get top k results
        top_results = primary_results[:k]
        
        # Enhance with related precedents if requested
        if include_related_precedents and self.citation_graph:
            related_results = self._find_related_precedents(top_results, query)
            
            # Merge and re-rank
            all_results = self._merge_and_deduplicate(top_results, related_results)
            return all_results[:k]
        
        return top_results
    
    def _find_related_precedents(self, primary_results: List[LegalSearchResult], query: LegalQuery) -> List[LegalSearchResult]:
        """Find precedents related to primary results through citation graph"""
        related_docs = set()
        
        for result in primary_results[:5]:  # Only use top 5 for finding related
            doc_id = result.document_id
            
            if doc_id in self.citation_graph.document_metadata:
                # Find citing and cited documents
                citing_docs = list(self.citation_graph.graph.predecessors(doc_id))
                cited_docs = list(self.citation_graph.graph.successors(doc_id))
                
                # Add highly relevant related documents
                for related_doc in citing_docs + cited_docs:
                    authority = self.citation_graph.authority_scores.get(related_doc)
                    if authority and authority.authority_score > 0.6:
                        related_docs.add(related_doc)
        
        # Retrieve related documents
        related_results = []
        for doc_id in list(related_docs)[:5]:  # Limit to 5 related docs
            # Find chunks for this document in FAISS index
            doc_chunks = [chunk for chunk in self.faiss_index.chunks.values() 
                         if chunk.document_id == doc_id]
            
            if doc_chunks:
                # Use the best chunk for this document
                best_chunk = max(doc_chunks, key=lambda x: x.authority_weight)
                
                # Create a search result for this related document
                related_result = LegalSearchResult(
                    chunk_id=best_chunk.chunk_id,
                    document_id=doc_id,
                    score=0.7,  # Base score for related documents
                    authority_boost=best_chunk.authority_weight,
                    temporal_boost=1.0,
                    citation_boost=1.2,  # Boost for citation relationship
                    final_score=0.7 * best_chunk.authority_weight * 1.2,
                    text=best_chunk.text,
                    metadata=best_chunk.legal_metadata,
                    context=best_chunk.section_context or "",
                    explanation="Related through citation network"
                )
                related_results.append(related_result)
        
        return related_results
    
    def _merge_and_deduplicate(self, primary: List[LegalSearchResult], related: List[LegalSearchResult]) -> List[LegalSearchResult]:
        """Merge and deduplicate search results"""
        seen_docs = set()
        merged_results = []
        
        # Add primary results first
        for result in primary:
            if result.document_id not in seen_docs:
                merged_results.append(result)
                seen_docs.add(result.document_id)
        
        # Add related results that aren't duplicates
        for result in related:
            if result.document_id not in seen_docs:
                merged_results.append(result)
                seen_docs.add(result.document_id)
        
        # Sort by final score
        merged_results.sort(key=lambda x: x.final_score, reverse=True)
        return merged_results

class LegalPromptConstructor:
    """
    Constructs professional legal prompts with proper authority citations
    """
    
    def __init__(self):
        self.prompt_template = LegalPromptTemplate()
    
    def construct_legal_prompt(
        self,
        query: LegalQuery,
        retrieved_authorities: List[LegalSearchResult],
        prompt_type: str = None
    ) -> str:
        """Construct legally-aware prompt with proper structure"""
        
        # Determine prompt type from query if not specified
        if not prompt_type:
            prompt_type = self._determine_prompt_type(query)
        
        # Format authorities with confidence and relevance indicators
        formatted_authorities = self._format_authorities_with_confidence(retrieved_authorities)
        
        # Build context section
        context_section = self._build_context_section(query)
        
        # Construct the complete prompt
        if prompt_type == "counter_argument":
            opposing_args = "\n".join(query.opposing_arguments) if query.opposing_arguments else "No specific opposing arguments provided."
            
            prompt = self.prompt_template.create_training_prompt(
                template_type="counter_argument",
                input_context=context_section,
                instruction=query.query_text,
                authorities=formatted_authorities,
                opposing_argument=opposing_args
            )
        elif prompt_type == "tax_compliance":
            prompt = self.prompt_template.create_training_prompt(
                template_type="tax_compliance",
                input_context=context_section,
                instruction=query.query_text,
                authorities=formatted_authorities
            )
        else:
            prompt = self.prompt_template.create_training_prompt(
                template_type="legal_analysis",
                input_context=context_section,
                instruction=query.query_text,
                authorities=formatted_authorities
            )
        
        return prompt
    
    def _determine_prompt_type(self, query: LegalQuery) -> str:
        """Determine the appropriate prompt type from query"""
        query_lower = query.query_text.lower()
        
        if query.query_type == "counter_argument" or any(term in query_lower for term in ['counter', 'defense', 'argue against']):
            return "counter_argument"
        elif query.legal_area in ['tax', 'vat', 'hmrc'] or any(term in query_lower for term in ['tax', 'hmrc', 'vat', 'duty']):
            return "tax_compliance"
        else:
            return "legal_analysis"
    
    def _format_authorities_with_confidence(self, authorities: List[LegalSearchResult]) -> List[Dict]:
        """Format authorities with confidence and relevance indicators"""
        formatted_authorities = []
        
        for i, authority in enumerate(authorities):
            metadata = authority.metadata
            
            # Build authority entry
            auth_entry = {
                'citation': getattr(metadata, 'citation', metadata.title),
                'title': metadata.title,
                'authority_level': getattr(metadata, 'authority_level', 'Unknown'),
                'relevance_score': f"{authority.final_score:.3f}",
                'text': authority.text[:500] + "..." if len(authority.text) > 500 else authority.text,
                'explanation': authority.explanation,
                'jurisdiction': getattr(metadata, 'jurisdiction', 'Unknown'),
                'legal_area': getattr(metadata, 'legal_area', getattr(metadata, 'tax_domain', 'General'))
            }
            
            # Add confidence indicators
            if authority.final_score > 0.8:
                auth_entry['confidence'] = "High"
            elif authority.final_score > 0.6:
                auth_entry['confidence'] = "Medium"
            else:
                auth_entry['confidence'] = "Low"
            
            formatted_authorities.append(auth_entry)
        
        return formatted_authorities
    
    def _build_context_section(self, query: LegalQuery) -> str:
        """Build comprehensive context section"""
        context_parts = []
        
        if query.client_context:
            context_parts.append(f"Client Context: {query.client_context}")
        
        if query.legal_area:
            context_parts.append(f"Legal Area: {query.legal_area}")
        
        if query.jurisdiction:
            context_parts.append(f"Jurisdiction: {query.jurisdiction}")
        
        if query.case_date:
            context_parts.append(f"Reference Date: {query.case_date.strftime('%Y-%m-%d')}")
        
        if query.urgency_level != "normal":
            context_parts.append(f"Urgency: {query.urgency_level}")
        
        return "\n".join(context_parts) if context_parts else "General legal inquiry"

class LegalReasoningEngine:
    """
    Advanced legal reasoning with structured analysis
    """
    
    def __init__(self):
        self.reasoning_templates = {
            'legal_analysis': [
                "1. Identify the key legal issues",
                "2. Determine the applicable legal framework",
                "3. Analyze relevant authorities and precedents", 
                "4. Apply the law to the specific facts",
                "5. Consider alternative interpretations",
                "6. Assess the strength of different arguments",
                "7. Reach a reasoned conclusion"
            ],
            'counter_argument': [
                "1. Analyze the opposing position",
                "2. Identify weaknesses in the opposing argument",
                "3. Find alternative legal interpretations",
                "4. Locate contradicting authorities",
                "5. Develop procedural challenges",
                "6. Assess evidentiary issues",
                "7. Construct the counter-argument strategy"
            ],
            'compliance_analysis': [
                "1. Identify applicable regulatory requirements",
                "2. Assess current compliance status",
                "3. Identify areas of non-compliance risk",
                "4. Evaluate mitigation strategies",
                "5. Consider enforcement implications",
                "6. Develop compliance recommendations",
                "7. Create monitoring and review procedures"
            ]
        }
    
    def generate_reasoning_chain(self, query: LegalQuery, authorities: List[LegalSearchResult]) -> List[str]:
        """Generate structured reasoning chain"""
        reasoning_type = self._determine_reasoning_type(query)
        template = self.reasoning_templates.get(reasoning_type, self.reasoning_templates['legal_analysis'])
        
        # Customize reasoning steps based on specific context
        customized_steps = []
        for step in template:
            customized_step = self._customize_reasoning_step(step, query, authorities)
            customized_steps.append(customized_step)
        
        return customized_steps
    
    def _determine_reasoning_type(self, query: LegalQuery) -> str:
        """Determine the type of reasoning required"""
        if query.query_type == "counter_argument":
            return "counter_argument"
        elif query.query_type == "compliance_advice":
            return "compliance_analysis"
        else:
            return "legal_analysis"
    
    def _customize_reasoning_step(self, step: str, query: LegalQuery, authorities: List[LegalSearchResult]) -> str:
        """Customize reasoning step based on specific context"""
        # Add specific context to generic reasoning steps
        if "legal issues" in step and query.legal_area:
            step += f" (focus on {query.legal_area} law)"
        
        if "applicable legal framework" in step and authorities:
            primary_authority = authorities[0] if authorities else None
            if primary_authority:
                step += f" (primary authority: {primary_authority.metadata.title})"
        
        if "jurisdiction" in query.jurisdiction and "jurisdiction" not in step:
            step += f" (considering {query.jurisdiction} jurisdiction)"
        
        return step

class AdvancedLegalRAGPipeline:
    """
    Complete advanced legal RAG pipeline integrating all components
    """
    
    def __init__(
        self,
        faiss_index: LegalFAISSIndex,
        citation_graph: Optional[LegalCitationGraph] = None,
        compliance_framework: Optional[LegalComplianceFramework] = None
    ):
        self.faiss_index = faiss_index
        self.citation_graph = citation_graph
        self.compliance_framework = compliance_framework or LegalComplianceFramework()
        
        # Initialize components
        self.retriever = AdvancedLegalRetriever(faiss_index, citation_graph)
        self.prompt_constructor = LegalPromptConstructor()
        self.reasoning_engine = LegalReasoningEngine()
        
        # Performance tracking
        self.query_history = []
        self.performance_metrics = defaultdict(list)
    
    def process_legal_query(
        self,
        query: LegalQuery,
        llm_generate_func: callable,
        max_retries: int = 2
    ) -> LegalRAGResult:
        """Process a complete legal query with RAG pipeline"""
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant authorities
            logger.info(f"Retrieving authorities for query: {query.query_text[:100]}...")
            retrieved_authorities = self.retriever.retrieve_with_legal_context(
                query=query,
                k=10,
                authority_threshold=0.1,
                include_related_precedents=True
            )
            
            if not retrieved_authorities:
                logger.warning("No authorities retrieved for query")
                retrieved_authorities = []
            
            # Step 2: Analyze citations and precedents
            citation_analysis = self._analyze_citations(retrieved_authorities)
            
            # Step 3: Generate reasoning chain
            reasoning_chain = self.reasoning_engine.generate_reasoning_chain(query, retrieved_authorities)
            
            # Step 4: Construct legal prompt
            prompt = self.prompt_constructor.construct_legal_prompt(
                query=query,
                retrieved_authorities=retrieved_authorities
            )
            
            # Step 5: Generate response with retries
            response = None
            attempts = 0
            
            while attempts < max_retries and not response:
                try:
                    response = llm_generate_func(prompt)
                    break
                except Exception as e:
                    attempts += 1
                    logger.warning(f"LLM generation attempt {attempts} failed: {e}")
                    if attempts >= max_retries:
                        response = self._generate_fallback_response(query, retrieved_authorities)
            
            # Step 6: Validate compliance
            compliance_validation = self.compliance_framework.validate_legal_response(
                response=response,
                response_type=query.query_type,
                legal_area=query.legal_area,
                source_metadata=retrieved_authorities[0].metadata if retrieved_authorities else None
            )
            
            # Step 7: Generate additional analysis
            alternative_interpretations = self._generate_alternative_interpretations(query, retrieved_authorities)
            procedural_considerations = self._identify_procedural_considerations(query, retrieved_authorities)
            risk_assessment = self._assess_risks(query, retrieved_authorities, response)
            
            # Step 8: Assess confidence
            confidence_assessment = self._assess_confidence(
                query, retrieved_authorities, response, compliance_validation
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Create result
            result = LegalRAGResult(
                query=query,
                response=response,
                retrieved_authorities=retrieved_authorities,
                citation_analysis=citation_analysis,
                confidence_assessment=confidence_assessment,
                compliance_validation=compliance_validation,
                reasoning_chain=reasoning_chain,
                alternative_interpretations=alternative_interpretations,
                procedural_considerations=procedural_considerations,
                risk_assessment=risk_assessment,
                generated_at=datetime.now(),
                processing_time=processing_time
            )
            
            # Track performance
            self.query_history.append(result)
            self.performance_metrics['processing_time'].append(processing_time)
            self.performance_metrics['authorities_retrieved'].append(len(retrieved_authorities))
            self.performance_metrics['compliance_score'].append(compliance_validation.overall_score)
            
            logger.info(f"Legal query processed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Legal RAG pipeline failed: {e}")
            raise
    
    def _analyze_citations(self, authorities: List[LegalSearchResult]) -> Dict:
        """Analyze citation patterns and authority relationships"""
        analysis = {
            'total_authorities': len(authorities),
            'authority_levels': defaultdict(int),
            'jurisdictions': defaultdict(int),
            'legal_areas': defaultdict(int),
            'citation_relationships': [],
            'landmark_cases': []
        }
        
        for authority in authorities:
            metadata = authority.metadata
            
            # Count authority levels
            auth_level = getattr(metadata, 'authority_level', 'Unknown')
            analysis['authority_levels'][str(auth_level)] += 1
            
            # Count jurisdictions
            jurisdiction = getattr(metadata, 'jurisdiction', 'Unknown')
            jurisdiction_value = jurisdiction.value if hasattr(jurisdiction, 'value') else str(jurisdiction)
            analysis['jurisdictions'][jurisdiction_value] += 1
            
            # Count legal areas
            legal_area = getattr(metadata, 'legal_area', getattr(metadata, 'tax_domain', 'Unknown'))
            area_value = legal_area.value if hasattr(legal_area, 'value') else str(legal_area)
            analysis['legal_areas'][area_value] += 1
            
            # Identify landmark cases (high authority + high final score)
            if authority.final_score > 0.8 and authority.authority_boost > 1.2:
                analysis['landmark_cases'].append({
                    'title': metadata.title,
                    'citation': getattr(metadata, 'citation', ''),
                    'authority_score': authority.final_score
                })
        
        return analysis
    
    def _generate_alternative_interpretations(self, query: LegalQuery, authorities: List[LegalSearchResult]) -> List[str]:
        """Generate alternative legal interpretations"""
        interpretations = []
        
        # Based on different authorities or legal approaches
        if len(authorities) > 1:
            interpretations.append("Alternative interpretation based on secondary authorities")
        
        if query.jurisdiction:
            interpretations.append(f"Interpretation under {query.jurisdiction} specific provisions")
        
        # Add generic alternative considerations
        interpretations.extend([
            "Consider alternative legal theories or approaches",
            "Evaluate potential distinguishing factors",
            "Assess impact of recent legal developments"
        ])
        
        return interpretations[:3]  # Limit to 3 alternatives
    
    def _identify_procedural_considerations(self, query: LegalQuery, authorities: List[LegalSearchResult]) -> List[str]:
        """Identify procedural considerations"""
        considerations = []
        
        if query.urgency_level == "urgent":
            considerations.append("Consider time-sensitive procedural requirements")
        
        if query.query_type == "counter_argument":
            considerations.extend([
                "Review disclosure obligations and deadlines",
                "Consider application for case management directions",
                "Assess potential for interim relief applications"
            ])
        
        # Add general procedural considerations
        considerations.extend([
            "Review applicable court rules and practice directions",
            "Consider limitation periods and procedural deadlines",
            "Assess evidence gathering requirements"
        ])
        
        return considerations[:5]  # Limit to 5 considerations
    
    def _assess_risks(self, query: LegalQuery, authorities: List[LegalSearchResult], response: str) -> Dict:
        """Assess legal and practical risks"""
        risk_assessment = {
            'legal_risks': [],
            'procedural_risks': [],
            'practical_risks': [],
            'overall_risk_level': 'medium'
        }
        
        # Assess based on authority strength
        if not authorities or all(auth.final_score < 0.6 for auth in authorities):
            risk_assessment['legal_risks'].append("Weak or insufficient legal authority")
        
        # Assess based on query urgency
        if query.urgency_level == "urgent":
            risk_assessment['procedural_risks'].append("Time pressure may limit thorough analysis")
        
        # Assess based on confidence level
        if query.confidence_required == "high" and authorities:
            avg_score = sum(auth.final_score for auth in authorities) / len(authorities)
            if avg_score < 0.7:
                risk_assessment['practical_risks'].append("May not meet high confidence threshold")
        
        # Determine overall risk level
        total_risks = len(risk_assessment['legal_risks']) + len(risk_assessment['procedural_risks']) + len(risk_assessment['practical_risks'])
        if total_risks > 3:
            risk_assessment['overall_risk_level'] = 'high'
        elif total_risks == 0:
            risk_assessment['overall_risk_level'] = 'low'
        
        return risk_assessment
    
    def _assess_confidence(
        self,
        query: LegalQuery,
        authorities: List[LegalSearchResult],
        response: str,
        compliance: LegalValidationResult
    ) -> Dict:
        """Assess confidence in the legal analysis"""
        confidence_factors = {
            'authority_strength': 0.0,
            'citation_accuracy': 0.0,
            'compliance_score': compliance.overall_score,
            'response_completeness': 0.0,
            'jurisdictional_alignment': 0.0
        }
        
        # Authority strength
        if authorities:
            avg_authority_score = sum(auth.final_score for auth in authorities) / len(authorities)
            confidence_factors['authority_strength'] = avg_authority_score
        
        # Citation accuracy
        confidence_factors['citation_accuracy'] = compliance.citation_accuracy
        
        # Response completeness (basic length and structure check)
        if len(response) > 500 and any(term in response.lower() for term in ['analysis', 'conclusion', 'recommendation']):
            confidence_factors['response_completeness'] = 0.8
        elif len(response) > 200:
            confidence_factors['response_completeness'] = 0.6
        else:
            confidence_factors['response_completeness'] = 0.3
        
        # Jurisdictional alignment
        if query.jurisdiction:
            jurisdiction_match = any(
                query.jurisdiction.lower() in str(auth.metadata.jurisdiction).lower()
                for auth in authorities
            )
            confidence_factors['jurisdictional_alignment'] = 1.0 if jurisdiction_match else 0.5
        else:
            confidence_factors['jurisdictional_alignment'] = 0.8
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors.values()) / len(confidence_factors)
        
        return {
            'factors': confidence_factors,
            'overall_confidence': overall_confidence,
            'confidence_level': 'high' if overall_confidence > 0.8 else 'medium' if overall_confidence > 0.6 else 'low'
        }
    
    def _generate_fallback_response(self, query: LegalQuery, authorities: List[LegalSearchResult]) -> str:
        """Generate fallback response if LLM generation fails"""
        fallback = f"""Based on the available legal authorities, this matter requires careful consideration of the following:

**Legal Framework:**
{authorities[0].metadata.title if authorities else 'Relevant legal principles'}

**Key Considerations:**
- {query.query_text}
- Applicable jurisdiction: {query.jurisdiction or 'To be determined'}
- Legal area: {query.legal_area or 'General'}

**Important Notice:**
This analysis could not be completed due to technical issues. Please consult with a qualified legal professional for comprehensive advice specific to your circumstances.

**Disclaimer:**
This response is generated by an AI system and should not be considered as formal legal advice. Professional legal consultation is recommended."""
        
        return fallback
    
    def get_performance_statistics(self) -> Dict:
        """Get pipeline performance statistics"""
        if not self.query_history:
            return {'status': 'no_queries_processed'}
        
        stats = {
            'total_queries': len(self.query_history),
            'average_processing_time': np.mean(self.performance_metrics['processing_time']),
            'average_authorities_retrieved': np.mean(self.performance_metrics['authorities_retrieved']),
            'average_compliance_score': np.mean(self.performance_metrics['compliance_score']),
            'query_types': defaultdict(int),
            'legal_areas': defaultdict(int),
            'success_rate': len([q for q in self.query_history if q.compliance_validation.overall_score > 0.6]) / len(self.query_history)
        }
        
        # Analyze query patterns
        for query_result in self.query_history:
            stats['query_types'][query_result.query.query_type] += 1
            if query_result.query.legal_area:
                stats['legal_areas'][query_result.query.legal_area] += 1
        
        return stats

def create_legal_rag_system(
    legal_documents: List[Tuple[str, Union[LegalMetadata, HMRCMetadata], str]],
    embedding_model_name: str = "nlpaueb/legal-bert-base-uncased",
    index_type: str = "IVF"
) -> AdvancedLegalRAGPipeline:
    """Create a complete legal RAG system from documents"""
    
    logger.info("Creating advanced legal RAG system...")
    
    # Initialize embedding model
    embedding_model = LegalEmbeddingModel(embedding_model_name)
    
    # Create FAISS index
    faiss_index = LegalFAISSIndex(embedding_model, index_type)
    
    # Add documents to index
    faiss_index.add_documents(legal_documents, chunk_size=1000, chunk_overlap=150)
    
    # Create citation graph
    citation_graph = LegalCitationGraph()
    
    # Add documents to citation graph
    for doc_id, metadata, content in legal_documents:
        citation_graph.add_document(doc_id, metadata)
        citation_graph.extract_citation_relationships(doc_id, content)
    
    # Build citation network
    citation_graph.build_citation_network()
    citation_graph.calculate_authority_scores()
    
    # Create compliance framework
    compliance_framework = LegalComplianceFramework()
    
    # Create RAG pipeline
    rag_pipeline = AdvancedLegalRAGPipeline(
        faiss_index=faiss_index,
        citation_graph=citation_graph,
        compliance_framework=compliance_framework
    )
    
    logger.info(f"Legal RAG system created with {len(legal_documents)} documents")
    
    return rag_pipeline
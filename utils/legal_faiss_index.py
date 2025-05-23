#!/usr/bin/env python3
"""
Legal FAISS Index with Authority-Aware Retrieval
Production-ready legal document retrieval system with hierarchical authority,
temporal filtering, and citation graph integration.
"""

import json
import logging
import pickle
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
import hashlib

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")
    SentenceTransformer = None

from .legal_metadata import LegalMetadata, LegalJurisdiction, LegalSourceType, CourtHierarchy
from .hmrc_metadata import HMRCMetadata, TaxAuthority, TaxDomain
from .legal_citation_graph import LegalCitationGraph, PrecedentAuthority

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LegalSearchResult:
    """Enhanced search result with legal context"""
    chunk_id: str
    document_id: str
    score: float
    authority_boost: float
    temporal_boost: float
    citation_boost: float
    final_score: float
    text: str
    metadata: Union[LegalMetadata, HMRCMetadata]
    context: str = ""
    explanation: str = ""

@dataclass
class LegalChunk:
    """Legal document chunk with enhanced metadata"""
    chunk_id: str
    document_id: str
    text: str
    section_context: Optional[str]
    chunk_index: int
    start_char: int
    end_char: int
    legal_metadata: Union[LegalMetadata, HMRCMetadata]
    chunk_type: str = "general"  # section, subsection, paragraph, etc.
    authority_weight: float = 0.5
    embedding: Optional[np.ndarray] = None

class LegalEmbeddingModel:
    """
    Legal domain-specific embedding model with fallback options
    """
    
    def __init__(self, model_name: str = None):
        self.model = None
        self.model_name = model_name
        self.dimension = 768  # Default dimension
        
        # Try to load legal-specific models first, then fallback
        model_options = [
            model_name,
            "nlpaueb/legal-bert-base-uncased",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L6-v2"
        ]
        
        for model_option in model_options:
            if model_option and self._try_load_model(model_option):
                break
        
        if self.model is None:
            logger.error("No suitable embedding model available")
            raise RuntimeError("Failed to load any embedding model")
    
    def _try_load_model(self, model_name: str) -> bool:
        """Try to load a specific model"""
        if SentenceTransformer is None:
            return False
        
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Loaded embedding model: {model_name} (dim={self.dimension})")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model {model_name}: {e}")
            return False
    
    def encode_with_legal_context(self, texts: List[str], legal_contexts: List[Dict]) -> np.ndarray:
        """Encode texts with legal context enhancement"""
        if not self.model:
            raise RuntimeError("No embedding model available")
        
        enhanced_texts = []
        for text, context in zip(texts, legal_contexts):
            # Enhance text with legal context
            context_prefix = self._build_legal_context_prefix(context)
            enhanced_text = f"{context_prefix} {text}"
            enhanced_texts.append(enhanced_text)
        
        return self.model.encode(enhanced_texts, convert_to_numpy=True, show_progress_bar=True)
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Standard encoding without legal context"""
        if not self.model:
            raise RuntimeError("No embedding model available")
        
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    def _build_legal_context_prefix(self, context: Dict) -> str:
        """Build context prefix for legal documents"""
        prefixes = []
        
        # Add jurisdiction context
        if 'jurisdiction' in context:
            prefixes.append(f"[{context['jurisdiction']}]")
        
        # Add legal area context
        if 'legal_area' in context:
            prefixes.append(f"[{context['legal_area']}]")
        elif 'tax_domain' in context:
            prefixes.append(f"[{context['tax_domain']}]")
        
        # Add authority context
        if 'authority_level' in context:
            prefixes.append(f"[authority:{context['authority_level']}]")
        
        # Add section context if available
        if 'section_context' in context and context['section_context']:
            prefixes.append(f"[{context['section_context']}]")
        
        return " ".join(prefixes)

class LegalFAISSIndex:
    """
    Production-ready FAISS index for legal document retrieval with authority awareness
    """
    
    def __init__(self, embedding_model: LegalEmbeddingModel, index_type: str = "IVF"):
        self.embedding_model = embedding_model
        self.dimension = embedding_model.dimension
        
        # Initialize FAISS index
        if index_type == "IVF":
            # IVF index for large document collections
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            # Flat index for smaller collections or exact search
            self.index = faiss.IndexFlatIP(self.dimension)
        
        # Metadata storage
        self.chunks: Dict[str, LegalChunk] = {}
        self.chunk_to_index: Dict[str, int] = {}
        self.index_to_chunk: Dict[int, str] = {}
        self.document_metadata: Dict[str, Union[LegalMetadata, HMRCMetadata]] = {}
        
        # Citation graph integration
        self.citation_graph: Optional[LegalCitationGraph] = None
        
        # Index statistics
        self.total_chunks = 0
        self.documents_indexed = 0
        
        # Authority weights mapping
        self.authority_weights = self._initialize_authority_weights()
    
    def _initialize_authority_weights(self) -> Dict:
        """Initialize authority weights for different document types"""
        weights = {}
        
        # Legal document authority weights
        for hierarchy in CourtHierarchy:
            weights[f"legal_{hierarchy.name}"] = 1.0 / hierarchy.value
        
        # Tax document authority weights  
        for authority in TaxAuthority:
            weights[f"tax_{authority.name}"] = 1.0 / authority.value
        
        return weights
    
    def add_documents(
        self, 
        documents: List[Tuple[str, str, Union[LegalMetadata, HMRCMetadata]]], 
        chunk_size: int = 1000,
        chunk_overlap: int = 150
    ) -> None:
        """Add legal documents to the index with intelligent chunking"""
        logger.info(f"Adding {len(documents)} documents to legal index...")
        
        all_chunks = []
        all_embeddings = []
        
        for doc_id, text, metadata in documents:
            # Store document metadata
            self.document_metadata[doc_id] = metadata
            
            # Create legal-aware chunks
            chunks = self._create_legal_chunks(doc_id, text, metadata, chunk_size, chunk_overlap)
            
            # Generate embeddings with legal context
            chunk_texts = [chunk.text for chunk in chunks]
            legal_contexts = [self._extract_legal_context(chunk) for chunk in chunks]
            
            if chunk_texts:
                embeddings = self.embedding_model.encode_with_legal_context(chunk_texts, legal_contexts)
                
                for chunk, embedding in zip(chunks, embeddings):
                    chunk.embedding = embedding
                    all_chunks.append(chunk)
                    all_embeddings.append(embedding)
        
        if all_embeddings:
            # Normalize embeddings for cosine similarity
            embeddings_array = np.array(all_embeddings).astype('float32')
            faiss.normalize_L2(embeddings_array)
            
            # Train index if necessary (for IVF)
            if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                logger.info("Training FAISS index...")
                self.index.train(embeddings_array)
            
            # Add embeddings to index
            start_idx = self.total_chunks
            self.index.add(embeddings_array)
            
            # Update mappings
            for i, chunk in enumerate(all_chunks):
                index_pos = start_idx + i
                self.chunks[chunk.chunk_id] = chunk
                self.chunk_to_index[chunk.chunk_id] = index_pos
                self.index_to_chunk[index_pos] = chunk.chunk_id
            
            self.total_chunks += len(all_chunks)
            self.documents_indexed += len(documents)
        
        logger.info(f"Added {len(all_chunks)} chunks from {len(documents)} documents")
    
    def _create_legal_chunks(
        self,
        doc_id: str,
        text: str,
        metadata: Union[LegalMetadata, HMRCMetadata],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[LegalChunk]:
        """Create legal-aware chunks preserving document structure"""
        chunks = []
        
        # Try to split by legal structure first
        sections = self._split_by_legal_structure(text)
        
        if len(sections) > 1:
            # Document has clear legal structure
            for section_idx, (section_title, section_text) in enumerate(sections):
                section_chunks = self._chunk_section(
                    doc_id, section_text, section_title, metadata, 
                    section_idx, chunk_size, chunk_overlap
                )
                chunks.extend(section_chunks)
        else:
            # Fall back to standard chunking
            standard_chunks = self._create_standard_chunks(
                doc_id, text, metadata, chunk_size, chunk_overlap
            )
            chunks.extend(standard_chunks)
        
        return chunks
    
    def _split_by_legal_structure(self, text: str) -> List[Tuple[str, str]]:
        """Split text by legal structure (sections, subsections, etc.)"""
        import re
        
        # Patterns for legal structure
        section_patterns = [
            r'(?:^|\n)\s*(?:Section|SECTION)\s+(\d+(?:\.\d+)*)\s*[:\-]?\s*(.*?)(?=\n(?:Section|SECTION)|\Z)',
            r'(?:^|\n)\s*(\d+\.)\s+(.*?)(?=\n\s*\d+\.|\Z)',
            r'(?:^|\n)\s*\((\d+)\)\s*(.*?)(?=\n\s*\(\d+\)|\Z)'
        ]
        
        sections = []
        
        for pattern in section_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.DOTALL)
            if len(matches) > 2:  # Need at least 3 sections to consider structured
                sections = [(f"Section {match[0]}", match[1].strip()) for match in matches]
                break
        
        if not sections:
            # Try paragraph splitting
            paragraphs = text.split('\n\n')
            if len(paragraphs) > 3:
                sections = [(f"Paragraph {i+1}", para.strip()) for i, para in enumerate(paragraphs) if para.strip()]
        
        return sections or [("Full Document", text)]
    
    def _chunk_section(
        self,
        doc_id: str,
        section_text: str,
        section_title: str,
        metadata: Union[LegalMetadata, HMRCMetadata],
        section_idx: int,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[LegalChunk]:
        """Chunk a legal section while preserving context"""
        chunks = []
        
        # Calculate authority weight
        authority_weight = self._calculate_authority_weight(metadata)
        
        if len(section_text) <= chunk_size:
            # Section fits in one chunk
            chunk_id = f"{doc_id}_s{section_idx}_c0"
            chunk = LegalChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                text=section_text,
                section_context=section_title,
                chunk_index=0,
                start_char=0,
                end_char=len(section_text),
                legal_metadata=metadata,
                chunk_type="section",
                authority_weight=authority_weight
            )
            chunks.append(chunk)
        else:
            # Split section into multiple chunks
            words = section_text.split()
            words_per_chunk = chunk_size // 5  # Approximate words per chunk
            overlap_words = chunk_overlap // 5
            
            start_word = 0
            chunk_idx = 0
            
            while start_word < len(words):
                end_word = min(len(words), start_word + words_per_chunk)
                chunk_words = words[start_word:end_word]
                chunk_text = ' '.join(chunk_words)
                
                chunk_id = f"{doc_id}_s{section_idx}_c{chunk_idx}"
                chunk = LegalChunk(
                    chunk_id=chunk_id,
                    document_id=doc_id,
                    text=chunk_text,
                    section_context=section_title,
                    chunk_index=chunk_idx,
                    start_char=0,  # Would need more precise calculation
                    end_char=len(chunk_text),
                    legal_metadata=metadata,
                    chunk_type="section_chunk",
                    authority_weight=authority_weight
                )
                chunks.append(chunk)
                
                if end_word >= len(words):
                    break
                
                start_word = end_word - overlap_words
                chunk_idx += 1
        
        return chunks
    
    def _create_standard_chunks(
        self,
        doc_id: str,
        text: str,
        metadata: Union[LegalMetadata, HMRCMetadata],
        chunk_size: int,
        chunk_overlap: int
    ) -> List[LegalChunk]:
        """Create standard text chunks as fallback"""
        chunks = []
        authority_weight = self._calculate_authority_weight(metadata)
        
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(len(text), start + chunk_size)
            chunk_text = text[start:end]
            
            chunk_id = f"{doc_id}_c{chunk_idx}"
            chunk = LegalChunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                text=chunk_text,
                section_context=None,
                chunk_index=chunk_idx,
                start_char=start,
                end_char=end,
                legal_metadata=metadata,
                chunk_type="general",
                authority_weight=authority_weight
            )
            chunks.append(chunk)
            
            if end >= len(text):
                break
            
            start = end - chunk_overlap
            chunk_idx += 1
        
        return chunks
    
    def _calculate_authority_weight(self, metadata: Union[LegalMetadata, HMRCMetadata]) -> float:
        """Calculate authority weight for a document"""
        if isinstance(metadata, LegalMetadata):
            # Legal document authority
            return 1.0 / metadata.authority_level if metadata.authority_level else 0.1
        elif isinstance(metadata, HMRCMetadata):
            # HMRC document authority
            return 1.0 / metadata.authority_level.value
        else:
            return 0.5  # Default weight
    
    def _extract_legal_context(self, chunk: LegalChunk) -> Dict:
        """Extract legal context for embedding enhancement"""
        metadata = chunk.legal_metadata
        context = {}
        
        if isinstance(metadata, LegalMetadata):
            context['jurisdiction'] = metadata.jurisdiction.value
            context['legal_area'] = metadata.legal_area
            context['authority_level'] = metadata.authority_level
        elif isinstance(metadata, HMRCMetadata):
            context['jurisdiction'] = 'uk_wide'  # HMRC applies UK-wide
            context['tax_domain'] = metadata.tax_domain.value
            context['authority_level'] = metadata.authority_level.value
        
        if chunk.section_context:
            context['section_context'] = chunk.section_context
        
        return context
    
    def search(
        self,
        query: str,
        k: int = 10,
        legal_area: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        authority_threshold: float = 0.0,
        temporal_filter: Optional[datetime] = None,
        boost_authority: bool = True,
        boost_citations: bool = True
    ) -> List[LegalSearchResult]:
        """Enhanced legal search with authority and temporal awareness"""
        
        # Create query embedding
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search for more candidates than needed for post-filtering
        search_k = min(k * 5, self.total_chunks)
        scores, indices = self.index.search(query_embedding, search_k)
        
        results = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # Invalid index
                continue
                
            chunk_id = self.index_to_chunk.get(idx)
            if not chunk_id:
                continue
                
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            
            # Apply filters
            if not self._passes_filters(chunk, legal_area, jurisdiction, authority_threshold, temporal_filter):
                continue
            
            # Calculate boosts
            authority_boost = self._calculate_authority_boost(chunk) if boost_authority else 1.0
            temporal_boost = self._calculate_temporal_boost(chunk, temporal_filter) if temporal_filter else 1.0
            citation_boost = self._calculate_citation_boost(chunk) if boost_citations and self.citation_graph else 1.0
            
            # Calculate final score
            final_score = score * authority_boost * temporal_boost * citation_boost
            
            # Create result
            result = LegalSearchResult(
                chunk_id=chunk_id,
                document_id=chunk.document_id,
                score=score,
                authority_boost=authority_boost,
                temporal_boost=temporal_boost,
                citation_boost=citation_boost,
                final_score=final_score,
                text=chunk.text,
                metadata=chunk.legal_metadata,
                context=chunk.section_context or "",
                explanation=self._generate_result_explanation(chunk, authority_boost, temporal_boost, citation_boost)
            )
            results.append(result)
        
        # Sort by final score and return top k
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results[:k]
    
    def _passes_filters(
        self,
        chunk: LegalChunk,
        legal_area: Optional[str],
        jurisdiction: Optional[str],
        authority_threshold: float,
        temporal_filter: Optional[datetime]
    ) -> bool:
        """Check if chunk passes all filters"""
        metadata = chunk.legal_metadata
        
        # Authority threshold filter
        if chunk.authority_weight < authority_threshold:
            return False
        
        # Legal area filter
        if legal_area:
            if isinstance(metadata, LegalMetadata) and metadata.legal_area != legal_area:
                return False
            elif isinstance(metadata, HMRCMetadata) and metadata.tax_domain.value != legal_area:
                return False
        
        # Jurisdiction filter
        if jurisdiction:
            if isinstance(metadata, LegalMetadata):
                if (metadata.jurisdiction.value != jurisdiction and 
                    metadata.jurisdiction.value != "uk_wide"):
                    return False
            # HMRC documents are always UK-wide, so they pass jurisdiction filters
        
        # Temporal filter (only include documents effective before the filter date)
        if temporal_filter:
            effective_date = getattr(metadata, 'effective_date', None)
            if effective_date and effective_date > temporal_filter:
                return False
        
        return True
    
    def _calculate_authority_boost(self, chunk: LegalChunk) -> float:
        """Calculate authority boost based on document hierarchy"""
        return 1.0 + chunk.authority_weight  # Simple boost based on authority weight
    
    def _calculate_temporal_boost(self, chunk: LegalChunk, reference_date: datetime) -> float:
        """Calculate temporal relevance boost"""
        metadata = chunk.legal_metadata
        effective_date = getattr(metadata, 'effective_date', None)
        
        if not effective_date:
            return 0.8  # Slight penalty for unknown dates
        
        # More recent = higher boost, but legal precedents don't lose authority quickly
        days_difference = (reference_date - effective_date).days
        
        if days_difference < 365:  # Within 1 year
            return 1.2
        elif days_difference < 1825:  # Within 5 years
            return 1.1
        elif days_difference < 3650:  # Within 10 years
            return 1.0
        else:
            return 0.9  # Older documents get slight penalty
    
    def _calculate_citation_boost(self, chunk: LegalChunk) -> float:
        """Calculate boost based on citation graph"""
        if not self.citation_graph:
            return 1.0
        
        # Get authority score from citation graph
        authority = self.citation_graph.authority_scores.get(chunk.document_id)
        if authority:
            return 1.0 + (authority.authority_score * 0.5)  # Moderate boost
        
        return 1.0
    
    def _generate_result_explanation(
        self,
        chunk: LegalChunk,
        authority_boost: float,
        temporal_boost: float,
        citation_boost: float
    ) -> str:
        """Generate human-readable explanation for result ranking"""
        explanations = []
        
        if authority_boost > 1.1:
            explanations.append("high authority source")
        elif authority_boost < 0.9:
            explanations.append("lower authority source")
        
        if temporal_boost > 1.05:
            explanations.append("recent")
        elif temporal_boost < 0.95:
            explanations.append("older precedent")
        
        if citation_boost > 1.05:
            explanations.append("frequently cited")
        
        if chunk.section_context:
            explanations.append(f"from {chunk.section_context}")
        
        return "; ".join(explanations) if explanations else "standard relevance"
    
    def set_citation_graph(self, citation_graph: LegalCitationGraph) -> None:
        """Set citation graph for enhanced retrieval"""
        self.citation_graph = citation_graph
        logger.info("Citation graph integrated with FAISS index")
    
    def save_index(self, index_path: Path) -> bool:
        """Save the complete index to disk"""
        try:
            index_path.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(index_path / "faiss.index"))
            
            # Save metadata
            metadata = {
                'chunks': {cid: asdict(chunk) for cid, chunk in self.chunks.items()},
                'chunk_to_index': self.chunk_to_index,
                'index_to_chunk': self.index_to_chunk,
                'document_metadata': {did: meta.to_dict() for did, meta in self.document_metadata.items()},
                'total_chunks': self.total_chunks,
                'documents_indexed': self.documents_indexed,
                'embedding_model': self.embedding_model.model_name,
                'dimension': self.dimension
            }
            
            with open(index_path / "metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            # Save authority weights
            with open(index_path / "authority_weights.json", 'w') as f:
                json.dump(self.authority_weights, f, indent=2)
            
            logger.info(f"Legal FAISS index saved to {index_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save index: {e}")
            return False
    
    @classmethod
    def load_index(cls, index_path: Path, embedding_model: LegalEmbeddingModel) -> 'LegalFAISSIndex':
        """Load index from disk"""
        try:
            # Load FAISS index
            index = faiss.read_index(str(index_path / "faiss.index"))
            
            # Create instance
            legal_index = cls(embedding_model)
            legal_index.index = index
            
            # Load metadata
            with open(index_path / "metadata.json", 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # Restore chunks (need to recreate LegalChunk objects)
            legal_index.chunks = {}
            for cid, chunk_data in metadata['chunks'].items():
                # Remove embedding from chunk_data as it's stored in FAISS
                chunk_data.pop('embedding', None)
                chunk = LegalChunk(**chunk_data)
                legal_index.chunks[cid] = chunk
            
            legal_index.chunk_to_index = metadata['chunk_to_index']
            legal_index.index_to_chunk = {int(k): v for k, v in metadata['index_to_chunk'].items()}
            legal_index.total_chunks = metadata['total_chunks']
            legal_index.documents_indexed = metadata['documents_indexed']
            
            # Restore document metadata (need to recreate metadata objects)
            legal_index.document_metadata = {}
            for did, meta_data in metadata['document_metadata'].items():
                # Determine metadata type and recreate
                if 'source_type' in meta_data:
                    meta = LegalMetadata.from_dict(meta_data)
                else:
                    meta = HMRCMetadata.from_dict(meta_data)
                legal_index.document_metadata[did] = meta
            
            logger.info(f"Legal FAISS index loaded from {index_path}")
            return legal_index
            
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            raise
    
    def get_statistics(self) -> Dict:
        """Get index statistics"""
        stats = {
            'total_chunks': self.total_chunks,
            'documents_indexed': self.documents_indexed,
            'index_dimension': self.dimension,
            'embedding_model': self.embedding_model.model_name
        }
        
        # Document type distribution
        doc_types = {}
        legal_areas = {}
        jurisdictions = {}
        
        for metadata in self.document_metadata.values():
            if isinstance(metadata, LegalMetadata):
                source_type = metadata.source_type.value
                doc_types[source_type] = doc_types.get(source_type, 0) + 1
                legal_areas[metadata.legal_area] = legal_areas.get(metadata.legal_area, 0) + 1
                jurisdictions[metadata.jurisdiction.value] = jurisdictions.get(metadata.jurisdiction.value, 0) + 1
            elif isinstance(metadata, HMRCMetadata):
                doc_type = metadata.document_type.value
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                tax_domain = metadata.tax_domain.value
                legal_areas[tax_domain] = legal_areas.get(tax_domain, 0) + 1
                jurisdictions['uk_wide'] = jurisdictions.get('uk_wide', 0) + 1
        
        stats.update({
            'document_types': doc_types,
            'legal_areas': legal_areas,
            'jurisdictions': jurisdictions
        })
        
        return stats
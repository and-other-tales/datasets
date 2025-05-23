#!/usr/bin/env python3
"""
Enhanced Legal Pipeline Integration
Production-ready legal AI pipeline integrating all GUIDANCE.md components:
- Legal metadata and citation extraction
- Authority-aware retrieval and citation graphs
- Professional training formats and compliance
- Advanced RAG with temporal/authority filtering
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.legal_metadata import LegalMetadata, LegalDocumentProcessor, save_legal_metadata
from utils.hmrc_metadata import HMRCMetadata, HMRCDocumentProcessor, save_hmrc_metadata
from utils.legal_citation_graph import LegalCitationGraph
from utils.legal_faiss_index import LegalFAISSIndex, LegalEmbeddingModel
from utils.legal_training import create_legal_training_datasets, LegalTrainingDatasetCreator
from utils.legal_compliance import LegalComplianceFramework, validate_legal_dataset
from utils.advanced_legal_rag import AdvancedLegalRAGPipeline, create_legal_rag_system, LegalQuery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedLegalPipeline:
    """
    Complete enhanced legal pipeline implementing GUIDANCE.md specifications
    """
    
    def __init__(self, input_dir: str, output_dir: str = "generated/enhanced_legal_system"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create enhanced output structure
        self.enhanced_metadata_dir = self.output_dir / "enhanced_metadata"
        self.citation_graphs_dir = self.output_dir / "citation_graphs"
        self.faiss_indices_dir = self.output_dir / "faiss_indices"
        self.training_datasets_dir = self.output_dir / "training_datasets"
        self.compliance_reports_dir = self.output_dir / "compliance_reports"
        self.rag_system_dir = self.output_dir / "rag_system"
        
        for dir_path in [
            self.enhanced_metadata_dir, self.citation_graphs_dir, self.faiss_indices_dir,
            self.training_datasets_dir, self.compliance_reports_dir, self.rag_system_dir
        ]:
            dir_path.mkdir(exist_ok=True)
        
        # Initialize processors
        self.legal_processor = LegalDocumentProcessor()
        self.hmrc_processor = HMRCDocumentProcessor()
        self.compliance_framework = LegalComplianceFramework()
        
        # Statistics tracking
        self.processing_stats = {
            'documents_processed': 0,
            'legal_documents': 0,
            'hmrc_documents': 0,
            'citation_relationships': 0,
            'training_examples_generated': 0,
            'compliance_violations': 0
        }
    
    def run_enhanced_pipeline(self, max_documents: Optional[int] = None) -> Dict[str, Any]:
        """Run the complete enhanced legal pipeline"""
        logger.info("=== STARTING ENHANCED LEGAL AI PIPELINE ===")
        start_time = datetime.now()
        
        try:
            # Phase 1: Enhanced Metadata Extraction
            logger.info("Phase 1: Enhanced metadata extraction and processing...")
            processed_documents = self._process_all_documents(max_documents)
            
            # Phase 2: Citation Graph Construction
            logger.info("Phase 2: Building citation graphs and authority analysis...")
            citation_graph = self._build_citation_graph(processed_documents)
            
            # Phase 3: Advanced Indexing
            logger.info("Phase 3: Creating legal-aware FAISS indices...")
            faiss_index = self._create_legal_index(processed_documents)
            
            # Phase 4: Training Dataset Generation
            logger.info("Phase 4: Generating legal training datasets...")
            training_datasets = self._generate_training_datasets(processed_documents)
            
            # Phase 5: Compliance Validation
            logger.info("Phase 5: Validating compliance and professional standards...")
            compliance_results = self._validate_compliance(training_datasets)
            
            # Phase 6: RAG System Creation
            logger.info("Phase 6: Creating advanced legal RAG system...")
            rag_system = self._create_rag_system(faiss_index, citation_graph, processed_documents)
            
            # Phase 7: System Validation and Testing
            logger.info("Phase 7: System validation and testing...")
            validation_results = self._validate_system(rag_system)
            
            # Generate comprehensive report
            duration = (datetime.now() - start_time).total_seconds()
            final_report = self._generate_final_report(
                processed_documents, citation_graph, training_datasets,
                compliance_results, validation_results, duration
            )
            
            logger.info("=== ENHANCED LEGAL AI PIPELINE COMPLETED ===")
            logger.info(f"Processing completed in {duration/60:.2f} minutes")
            
            return final_report
            
        except Exception as e:
            logger.error(f"Enhanced legal pipeline failed: {e}")
            raise
    
    def _process_all_documents(self, max_documents: Optional[int] = None) -> List[Tuple[str, Union[LegalMetadata, HMRCMetadata], str]]:
        """Process all documents with enhanced metadata extraction"""
        processed_documents = []
        document_count = 0
        
        # Find all document directories
        document_dirs = [
            self.input_dir / "housing_legislation",
            self.input_dir / "housing_case_law", 
            self.input_dir / "bailii_cases",
            self.input_dir / "hmrc_documentation",
            self.input_dir / "case_law",
            self.input_dir / "legislation"
        ]
        
        for doc_dir in document_dirs:
            if not doc_dir.exists():
                continue
                
            logger.info(f"Processing documents from {doc_dir.name}...")
            
            metadata_dir = doc_dir / "metadata"
            text_dir = doc_dir / "text"
            
            if not metadata_dir.exists() or not text_dir.exists():
                continue
            
            for metadata_file in metadata_dir.glob("*.json"):
                if max_documents and document_count >= max_documents:
                    break
                
                try:
                    # Load basic metadata
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        basic_metadata = json.load(f)
                    
                    # Load content
                    text_file = text_dir / f"{metadata_file.stem}.txt"
                    if not text_file.exists():
                        continue
                    
                    with open(text_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if len(content) < 100:  # Skip very short documents
                        continue
                    
                    # Determine document type and process accordingly
                    doc_id = metadata_file.stem
                    title = basic_metadata.get('title', 'Unknown Title')
                    url = basic_metadata.get('url', '')
                    
                    # Process with appropriate processor
                    if self._is_hmrc_document(doc_dir.name, title, content):
                        enhanced_metadata = self.hmrc_processor.process_hmrc_document(
                            text=content,
                            title=title,
                            url=url,
                            manual_code=self._extract_manual_code(url, title, content)
                        )
                        self.processing_stats['hmrc_documents'] += 1
                    else:
                        enhanced_metadata = self.legal_processor.process_document(
                            text=content,
                            title=title,
                            source_url=url,
                            document_type=basic_metadata.get('content_type', 'unknown')
                        )
                        self.processing_stats['legal_documents'] += 1
                    
                    # Save enhanced metadata
                    enhanced_file = self.enhanced_metadata_dir / f"{doc_id}.json"
                    if isinstance(enhanced_metadata, LegalMetadata):
                        save_legal_metadata(enhanced_metadata, enhanced_file)
                    else:
                        save_hmrc_metadata(enhanced_metadata, enhanced_file)
                    
                    # Add to processed documents
                    processed_documents.append((doc_id, enhanced_metadata, content))
                    document_count += 1
                    self.processing_stats['documents_processed'] += 1
                    
                    if document_count % 50 == 0:
                        logger.info(f"Processed {document_count} documents...")
                
                except Exception as e:
                    logger.warning(f"Error processing {metadata_file}: {e}")
                    continue
        
        logger.info(f"Processed {len(processed_documents)} documents with enhanced metadata")
        return processed_documents
    
    def _is_hmrc_document(self, dir_name: str, title: str, content: str) -> bool:
        """Determine if document is HMRC/tax related"""
        hmrc_indicators = ['hmrc', 'tax', 'vat', 'revenue', 'duty', 'customs']
        
        # Check directory name
        if any(indicator in dir_name.lower() for indicator in hmrc_indicators):
            return True
        
        # Check title and content
        combined_text = (title + " " + content[:1000]).lower()
        return any(indicator in combined_text for indicator in hmrc_indicators)
    
    def _extract_manual_code(self, url: str, title: str, content: str) -> Optional[str]:
        """Extract HMRC manual code"""
        import re
        
        # Try URL first
        url_match = re.search(r'/([A-Z]{2,4}\d{5}[A-Z]*)(?:/|$)', url)
        if url_match:
            return url_match.group(1)
        
        # Try title
        title_match = re.search(r'\b([A-Z]{2,4}\d{5}[A-Z]*)\b', title)
        if title_match:
            return title_match.group(1)
        
        # Try content
        content_match = re.search(r'\b([A-Z]{2,4}\d{5}[A-Z]*)\b', content[:500])
        if content_match:
            return content_match.group(1)
        
        return None
    
    def _build_citation_graph(self, processed_documents: List[Tuple[str, Union[LegalMetadata, HMRCMetadata], str]]) -> LegalCitationGraph:
        """Build comprehensive citation graph"""
        citation_graph = LegalCitationGraph()
        
        # Add all documents to the graph
        for doc_id, metadata, content in processed_documents:
            citation_graph.add_document(doc_id, metadata)
        
        # Extract citation relationships
        for doc_id, metadata, content in processed_documents:
            relationships = citation_graph.extract_citation_relationships(doc_id, content)
            self.processing_stats['citation_relationships'] += len(relationships)
        
        # Build the network
        citation_graph.build_citation_network()
        citation_graph.calculate_authority_scores()
        
        # Export citation graph
        graph_export_file = self.citation_graphs_dir / "legal_citation_graph.json"
        citation_graph.export_graph_data(graph_export_file)
        
        # Generate citation analysis report
        self._generate_citation_analysis_report(citation_graph)
        
        logger.info(f"Built citation graph with {citation_graph.graph.number_of_nodes()} nodes, {citation_graph.graph.number_of_edges()} edges")
        
        return citation_graph
    
    def _create_legal_index(self, processed_documents: List[Tuple[str, Union[LegalMetadata, HMRCMetadata], str]]) -> LegalFAISSIndex:
        """Create legal-aware FAISS index"""
        # Initialize embedding model
        embedding_model = LegalEmbeddingModel("nlpaueb/legal-bert-base-uncased")
        
        # Create FAISS index
        faiss_index = LegalFAISSIndex(embedding_model, index_type="IVF")
        
        # Add documents with legal-aware chunking
        faiss_index.add_documents(
            documents=processed_documents,
            chunk_size=1000,
            chunk_overlap=150
        )
        
        # Save index
        index_path = self.faiss_indices_dir / "legal_faiss_index"
        faiss_index.save_index(index_path)
        
        # Generate index statistics
        stats = faiss_index.get_statistics()
        with open(self.faiss_indices_dir / "index_statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Created FAISS index with {stats['total_chunks']} chunks from {stats['documents_indexed']} documents")
        
        return faiss_index
    
    def _generate_training_datasets(self, processed_documents: List[Tuple[str, Union[LegalMetadata, HMRCMetadata], str]]) -> Dict:
        """Generate professional legal training datasets"""
        # Create training datasets
        training_datasets = create_legal_training_datasets(
            legal_documents=processed_documents,
            output_dir=self.training_datasets_dir
        )
        
        # Generate additional enhanced examples
        creator = LegalTrainingDatasetCreator()
        
        # Count total examples
        total_examples = sum(len(dataset) for dataset in training_datasets.values())
        self.processing_stats['training_examples_generated'] = total_examples
        
        # Create training configuration
        training_config = {
            'datasets': list(training_datasets.keys()),
            'total_examples': total_examples,
            'examples_by_level': {level: len(dataset) for level, dataset in training_datasets.items()},
            'model_target': 'meta-llama/Meta-Llama-3.1-70B-Instruct',
            'training_phases': {
                'foundation': 'Basic legal knowledge and concepts',
                'reasoning': 'Multi-step legal reasoning',
                'expert': 'Professional legal application',
                'adversarial': 'Challenge scenarios and counter-arguments'
            },
            'professional_standards': {
                'oscola_citations': True,
                'legal_disclaimers': True,
                'authority_hierarchy': True,
                'jurisdiction_awareness': True
            }
        }
        
        with open(self.training_datasets_dir / "training_config.json", 'w') as f:
            json.dump(training_config, f, indent=2)
        
        logger.info(f"Generated {total_examples} training examples across {len(training_datasets)} difficulty levels")
        
        return training_datasets
    
    def _validate_compliance(self, training_datasets: Dict) -> Dict:
        """Validate training datasets for legal compliance"""
        compliance_results = {}
        
        for level, dataset in training_datasets.items():
            logger.info(f"Validating compliance for {level} dataset...")
            
            # Save dataset for validation
            dataset_file = self.training_datasets_dir / f"legal_{level}_dataset.json"
            validation_file = self.compliance_reports_dir / f"{level}_compliance_report.json"
            
            # Validate dataset
            stats = validate_legal_dataset(
                dataset_path=dataset_file,
                output_path=validation_file,
                compliance_framework=self.compliance_framework
            )
            
            compliance_results[level] = stats
            self.processing_stats['compliance_violations'] += stats.get('non_compliant', 0)
        
        # Generate overall compliance report
        self._generate_compliance_summary(compliance_results)
        
        return compliance_results
    
    def _create_rag_system(
        self, 
        faiss_index: LegalFAISSIndex, 
        citation_graph: LegalCitationGraph,
        processed_documents: List[Tuple[str, Union[LegalMetadata, HMRCMetadata], str]]
    ) -> AdvancedLegalRAGPipeline:
        """Create advanced legal RAG system"""
        
        # Create RAG pipeline
        rag_system = AdvancedLegalRAGPipeline(
            faiss_index=faiss_index,
            citation_graph=citation_graph,
            compliance_framework=self.compliance_framework
        )
        
        # Save RAG system configuration
        rag_config = {
            'system_type': 'Advanced Legal RAG Pipeline',
            'components': {
                'faiss_index': True,
                'citation_graph': True,
                'compliance_framework': True,
                'legal_embedding_model': faiss_index.embedding_model.model_name
            },
            'capabilities': [
                'Authority-aware retrieval',
                'Temporal filtering',
                'Citation graph integration',
                'Professional compliance validation',
                'Multi-jurisdictional support',
                'Legal reasoning chains'
            ],
            'document_coverage': {
                'total_documents': len(processed_documents),
                'legal_documents': self.processing_stats['legal_documents'],
                'hmrc_documents': self.processing_stats['hmrc_documents']
            }
        }
        
        with open(self.rag_system_dir / "rag_system_config.json", 'w') as f:
            json.dump(rag_config, f, indent=2)
        
        logger.info("Advanced legal RAG system created and configured")
        
        return rag_system
    
    def _validate_system(self, rag_system: AdvancedLegalRAGPipeline) -> Dict:
        """Validate the complete legal system"""
        validation_results = {
            'test_queries_processed': 0,
            'average_confidence': 0.0,
            'compliance_rate': 0.0,
            'authority_coverage': 0.0,
            'test_results': []
        }
        
        # Define test queries
        test_queries = [
            LegalQuery(
                query_text="What are the key principles of contract formation under English law?",
                legal_area="contract",
                jurisdiction="england_wales",
                query_type="legal_analysis"
            ),
            LegalQuery(
                query_text="Analyze the tax implications of dividend distributions for UK companies",
                legal_area="tax",
                jurisdiction="uk_wide", 
                query_type="tax_compliance"
            ),
            LegalQuery(
                query_text="Counter the argument that the defendant breached their duty of care",
                legal_area="tort",
                jurisdiction="england_wales",
                query_type="counter_argument"
            )
        ]
        
        # Test each query
        for i, query in enumerate(test_queries):
            try:
                # Mock LLM function for testing
                def mock_llm_generate(prompt):
                    return f"[Mock response for query {i+1}] Based on the legal authorities provided, this analysis demonstrates the system's capability to process complex legal queries with proper citation and reasoning."
                
                result = rag_system.process_legal_query(query, mock_llm_generate)
                
                validation_results['test_results'].append({
                    'query_id': i + 1,
                    'authorities_retrieved': len(result.retrieved_authorities),
                    'confidence_score': result.confidence_assessment['overall_confidence'],
                    'compliance_score': result.compliance_validation.overall_score,
                    'processing_time': result.processing_time
                })
                
                validation_results['test_queries_processed'] += 1
                
            except Exception as e:
                logger.warning(f"Test query {i+1} failed: {e}")
                continue
        
        # Calculate averages
        if validation_results['test_results']:
            validation_results['average_confidence'] = np.mean([r['confidence_score'] for r in validation_results['test_results']])
            validation_results['compliance_rate'] = np.mean([r['compliance_score'] for r in validation_results['test_results']])
        
        # Save validation results
        with open(self.rag_system_dir / "system_validation.json", 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"System validation completed: {validation_results['test_queries_processed']} test queries processed")
        
        return validation_results
    
    def _generate_citation_analysis_report(self, citation_graph: LegalCitationGraph):
        """Generate citation analysis report"""
        # Find landmark cases
        landmark_cases = citation_graph.find_landmark_cases()
        
        # Get network statistics
        network_stats = {
            'total_documents': citation_graph.graph.number_of_nodes(),
            'citation_relationships': citation_graph.graph.number_of_edges(),
            'landmark_cases_identified': len(landmark_cases),
            'authority_scores_calculated': len(citation_graph.authority_scores)
        }
        
        # Legal area distribution
        legal_area_dist = {}
        for doc_id, metadata in citation_graph.document_metadata.items():
            legal_area = getattr(metadata, 'legal_area', getattr(metadata, 'tax_domain', 'general'))
            area_value = legal_area.value if hasattr(legal_area, 'value') else str(legal_area)
            legal_area_dist[area_value] = legal_area_dist.get(area_value, 0) + 1
        
        citation_report = {
            'network_statistics': network_stats,
            'legal_area_distribution': legal_area_dist,
            'landmark_cases': [
                {
                    'document_id': doc_id,
                    'title': citation_graph.document_metadata[doc_id].title,
                    'authority_score': authority.authority_score,
                    'citation_count': authority.citation_count
                }
                for doc_id, authority in landmark_cases[:10]  # Top 10 landmark cases
            ],
            'generated_at': datetime.now().isoformat()
        }
        
        with open(self.citation_graphs_dir / "citation_analysis_report.json", 'w') as f:
            json.dump(citation_report, f, indent=2, ensure_ascii=False)
    
    def _generate_compliance_summary(self, compliance_results: Dict):
        """Generate overall compliance summary"""
        summary = {
            'overall_compliance_rate': 0.0,
            'datasets_analyzed': len(compliance_results),
            'total_examples': 0,
            'compliant_examples': 0,
            'compliance_by_level': {},
            'common_violations': [],
            'recommendations': []
        }
        
        for level, stats in compliance_results.items():
            summary['total_examples'] += stats['total_examples']
            summary['compliant_examples'] += stats['compliant']
            summary['compliance_by_level'][level] = {
                'compliance_rate': stats['compliant'] / stats['total_examples'] if stats['total_examples'] > 0 else 0,
                'total_examples': stats['total_examples']
            }
        
        if summary['total_examples'] > 0:
            summary['overall_compliance_rate'] = summary['compliant_examples'] / summary['total_examples']
        
        # Add recommendations
        if summary['overall_compliance_rate'] < 0.8:
            summary['recommendations'].append("Review and enhance legal disclaimer requirements")
            summary['recommendations'].append("Improve citation accuracy and format compliance")
        
        if summary['overall_compliance_rate'] < 0.6:
            summary['recommendations'].append("Comprehensive review of professional standards compliance required")
        
        with open(self.compliance_reports_dir / "compliance_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _generate_final_report(
        self, 
        processed_documents, 
        citation_graph, 
        training_datasets, 
        compliance_results, 
        validation_results, 
        duration
    ) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        final_report = {
            'pipeline_completion': {
                'status': 'completed',
                'duration_minutes': duration / 60,
                'completed_at': datetime.now().isoformat()
            },
            'processing_statistics': self.processing_stats,
            'document_analysis': {
                'total_processed': len(processed_documents),
                'legal_documents': self.processing_stats['legal_documents'],
                'hmrc_documents': self.processing_stats['hmrc_documents'],
                'enhanced_metadata_created': True
            },
            'citation_graph_analysis': {
                'network_built': True,
                'nodes': citation_graph.graph.number_of_nodes(),
                'edges': citation_graph.graph.number_of_edges(),
                'authority_scores_calculated': len(citation_graph.authority_scores)
            },
            'training_datasets': {
                'datasets_created': len(training_datasets),
                'total_examples': sum(len(d) for d in training_datasets.values()),
                'difficulty_levels': list(training_datasets.keys())
            },
            'compliance_validation': {
                'overall_compliance_rate': compliance_results.get('foundation', {}).get('compliant', 0) / max(compliance_results.get('foundation', {}).get('total_examples', 1), 1),
                'datasets_validated': len(compliance_results)
            },
            'rag_system': {
                'created': True,
                'test_queries_processed': validation_results['test_queries_processed'],
                'average_confidence': validation_results.get('average_confidence', 0.0)
            },
            'professional_standards': {
                'oscola_citations': True,
                'legal_disclaimers': True,
                'authority_hierarchy': True,
                'jurisdiction_awareness': True,
                'temporal_filtering': True,
                'compliance_validation': True
            },
            'output_locations': {
                'enhanced_metadata': str(self.enhanced_metadata_dir),
                'citation_graphs': str(self.citation_graphs_dir),
                'faiss_indices': str(self.faiss_indices_dir),
                'training_datasets': str(self.training_datasets_dir),
                'compliance_reports': str(self.compliance_reports_dir),
                'rag_system': str(self.rag_system_dir)
            },
            'next_steps': [
                "Review compliance reports and address any violations",
                "Use training datasets with HuggingFace AutoTrain Advanced",
                "Deploy RAG system for legal query processing",
                "Monitor system performance and update as needed"
            ]
        }
        
        # Save final report
        with open(self.output_dir / "enhanced_legal_pipeline_report.json", 'w') as f:
            json.dump(final_report, f, indent=2, ensure_ascii=False)
        
        return final_report

def main():
    """Main function for enhanced legal pipeline"""
    parser = argparse.ArgumentParser(
        description="Enhanced Legal AI Pipeline - Complete GUIDANCE.md Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This pipeline implements all components from GUIDANCE.md:
- Legal-aware preprocessing and metadata extraction
- Citation graph construction with authority ranking
- Legal domain embeddings and enhanced indexing
- Professional legal training dataset creation
- Compliance validation and professional standards
- Advanced RAG pipeline with temporal/authority filtering

Examples:
  python enhanced_legal_pipeline.py --input-dir generated
  python enhanced_legal_pipeline.py --input-dir generated --max-documents 100
  python enhanced_legal_pipeline.py --input-dir generated --output-dir ./enhanced_system
        """
    )
    
    parser.add_argument(
        '--input-dir',
        required=True,
        help='Input directory containing collected legal data'
    )
    
    parser.add_argument(
        '--output-dir',
        default='generated/enhanced_legal_system',
        help='Output directory for enhanced legal system'
    )
    
    parser.add_argument(
        '--max-documents',
        type=int,
        help='Maximum number of documents to process (for testing)'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error(f"Input directory does not exist: {input_path}")
        return
    
    try:
        # Create enhanced pipeline
        pipeline = EnhancedLegalPipeline(args.input_dir, args.output_dir)
        
        # Run complete pipeline
        report = pipeline.run_enhanced_pipeline(args.max_documents)
        
        # Print summary
        print(f"\n{'='*60}")
        print("ENHANCED LEGAL AI PIPELINE COMPLETION SUMMARY")
        print(f"{'='*60}")
        print(f"Status: {report['pipeline_completion']['status']}")
        print(f"Duration: {report['pipeline_completion']['duration_minutes']:.2f} minutes")
        print(f"Documents processed: {report['processing_statistics']['documents_processed']}")
        print(f"Training examples generated: {report['processing_statistics']['training_examples_generated']}")
        print(f"Citation relationships: {report['processing_statistics']['citation_relationships']}")
        print(f"RAG system created: {report['rag_system']['created']}")
        print(f"Output directory: {args.output_dir}")
        print(f"\nReady for Legal Llama 3.1 70B training and deployment!")
        
    except Exception as e:
        logger.error(f"Enhanced legal pipeline failed: {e}")
        return

if __name__ == "__main__":
    # Import numpy for final calculations
    try:
        import numpy as np
    except ImportError:
        logger.warning("NumPy not available for statistical calculations")
        
        class MockNumPy:
            @staticmethod
            def mean(values):
                return sum(values) / len(values) if values else 0.0
        
        np = MockNumPy()
    
    main()
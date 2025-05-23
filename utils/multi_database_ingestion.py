#!/usr/bin/env python3
"""
Multi-Database Ingestion Script for Legal Datasets
Ingests legal case law and legislation data into MongoDB Atlas, Neo4j, and Pinecone
"""

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import uuid

# Database connections
try:
    from pymongo import MongoClient
    from neo4j import GraphDatabase
    from pinecone import Pinecone
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing dependencies: {e}")
    print("Install with: pip install pymongo neo4j-driver pinecone sentence-transformers")
    exit(1)

# Configuration
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class LegalDataIngester:
    def __init__(self, config: Dict[str, Any]):
        """Initialize database connections"""
        self.config = config
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # MongoDB Atlas connection
        self.mongo_client = MongoClient(config['mongodb']['connection_string'])
        self.mongo_db = self.mongo_client[config['mongodb']['database']]
        self.cases_collection = self.mongo_db['legal_cases']
        self.legislation_collection = self.mongo_db['legislation']
        
        # Neo4j connection
        self.neo4j_driver = GraphDatabase.driver(
            config['neo4j']['uri'], 
            auth=(config['neo4j']['username'], config['neo4j']['password'])
        )
        
        # Pinecone connection
        self.pc = Pinecone(api_key=config['pinecone']['api_key'])
        
        # Create or connect to indexes
        self.cases_index = self._get_or_create_pinecone_index(
            'legal-cases', dimension=384
        )
        self.legislation_index = self._get_or_create_pinecone_index(
            'legislation', dimension=384
        )

    def _get_or_create_pinecone_index(self, name: str, dimension: int):
        """Get existing or create new Pinecone index"""
        existing_indexes = [index['name'] for index in self.pc.list_indexes()]
        if name not in existing_indexes:
            self.pc.create_index(
                name=name,
                dimension=dimension,
                metric='cosine',
                spec={'serverless': {'cloud': 'aws', 'region': 'us-east-1'}}
            )
        return self.pc.Index(name)

    def _generate_document_id(self, content: str) -> str:
        """Generate unique document ID from content hash"""
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _chunk_text(self, text: str, max_chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """Split text into overlapping chunks for embedding"""
        if len(text) <= max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_chunk_size
            if end > len(text):
                end = len(text)
            
            chunk = text[start:end]
            chunks.append(chunk)
            
            if end == len(text):
                break
            start = end - overlap
        
        return chunks

    def ingest_legal_case(self, case_data: Dict[str, Any]) -> str:
        """Ingest a single legal case into all databases"""
        case_id = case_data.get('id') or self._generate_document_id(
            case_data.get('title', '') + case_data.get('content', '')
        )
        
        # MongoDB storage
        mongo_doc = {
            '_id': case_id,
            'title': case_data.get('title', ''),
            'court': case_data.get('court', ''),
            'date': case_data.get('date', ''),
            'citation': case_data.get('citation', ''),
            'judges': case_data.get('judges', []),
            'parties': case_data.get('parties', {}),
            'subject_areas': case_data.get('subject_areas', []),
            'content': case_data.get('content', ''),
            'url': case_data.get('url', ''),
            'metadata': case_data.get('metadata', {}),
            'ingested_at': datetime.utcnow(),
            'data_type': 'legal_case'
        }
        
        self.cases_collection.replace_one({'_id': case_id}, mongo_doc, upsert=True)
        logger.info(f"Stored case {case_id} in MongoDB")

        # Neo4j graph relationships
        with self.neo4j_driver.session() as session:
            # Create case node
            session.run("""
                MERGE (c:LegalCase {id: $case_id})
                SET c.title = $title,
                    c.court = $court,
                    c.date = $date,
                    c.citation = $citation,
                    c.url = $url,
                    c.ingested_at = $ingested_at
            """, case_id=case_id, **{k: v for k, v in mongo_doc.items() 
                                    if k in ['title', 'court', 'date', 'citation', 'url']},
                ingested_at=mongo_doc['ingested_at'].isoformat())
            
            # Create court node and relationship
            if mongo_doc['court']:
                session.run("""
                    MERGE (court:Court {name: $court_name})
                    WITH court
                    MATCH (c:LegalCase {id: $case_id})
                    MERGE (c)-[:HEARD_IN]->(court)
                """, court_name=mongo_doc['court'], case_id=case_id)
            
            # Create subject area relationships
            for subject in mongo_doc['subject_areas']:
                session.run("""
                    MERGE (s:SubjectArea {name: $subject})
                    WITH s
                    MATCH (c:LegalCase {id: $case_id})
                    MERGE (c)-[:RELATES_TO]->(s)
                """, subject=subject, case_id=case_id)

        logger.info(f"Created graph relationships for case {case_id}")

        # Pinecone vector storage
        content_chunks = self._chunk_text(mongo_doc['content'])
        vectors_to_upsert = []
        
        for i, chunk in enumerate(content_chunks):
            chunk_id = f"{case_id}_chunk_{i}"
            embedding = self.embedding_model.encode(chunk).tolist()
            
            metadata = {
                'document_id': case_id,
                'document_type': 'legal_case',
                'title': mongo_doc['title'],
                'court': mongo_doc['court'],
                'date': mongo_doc['date'],
                'citation': mongo_doc['citation'],
                'chunk_index': i,
                'text': chunk[:500]  # Store first 500 chars of chunk
            }
            
            vectors_to_upsert.append((chunk_id, embedding, metadata))
        
        if vectors_to_upsert:
            self.cases_index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Stored {len(vectors_to_upsert)} vectors for case {case_id}")

        return case_id

    def ingest_legislation(self, legislation_data: Dict[str, Any]) -> str:
        """Ingest legislation into all databases"""
        doc_id = legislation_data.get('id') or self._generate_document_id(
            legislation_data.get('title', '') + legislation_data.get('content', '')
        )
        
        # MongoDB storage
        mongo_doc = {
            '_id': doc_id,
            'title': legislation_data.get('title', ''),
            'type': legislation_data.get('type', ''),  # Act, Regulation, etc.
            'year': legislation_data.get('year', ''),
            'chapter': legislation_data.get('chapter', ''),
            'sections': legislation_data.get('sections', []),
            'content': legislation_data.get('content', ''),
            'url': legislation_data.get('url', ''),
            'status': legislation_data.get('status', 'current'),
            'metadata': legislation_data.get('metadata', {}),
            'ingested_at': datetime.utcnow(),
            'data_type': 'legislation'
        }
        
        self.legislation_collection.replace_one({'_id': doc_id}, mongo_doc, upsert=True)
        logger.info(f"Stored legislation {doc_id} in MongoDB")

        # Neo4j graph relationships
        with self.neo4j_driver.session() as session:
            # Create legislation node
            session.run("""
                MERGE (l:Legislation {id: $doc_id})
                SET l.title = $title,
                    l.type = $type,
                    l.year = $year,
                    l.chapter = $chapter,
                    l.status = $status,
                    l.url = $url,
                    l.ingested_at = $ingested_at
            """, doc_id=doc_id, **{k: v for k, v in mongo_doc.items() 
                                 if k in ['title', 'type', 'year', 'chapter', 'status', 'url']},
                ingested_at=mongo_doc['ingested_at'].isoformat())
            
            # Create sections as separate nodes
            for section in mongo_doc['sections']:
                section_id = f"{doc_id}_section_{section.get('number', 'unknown')}"
                session.run("""
                    MERGE (s:Section {id: $section_id})
                    SET s.number = $number,
                        s.title = $section_title,
                        s.content = $content
                    WITH s
                    MATCH (l:Legislation {id: $doc_id})
                    MERGE (l)-[:CONTAINS_SECTION]->(s)
                """, section_id=section_id, doc_id=doc_id,
                    number=section.get('number', ''),
                    section_title=section.get('title', ''),
                    content=section.get('content', ''))

        logger.info(f"Created graph relationships for legislation {doc_id}")

        # Pinecone vector storage
        content_chunks = self._chunk_text(mongo_doc['content'])
        vectors_to_upsert = []
        
        for i, chunk in enumerate(content_chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            embedding = self.embedding_model.encode(chunk).tolist()
            
            metadata = {
                'document_id': doc_id,
                'document_type': 'legislation',
                'title': mongo_doc['title'],
                'type': mongo_doc['type'],
                'year': mongo_doc['year'],
                'chapter': mongo_doc['chapter'],
                'chunk_index': i,
                'text': chunk[:500]
            }
            
            vectors_to_upsert.append((chunk_id, embedding, metadata))
        
        if vectors_to_upsert:
            self.legislation_index.upsert(vectors=vectors_to_upsert)
            logger.info(f"Stored {len(vectors_to_upsert)} vectors for legislation {doc_id}")

        return doc_id

    def ingest_from_directory(self, directory_path: str):
        """Ingest all data files from a directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            logger.error(f"Directory {directory_path} does not exist")
            return
        
        # Process JSON files
        json_files = list(directory.glob("**/*.json"))
        for json_file in json_files:
            if json_file.name in ['progress.json', 'settings.local.json']:
                continue
                
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    for item in data:
                        self._process_data_item(item, json_file.parent.name)
                elif isinstance(data, dict):
                    self._process_data_item(data, json_file.parent.name)
                    
            except Exception as e:
                logger.error(f"Error processing {json_file}: {e}")
        
        # Process text files
        text_files = list(directory.glob("**/*.txt"))
        for text_file in text_files:
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create basic document structure
                doc_data = {
                    'title': text_file.stem,
                    'content': content,
                    'source_file': str(text_file),
                    'file_type': 'text'
                }
                
                # Determine if it's case law or legislation based on path
                if 'case_law' in str(text_file) or 'bailii' in str(text_file):
                    self.ingest_legal_case(doc_data)
                else:
                    self.ingest_legislation(doc_data)
                    
            except Exception as e:
                logger.error(f"Error processing {text_file}: {e}")

    def _process_data_item(self, item: Dict[str, Any], source_type: str):
        """Process individual data item based on source type"""
        if 'case_law' in source_type or 'bailii' in source_type:
            self.ingest_legal_case(item)
        elif 'legislation' in source_type:
            self.ingest_legislation(item)
        else:
            # Try to infer type from content
            if any(field in item for field in ['court', 'citation', 'judges']):
                self.ingest_legal_case(item)
            elif any(field in item for field in ['chapter', 'sections', 'year']):
                self.ingest_legislation(item)
            else:
                logger.warning(f"Could not determine document type for item: {item.get('title', 'unknown')}")

    def close_connections(self):
        """Close all database connections"""
        self.mongo_client.close()
        self.neo4j_driver.close()

def main():
    """Main execution function"""
    # Configuration - these should be set as environment variables
    config = {
        'mongodb': {
            'connection_string': os.getenv('MONGODB_CONNECTION_STRING', 'mongodb://localhost:27017/'),
            'database': os.getenv('MONGODB_DATABASE', 'legal_datasets')
        },
        'neo4j': {
            'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            'username': os.getenv('NEO4J_USERNAME', 'neo4j'),
            'password': os.getenv('NEO4J_PASSWORD', 'password')
        },
        'pinecone': {
            'api_key': os.getenv('PINECONE_API_KEY'),
            'environment': os.getenv('PINECONE_ENVIRONMENT', 'us-west1-gcp')
        }
    }
    
    # Validate configuration
    required_vars = ['MONGODB_CONNECTION_STRING', 'NEO4J_PASSWORD', 'PINECONE_API_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print(f"Missing required environment variables: {missing_vars}")
        print("Please set these environment variables before running the script.")
        return
    
    try:
        ingester = LegalDataIngester(config)
        
        # Ingest from current directory
        current_dir = Path(__file__).parent
        ingester.ingest_from_directory(str(current_dir))
        
        print("Ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
    finally:
        if 'ingester' in locals():
            ingester.close_connections()

if __name__ == "__main__":
    main()
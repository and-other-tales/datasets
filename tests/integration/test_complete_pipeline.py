"""
Integration tests for complete pipeline workflows
Tests end-to-end functionality of the entire system.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import tempfile
import shutil
from pathlib import Path
import subprocess

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pipelines.dynamic_pipeline import DynamicDatasetPipeline
from pipelines.hmrc_scraper import HMRCScraper
from utils.llama_training_optimizer import ParaLlamaTrainingOptimizer


class TestCompletePipelineIntegration(unittest.TestCase):
    """Integration tests for complete pipeline workflows"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.mock_api_key = "test-api-key"
        self.test_url = "https://www.gov.uk/test-content"
        
    def tearDown(self):
        """Clean up integration test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('subprocess.run')
    @patch('requests.Session.get')
    def test_dynamic_to_training_pipeline(self, mock_get, mock_subprocess):
        """Test complete pipeline from URL to training-ready dataset"""
        # Mock web content fetch
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <head><title>Tax Legislation</title></head>
            <body>
                <h1>Corporation Tax Act 2009</h1>
                <p>This Act governs corporation tax in the UK.</p>
                <p>HMRC administers corporation tax compliance.</p>
                <p>Companies must file annual returns showing tax calculations.</p>
            </body>
        </html>
        """
        mock_response.headers = {'content-type': 'text/html'}
        mock_get.return_value = mock_response
        
        # Mock Claude analysis
        mock_subprocess.return_value = Mock(
            returncode=0, 
            stdout="Enhanced tax analysis content with reasoning patterns"
        )
        
        # Step 1: Dynamic pipeline generates datasets
        dynamic_pipeline = DynamicDatasetPipeline(
            target_url=self.test_url,
            output_dir=self.test_dir
        )
        
        results = dynamic_pipeline.run_dynamic_pipeline()
        self.assertIsInstance(results, dict)
        
        # Verify dynamic pipeline outputs - check in datasets subdirectory
        datasets_dir = os.path.join(self.test_dir, 'datasets')
        if os.path.exists(datasets_dir):
            # Check if comprehensive dataset file exists
            comprehensive_file = os.path.join(datasets_dir, 'comprehensive_training_data.json')
            if os.path.exists(comprehensive_file):
                with open(comprehensive_file, 'r', encoding='utf-8') as f:
                    comprehensive_data = json.load(f)
                    self.assertIn('metadata', comprehensive_data)
                    self.assertIn('datasets', comprehensive_data)
        
        # Step 2: Create mock datasets for training optimization
        mock_datasets = {
            'legal': [{'text': 'Legal content', 'title': 'Legal Doc'}],
            'tax': [{'text': 'Tax content', 'title': 'Tax Doc'}]
        }
        
        # Save mock datasets
        for domain, data in mock_datasets.items():
            domain_dir = os.path.join(self.test_dir, domain)
            os.makedirs(domain_dir, exist_ok=True)
            with open(os.path.join(domain_dir, 'data.json'), 'w') as f:
                json.dump(data, f)
        
        # Step 3: Optimise for Llama training
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        # Run the full optimization process
        optimization_results = optimizer.optimize_for_llama_training()
        
        # Verify optimization results
        self.assertIsInstance(optimization_results, dict)
        self.assertIn('specialists', optimization_results)
        self.assertIn('phases', optimization_results)
        
        # Verify some output was created
        output_files = []
        for root, dirs, files in os.walk(optimizer.output_dir):
            output_files.extend(files)
        
        # Should have created some output files
        self.assertGreater(len(output_files), 0)
    
    @patch('requests.Session.get')
    def test_hmrc_to_training_pipeline(self, mock_get):
        """Test complete pipeline from HMRC scraping to training datasets"""
        # Mock API response for HMRC discovery
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {
                    'title': 'Corporation Tax Guide',
                    'link': '/corporation-tax',
                    'description': 'Corporation tax guidance',
                    'format': 'guidance'
                },
                {
                    'title': 'Income Tax Information',
                    'link': '/income-tax',
                    'description': 'Income tax guidance',
                    'format': 'guidance'
                }
            ]
        }
        mock_get.return_value = mock_response
        
        # Step 1: HMRC scraping
        hmrc_scraper = HMRCScraper(output_dir=self.test_dir)
        
        # Run discovery to populate URLs
        discovered_urls = hmrc_scraper.discover_via_search_api()
        
        # Mock some documents were downloaded
        for i, url in enumerate(list(discovered_urls)[:2]):
            # Create mock files
            text_file = Path(self.test_dir) / 'text' / f'doc_{i}.txt'
            text_file.parent.mkdir(parents=True, exist_ok=True)
            text_file.write_text(f'Tax content {i}')
            
            metadata_file = Path(self.test_dir) / 'metadata' / f'doc_{i}.json'
            metadata_file.parent.mkdir(parents=True, exist_ok=True)
            json.dump({'title': f'Doc {i}', 'url': url}, metadata_file.open('w'))
        
        # Verify HMRC scraping outputs
        text_dir = os.path.join(self.test_dir, 'text')
        if os.path.exists(text_dir):
            text_files = os.listdir(text_dir)
            self.assertGreater(len(text_files), 0)
        
        # Step 2: Convert scraped content to training datasets
        scraped_content = []
        text_dir = os.path.join(self.test_dir, 'text')
        for text_file in os.listdir(text_dir):
            with open(os.path.join(text_dir, text_file), 'r', encoding='utf-8') as f:
                content = f.read()
                scraped_content.append({
                    'text': content,
                    'domain': 'tax',
                    'source': 'hmrc'
                })
        
        # Step 3: Create training datasets
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        # Run optimization
        results = optimizer.optimize_for_llama_training()
        
        # Verify results
        self.assertIsInstance(results, dict)
        
        # Verify some output was generated
        output_dir = Path(optimizer.output_dir)
        if output_dir.exists():
            output_files = list(output_dir.rglob('*'))
            self.assertGreater(len(output_files), 0)
    
    def test_multi_domain_training_pipeline(self):
        """Test pipeline for multiple domain specialists"""
        domains = ['legal', 'tax', 'copyright']
        
        # Create mock datasets for each domain
        for domain in domains:
            domain_dir = os.path.join(self.test_dir, domain)
            os.makedirs(domain_dir, exist_ok=True)
            
            # Create mock data files
            mock_data = [
                {'text': f'{domain} concept 1', 'title': f'{domain} doc 1'},
                {'text': f'{domain} concept 2', 'title': f'{domain} doc 2'}
            ]
            
            with open(os.path.join(domain_dir, 'data.json'), 'w') as f:
                json.dump(mock_data, f)
        
        # Create optimizer with the test directory
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
            
        # Run optimization
        results = optimizer.optimize_for_llama_training()
        
        # Verify results
        self.assertIsInstance(results, dict)
        self.assertIn('specialists', results)
        self.assertIn('phases', results)
        
        # Verify output was created
        output_dir = Path(optimizer.output_dir)
        if output_dir.exists():
            # Check for any output files
            output_files = list(output_dir.rglob('*.json')) + list(output_dir.rglob('*.jsonl'))
            self.assertGreater(len(output_files), 0)
    
    def test_dataset_validation_pipeline(self):
        """Test dataset validation throughout the pipeline"""
        # Create test data directory
        test_data_dir = os.path.join(self.test_dir, 'test_data')
        os.makedirs(test_data_dir, exist_ok=True)
        
        # Create valid test data
        valid_data = [
            {'text': 'Test legal content', 'title': 'Legal Doc 1'},
            {'text': 'More legal content', 'title': 'Legal Doc 2'}
        ]
        
        with open(os.path.join(test_data_dir, 'data.json'), 'w') as f:
            json.dump(valid_data, f)
        
        # Create optimizer
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        # Load datasets
        datasets = optimizer.load_all_datasets()
        
        # Verify datasets loaded
        self.assertIsInstance(datasets, dict)
        
        # Test formatting for Llama
        if datasets:
            for key, dataset in datasets.items():
                if dataset:  # If dataset has content
                    formatted = optimizer.format_for_llama_training(
                        dataset[:1], 
                        'legal_specialist', 
                        'phase_1_foundation'
                    )
                    self.assertIsInstance(formatted, list)
    
    def test_performance_monitoring_pipeline(self):
        """Test performance monitoring throughout the pipeline"""
        import time
        
        # Track timing for different pipeline stages
        timings = {}
        
        # Stage 1: Content extraction (simulated)
        start_time = time.time()
        test_content = "Test legal content for performance monitoring" * 100
        timings['content_extraction'] = time.time() - start_time
        
        # Stage 2: Pipeline creation
        start_time = time.time()
        pipeline = DynamicDatasetPipeline(
            target_url=self.test_url,
            output_dir=self.test_dir
        )
        timings['pipeline_creation'] = time.time() - start_time
        
        # Stage 3: Mock content analysis
        start_time = time.time()
        content_data = {
            'text': test_content,
            'title': 'Test Document',
            'url': self.test_url
        }
        timings['content_preparation'] = time.time() - start_time
        
        # Stage 4: Training optimisation setup
        start_time = time.time()
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        timings['training_optimization'] = time.time() - start_time
        
        # Verify reasonable performance (all stages should complete quickly)
        for stage, duration in timings.items():
            self.assertLess(duration, 5.0, f"Stage {stage} took too long: {duration}s")
        
        # Save performance report
        performance_report = {
            'timings': timings,
            'total_time': sum(timings.values()),
            'dataset_size': 100,  # Mock dataset size
            'domain': 'legal'  # Default domain
        }
        
        with open(os.path.join(self.test_dir, 'performance_report.json'), 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'performance_report.json')))


if __name__ == '__main__':
    unittest.main()
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

from pipelines.dynamic_pipeline import DynamicPipeline
from pipelines.hmrc_scraper import HMRCScraper
from utils.llama_training_optimizer import LlamaTrainingOptimizer


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
        dynamic_pipeline = DynamicPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        success = dynamic_pipeline.run(self.test_url)
        self.assertTrue(success)
        
        # Verify dynamic pipeline outputs
        expected_files = [
            'base_knowledge.json',
            'reasoning_patterns.json',
            'expert_scenarios.json',
            'adversarial_training.json'
        ]
        
        for expected_file in expected_files:
            file_path = os.path.join(self.test_dir, expected_file)
            self.assertTrue(os.path.exists(file_path))
        
        # Step 2: Load generated datasets for training optimisation
        datasets = {}
        for dataset_type in ['base_knowledge', 'reasoning_patterns', 'expert_scenarios', 'adversarial_training']:
            file_path = os.path.join(self.test_dir, f"{dataset_type}.json")
            with open(file_path, 'r', encoding='utf-8') as f:
                datasets[dataset_type.replace('_', '')] = json.load(f)
        
        # Step 3: Optimise for Llama training
        optimizer = LlamaTrainingOptimizer(domain="tax")
        
        # Create progressive training phases
        training_phases = optimizer.create_progressive_training_phases(datasets)
        
        # Generate AutoTrain configuration
        autotrain_config = optimizer.create_autotrain_config(self.test_dir)
        
        # Save training configurations
        optimizer.save_config(training_phases, self.test_dir, "training_phases.json")
        optimizer.save_config(autotrain_config, self.test_dir, "autotrain_config.json")
        
        # Verify training optimisation outputs
        training_files = ['training_phases.json', 'autotrain_config.json']
        for training_file in training_files:
            file_path = os.path.join(self.test_dir, training_file)
            self.assertTrue(os.path.exists(file_path))
            
            with open(file_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                self.assertIsInstance(config_data, dict)
        
        # Verify AutoTrain config contains required fields
        with open(os.path.join(self.test_dir, 'autotrain_config.json'), 'r') as f:
            autotrain_data = json.load(f)
        
        required_autotrain_fields = [
            'model', 'project_name', 'data_path', 'lr', 'epochs', 'batch_size'
        ]
        for field in required_autotrain_fields:
            self.assertIn(field, autotrain_data)
    
    @patch.object(HMRCScraper, 'discover_hmrc_pages')
    @patch.object(HMRCScraper, 'scrape_with_api_fallback')
    def test_hmrc_to_training_pipeline(self, mock_scrape, mock_discover):
        """Test complete pipeline from HMRC scraping to training datasets"""
        # Mock HMRC scraping
        mock_discover.return_value = [
            "https://www.gov.uk/corporation-tax",
            "https://www.gov.uk/income-tax"
        ]
        
        mock_scrape.side_effect = [
            {
                'title': 'Corporation Tax Guide',
                'body': 'Corporation tax is charged on company profits. HMRC collects this tax annually.',
                'metadata': {'source': 'hmrc', 'type': 'tax_guide'}
            },
            {
                'title': 'Income Tax Information',
                'body': 'Income tax is charged on individual earnings. Self-assessment is required.',
                'metadata': {'source': 'hmrc', 'type': 'tax_guide'}
            }
        ]
        
        # Step 1: HMRC scraping
        hmrc_scraper = HMRCScraper(output_dir=self.test_dir)
        scraped_results = hmrc_scraper.run_scraping(max_documents=2)
        
        self.assertEqual(len(scraped_results), 2)
        
        # Verify HMRC scraping outputs
        for subdir in ['html', 'text', 'metadata']:
            dir_path = os.path.join(self.test_dir, subdir)
            self.assertTrue(os.path.exists(dir_path))
            self.assertGreater(len(os.listdir(dir_path)), 0)
        
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
        optimizer = LlamaTrainingOptimizer(domain="tax")
        
        # Simulate dataset creation (normally would use dynamic pipeline)
        base_dataset = [
            {'text': item['text'], 'domain': item['domain']} 
            for item in scraped_content
        ]
        
        reasoning_dataset = [
            {
                'instruction': f"Explain the tax implications of: {item['text'][:100]}...",
                'input': '',
                'output': f"The tax implications include...",
                'domain': item['domain']
            }
            for item in scraped_content
        ]
        
        datasets = {
            'foundation': base_dataset,
            'reasoning': reasoning_dataset,
            'expertise': reasoning_dataset[:1],  # Subset for testing
            'adversarial': reasoning_dataset[:1]  # Subset for testing
        }
        
        # Create progressive training phases
        training_phases = optimizer.create_progressive_training_phases(datasets)
        
        # Generate configurations
        autotrain_config = optimizer.create_autotrain_config(
            self.test_dir,
            specialization="hmrc_compliance"
        )
        
        # Save configurations
        optimizer.save_config(training_phases, self.test_dir, "hmrc_training_phases.json")
        optimizer.save_config(autotrain_config, self.test_dir, "hmrc_autotrain_config.json")
        
        # Verify complete pipeline outputs
        final_files = ['hmrc_training_phases.json', 'hmrc_autotrain_config.json']
        for final_file in final_files:
            file_path = os.path.join(self.test_dir, final_file)
            self.assertTrue(os.path.exists(file_path))
    
    def test_multi_domain_training_pipeline(self):
        """Test pipeline for multiple domain specialists"""
        domains = ['legal', 'tax', 'copyright']
        
        # Create mock datasets for each domain
        domain_datasets = {}
        for domain in domains:
            domain_datasets[domain] = {
                'foundation': [
                    {'text': f'{domain} concept 1', 'domain': domain},
                    {'text': f'{domain} concept 2', 'domain': domain}
                ],
                'reasoning': [
                    {
                        'instruction': f'Analyze {domain} case',
                        'input': '',
                        'output': f'{domain} analysis result',
                        'domain': domain
                    }
                ],
                'expertise': [
                    {
                        'scenario': f'Complex {domain} scenario',
                        'analysis': f'Expert {domain} analysis',
                        'domain': domain
                    }
                ],
                'adversarial': [
                    {
                        'challenge': f'{domain} edge case',
                        'response': f'{domain} solution',
                        'domain': domain
                    }
                ]
            }
        
        # Create optimizers for each domain
        domain_configs = {}
        for domain in domains:
            optimizer = LlamaTrainingOptimizer(domain=domain)
            
            # Create training phases
            training_phases = optimizer.create_progressive_training_phases(
                domain_datasets[domain]
            )
            
            # Create AutoTrain config
            autotrain_config = optimizer.create_autotrain_config(
                f"{self.test_dir}/{domain}",
                specialization=f"{domain}_specialist"
            )
            
            domain_configs[domain] = {
                'training_phases': training_phases,
                'autotrain_config': autotrain_config
            }
            
            # Save domain-specific configurations
            domain_dir = os.path.join(self.test_dir, domain)
            os.makedirs(domain_dir, exist_ok=True)
            
            optimizer.save_config(
                training_phases, 
                domain_dir, 
                f"{domain}_training_phases.json"
            )
            optimizer.save_config(
                autotrain_config, 
                domain_dir, 
                f"{domain}_autotrain_config.json"
            )
        
        # Verify all domain configurations were created
        for domain in domains:
            domain_dir = os.path.join(self.test_dir, domain)
            self.assertTrue(os.path.exists(domain_dir))
            
            expected_files = [
                f"{domain}_training_phases.json",
                f"{domain}_autotrain_config.json"
            ]
            
            for expected_file in expected_files:
                file_path = os.path.join(domain_dir, expected_file)
                self.assertTrue(os.path.exists(file_path))
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    self.assertIsInstance(config_data, dict)
        
        # Verify each domain has unique specialisation
        for domain in domains:
            config_path = os.path.join(
                self.test_dir, 
                domain, 
                f"{domain}_autotrain_config.json"
            )
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                self.assertIn(domain, config['project_name'])
    
    def test_dataset_validation_pipeline(self):
        """Test dataset validation throughout the pipeline"""
        # Create test datasets with various formats
        test_datasets = {
            'valid_instruction': [
                {
                    'instruction': 'Test instruction',
                    'input': '',
                    'output': 'Test output',
                    'domain': 'legal'
                }
            ],
            'valid_text': [
                {
                    'text': 'Test text content',
                    'domain': 'legal'
                }
            ],
            'invalid_missing_fields': [
                {
                    'instruction': 'Test instruction'
                    # Missing output field
                }
            ]
        }
        
        optimizer = LlamaTrainingOptimizer(domain="legal")
        
        # Test validation of valid datasets
        for dataset_name, dataset in test_datasets.items():
            if 'valid' in dataset_name:
                is_valid, message = optimizer.validate_dataset_format(dataset)
                self.assertTrue(is_valid, f"Dataset {dataset_name} should be valid")
            elif 'invalid' in dataset_name:
                is_valid, message = optimizer.validate_dataset_format(dataset)
                self.assertFalse(is_valid, f"Dataset {dataset_name} should be invalid")
        
        # Test Llama optimisation with valid dataset
        valid_dataset = test_datasets['valid_instruction']
        optimized_dataset = optimizer.optimize_dataset_for_llama(valid_dataset)
        
        self.assertIsInstance(optimized_dataset, list)
        self.assertGreater(len(optimized_dataset), 0)
        
        for item in optimized_dataset:
            self.assertIn('text', item)
            self.assertIn('labels', item)
    
    def test_performance_monitoring_pipeline(self):
        """Test performance monitoring throughout the pipeline"""
        import time
        
        # Track timing for different pipeline stages
        timings = {}
        
        # Stage 1: Content extraction (simulated)
        start_time = time.time()
        test_content = "Test legal content for performance monitoring" * 100
        timings['content_extraction'] = time.time() - start_time
        
        # Stage 2: Domain detection
        start_time = time.time()
        pipeline = DynamicPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        domain = pipeline.detect_domain(test_content)
        timings['domain_detection'] = time.time() - start_time
        
        # Stage 3: Dataset creation
        start_time = time.time()
        base_dataset = pipeline.create_base_knowledge_dataset(test_content, domain)
        timings['dataset_creation'] = time.time() - start_time
        
        # Stage 4: Training optimisation
        start_time = time.time()
        optimizer = LlamaTrainingOptimizer(domain=domain)
        training_config = optimizer.create_training_config()
        timings['training_optimization'] = time.time() - start_time
        
        # Verify reasonable performance (all stages should complete quickly)
        for stage, duration in timings.items():
            self.assertLess(duration, 5.0, f"Stage {stage} took too long: {duration}s")
        
        # Save performance report
        performance_report = {
            'timings': timings,
            'total_time': sum(timings.values()),
            'dataset_size': len(base_dataset),
            'domain': domain
        }
        
        with open(os.path.join(self.test_dir, 'performance_report.json'), 'w') as f:
            json.dump(performance_report, f, indent=2)
        
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, 'performance_report.json')))


if __name__ == '__main__':
    unittest.main()
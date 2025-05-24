"""
Unit tests for dynamic_pipeline.py
Tests the dynamic dataset generation functionality.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import os
import tempfile
import shutil
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pipelines.dynamic_pipeline import DynamicDatasetPipeline


class TestDynamicPipeline(unittest.TestCase):
    """Test cases for DynamicDatasetPipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_url = "https://example.com/legal-content"
        self.mock_api_key = "test-api-key"
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_init_with_valid_parameters(self):
        """Test DynamicDatasetPipeline initialisation with valid parameters"""
        pipeline = DynamicDatasetPipeline(
            target_url=self.test_url,
            output_dir=self.test_dir
        )
        
        self.assertEqual(pipeline.target_url, self.test_url)
        self.assertEqual(str(pipeline.output_dir), self.test_dir)
        self.assertIsNotNone(pipeline.session)
    
    def test_init_creates_output_directory(self):
        """Test that initialisation creates output directory if it doesn't exist"""
        non_existent_dir = os.path.join(self.test_dir, "new_dir")
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=non_existent_dir
        )
        
        self.assertTrue(os.path.exists(non_existent_dir))
    
    @patch('requests.Session.get')
    def test_fetch_content_success(self, mock_get):
        """Test successful content fetching"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><body>Test content</body></html>"
        mock_response.headers = {'content-type': 'text/html'}
        mock_get.return_value = mock_response
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        content = pipeline.fetch_content(self.test_url)
        
        self.assertIsNotNone(content)
        self.assertIn("Test content", content)
        mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_fetch_content_failure(self, mock_get):
        """Test content fetching failure handling"""
        mock_get.side_effect = Exception("Network error")
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        content = pipeline.fetch_content(self.test_url)
        
        self.assertIsNone(content)
    
    def test_detect_domain_legal(self):
        """Test domain detection for legal content"""
        legal_content = "contract law statute regulation court case legal"
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        domain = pipeline.detect_domain(legal_content)
        
        self.assertEqual(domain, "legal")
    
    def test_detect_domain_tax(self):
        """Test domain detection for tax content"""
        tax_content = "HMRC tax deduction allowance income tax corporation tax"
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        domain = pipeline.detect_domain(tax_content)
        
        self.assertEqual(domain, "tax")
    
    def test_detect_domain_general(self):
        """Test domain detection for general content"""
        general_content = "general information about various topics"
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        domain = pipeline.detect_domain(general_content)
        
        self.assertEqual(domain, "general")
    
    def test_create_base_knowledge_dataset(self):
        """Test base knowledge dataset creation"""
        test_content = "The Consumer Rights Act 2015 protects consumers."
        domain = "legal"
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        dataset = pipeline.create_base_knowledge_dataset(test_content, domain)
        
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        
        for item in dataset:
            self.assertIn('text', item)
            self.assertIn('domain', item)
            self.assertEqual(item['domain'], domain)
    
    def test_create_reasoning_dataset(self):
        """Test reasoning pattern dataset creation"""
        test_content = "Legal precedent analysis requires examining case law."
        domain = "legal"
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        dataset = pipeline.create_reasoning_dataset(test_content, domain)
        
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        
        for item in dataset:
            self.assertIn('instruction', item)
            self.assertIn('input', item)
            self.assertIn('output', item)
            self.assertIn('domain', item)
    
    def test_create_expert_scenarios_dataset(self):
        """Test expert scenarios dataset creation"""
        test_content = "Complex legal cases require expert analysis."
        domain = "legal"
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        dataset = pipeline.create_expert_scenarios_dataset(test_content, domain)
        
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        
        for item in dataset:
            self.assertIn('scenario', item)
            self.assertIn('analysis', item)
            self.assertIn('domain', item)
            self.assertIn('complexity', item)
    
    def test_create_adversarial_dataset(self):
        """Test adversarial training dataset creation"""
        test_content = "Edge cases in legal reasoning."
        domain = "legal"
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        dataset = pipeline.create_adversarial_dataset(test_content, domain)
        
        self.assertIsInstance(dataset, list)
        self.assertGreater(len(dataset), 0)
        
        for item in dataset:
            self.assertIn('challenge', item)
            self.assertIn('response', item)
            self.assertIn('domain', item)
            self.assertIn('difficulty', item)
    
    def test_save_dataset(self):
        """Test dataset saving functionality"""
        test_dataset = [
            {"text": "test content", "domain": "legal"},
            {"text": "more content", "domain": "legal"}
        ]
        filename = "test_dataset.json"
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        pipeline.save_dataset(test_dataset, filename)
        
        saved_file = os.path.join(self.test_dir, filename)
        self.assertTrue(os.path.exists(saved_file))
        
        with open(saved_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
        
        self.assertEqual(loaded_data, test_dataset)
    
    def test_generate_llama_config(self):
        """Test Llama training configuration generation"""
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        config = pipeline.generate_llama_config("legal", 1000)
        
        self.assertIsInstance(config, dict)
        self.assertIn('model_name', config)
        self.assertIn('domain', config)
        self.assertIn('dataset_size', config)
        self.assertIn('training_config', config)
        self.assertEqual(config['domain'], "legal")
        self.assertEqual(config['dataset_size'], 1000)
    
    @patch('subprocess.run')
    @patch.object(DynamicDatasetPipeline, 'fetch_content')
    def test_run_pipeline_success(self, mock_fetch, mock_subprocess):
        """Test successful pipeline execution"""
        mock_fetch.return_value = "Legal content about contracts and statutes"
        mock_subprocess.return_value = Mock(returncode=0, stdout="Claude analysis result")
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        result = pipeline.run(self.test_url)
        
        self.assertTrue(result)
        mock_fetch.assert_called_once_with(self.test_url)
    
    @patch.object(DynamicDatasetPipeline, 'fetch_content')
    def test_run_pipeline_fetch_failure(self, mock_fetch):
        """Test pipeline execution with fetch failure"""
        mock_fetch.return_value = None
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        result = pipeline.run(self.test_url)
        
        self.assertFalse(result)


class TestDynamicPipelineIntegration(unittest.TestCase):
    """Integration tests for DynamicPipeline"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.mock_api_key = "test-api-key"
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_full_dataset_creation_workflow(self):
        """Test complete dataset creation workflow"""
        test_content = """
        The Consumer Rights Act 2015 is a UK statute that protects consumers.
        It covers contract law, sale of goods, and digital content rights.
        """
        
        pipeline = DynamicDatasetPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        # Test domain detection
        domain = pipeline.detect_domain(test_content)
        self.assertEqual(domain, "legal")
        
        # Test all dataset types creation
        base_dataset = pipeline.create_base_knowledge_dataset(test_content, domain)
        reasoning_dataset = pipeline.create_reasoning_dataset(test_content, domain)
        expert_dataset = pipeline.create_expert_scenarios_dataset(test_content, domain)
        adversarial_dataset = pipeline.create_adversarial_dataset(test_content, domain)
        
        # Verify datasets are created
        self.assertGreater(len(base_dataset), 0)
        self.assertGreater(len(reasoning_dataset), 0)
        self.assertGreater(len(expert_dataset), 0)
        self.assertGreater(len(adversarial_dataset), 0)
        
        # Test saving
        pipeline.save_dataset(base_dataset, "base_knowledge.json")
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "base_knowledge.json")))
        
        # Test config generation
        config = pipeline.generate_llama_config(domain, len(base_dataset))
        self.assertIn('training_config', config)


if __name__ == '__main__':
    unittest.main()
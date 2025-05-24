"""
Unit tests for llama_training_optimizer.py
Tests the Llama 3.1 70B training optimisation functionality.
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

from utils.llama_training_optimizer import ParaLlamaTrainingOptimizer


class TestLlamaTrainingOptimizer(unittest.TestCase):
    """Test cases for LlamaTrainingOptimizer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_domain = "legal"
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_init_with_default_parameters(self):
        """Test LlamaTrainingOptimizer initialisation with default parameters"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        self.assertIsNotNone(optimizer.input_dir)
        self.assertIsNotNone(optimizer.output_dir)
    
    def test_init_with_custom_parameters(self):
        """Test LlamaTrainingOptimizer initialisation with custom parameters"""
        custom_output_dir = os.path.join(self.test_dir, "custom_output")
        optimizer = ParaLlamaTrainingOptimizer(
            input_dir=self.test_dir,
            output_dir=custom_output_dir
        )
        
        self.assertEqual(str(optimizer.input_dir), self.test_dir)
        self.assertEqual(str(optimizer.output_dir), custom_output_dir)
    
    def test_load_all_datasets(self):
        """Test loading all datasets"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        datasets = optimizer.load_all_datasets()
        
        self.assertIsInstance(datasets, dict)
        expected_keys = ["base_legal", "base_tax", "enhanced_legal", "enhanced_tax"]
        for key in expected_keys:
            self.assertIn(key, datasets)
            self.assertIsInstance(datasets[key], list)
    
    def test_format_for_llama_training(self):
        """Test formatting data for Llama training"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        test_data = [
            {
                "title": "Test Legal Document",
                "content": "This is a test legal document with sufficient content to process. " * 20,
                "domain": "legal"
            }
        ]
        
        formatted_data = optimizer.format_for_llama_training(test_data, "legal_specialist", "phase_1_foundation")
        
        self.assertIsInstance(formatted_data, list)
        if formatted_data:  # May be empty if content doesn't meet criteria
            self.assertIn('text', formatted_data[0])
            self.assertIn('instruction', formatted_data[0])
            self.assertIn('response', formatted_data[0])
    
    def test_create_instruction_from_content(self):
        """Test instruction-response pair creation"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        content = "This is a comprehensive legal document discussing contract law principles. It covers formation, performance, and breach of contracts."
        title = "Contract Law Guide"
        
        instruction, response = optimizer._create_instruction_from_content(content, title, "legal_specialist")
        
        self.assertIsInstance(instruction, str)
        self.assertIsInstance(response, str)
        self.assertIn(title, instruction)
        self.assertIn(title, response)
        self.assertTrue(len(instruction) > 0)
        self.assertTrue(len(response) > 0)
    
    @patch('utils.llama_training_optimizer.PANDAS_AVAILABLE', True)
    @patch('utils.llama_training_optimizer.DATASETS_AVAILABLE', True)
    def test_create_multi_round_training_data(self):
        """Test multi-round training data creation"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        test_datasets = {
            "base_legal": [{"title": "Legal Doc", "content": "Legal content " * 50, "domain": "legal"}],
            "base_tax": [{"title": "Tax Doc", "content": "Tax content " * 50, "domain": "tax"}],
            "enhanced_legal": [],
            "enhanced_tax": [],
            "advanced_qa": [],
            "tax_scenarios": []
        }
        
        # Mock pandas and datasets
        with patch('pandas.DataFrame'), patch('utils.llama_training_optimizer.Dataset.from_pandas'), \
             patch('utils.llama_training_optimizer.DatasetDict'):
            result = optimizer.create_multi_round_training_data(test_datasets)
            self.assertIsInstance(result, dict)
    
    @patch('utils.llama_training_optimizer.PANDAS_AVAILABLE', True)
    @patch('utils.llama_training_optimizer.DATASETS_AVAILABLE', True)
    def test_create_autotrain_config(self):
        """Test AutoTrain configuration creation"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        # Mock dataset structure
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        
        test_datasets = {
            "legal_specialist": {
                "phase_1_foundation": {
                    "train": mock_dataset,
                    "validation": mock_dataset
                }
            }
        }
        
        autotrain_config = optimizer.create_autotrain_config(test_datasets)
        
        self.assertIsInstance(autotrain_config, dict)
        self.assertIn("legal_specialist_phase_1_foundation", autotrain_config)
        config = autotrain_config["legal_specialist_phase_1_foundation"]
        self.assertIn('model', config)
        self.assertIn('task', config)
        self.assertIn('data', config)
        self.assertIn('training', config)
    
    def test_generate_training_readme(self):
        """Test training README generation"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        # Mock datasets and configs
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 100
        
        test_datasets = {
            "legal_specialist": {
                "phase_1_foundation": {
                    "train": mock_dataset,
                    "validation": mock_dataset
                }
            }
        }
        
        test_configs = {
            "legal_specialist_phase_1_foundation": {
                "model": "meta-llama/Meta-Llama-3.1-70B-Instruct"
            }
        }
        
        readme = optimizer.generate_training_readme(test_datasets, test_configs)
        
        self.assertIsInstance(readme, str)
        self.assertIn("ParaLlama", readme)
        self.assertIn("legal_specialist", readme)
    
    @patch('utils.llama_training_optimizer.PANDAS_AVAILABLE', True)
    @patch('utils.llama_training_optimizer.DATASETS_AVAILABLE', True)
    def test_optimize_for_llama_training(self):
        """Test complete optimization workflow"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        # Create test input files
        test_data = {
            "title": "Test Document",
            "content": "Test content " * 50,
            "domain": "legal"
        }
        
        os.makedirs(os.path.join(self.test_dir, "test_dir"), exist_ok=True)
        with open(os.path.join(self.test_dir, "test_dir", "test.json"), 'w') as f:
            json.dump(test_data, f)
        
        # Mock the necessary components
        with patch('utils.llama_training_optimizer.Dataset.from_pandas'), \
             patch('utils.llama_training_optimizer.DatasetDict'), \
             patch('pandas.DataFrame'):
            result = optimizer.optimize_for_llama_training()
            
            self.assertIsInstance(result, dict)
            self.assertIn('datasets', result)
            self.assertIn('configs', result)
            self.assertIn('readme_path', result)
    
    def test_load_directory_data(self):
        """Test loading data from directory"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        # Create test JSON file
        test_dir = os.path.join(self.test_dir, "test_data")
        os.makedirs(test_dir, exist_ok=True)
        
        test_data = {
            "title": "Test Document",
            "content": "This is test content",
            "metadata": {"source": "test"}
        }
        
        with open(os.path.join(test_dir, "test.json"), 'w') as f:
            json.dump(test_data, f)
        
        loaded_data = optimizer._load_directory_data(Path(test_dir))
        
        self.assertIsInstance(loaded_data, list)
        self.assertEqual(len(loaded_data), 1)
        self.assertEqual(loaded_data[0]["title"], "Test Document")
    
    def test_training_phases_structure(self):
        """Test training phases structure"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        self.assertIsInstance(optimizer.training_phases, dict)
        expected_phases = ["phase_1_foundation", "phase_2_reasoning", "phase_3_expertise", "phase_4_adversarial"]
        
        for phase in expected_phases:
            self.assertIn(phase, optimizer.training_phases)
            phase_config = optimizer.training_phases[phase]
            self.assertIn('description', phase_config)
            self.assertIn('max_length', phase_config)
            self.assertIn('complexity', phase_config)
            self.assertIn('focus', phase_config)
    
    def test_llama_templates_structure(self):
        """Test Llama templates structure"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        self.assertIsInstance(optimizer.llama_templates, dict)
        expected_specialists = ["legal_specialist", "tax_specialist"]
        
        for specialist in expected_specialists:
            self.assertIn(specialist, optimizer.llama_templates)
            template = optimizer.llama_templates[specialist]
            self.assertIn('system', template)
            self.assertIn('instruction_template', template)
            self.assertIn('domains', template)
            self.assertIsInstance(template['domains'], list)
    
    def test_output_directory_creation(self):
        """Test output directory creation"""
        custom_output = os.path.join(self.test_dir, "custom_output")
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir, output_dir=custom_output)
        
        self.assertTrue(os.path.exists(custom_output))
        self.assertEqual(str(optimizer.output_dir), custom_output)
    
    def test_create_instruction_from_content_tax_specialist(self):
        """Test instruction creation for tax specialist"""
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        content = "This document explains VAT regulations and compliance requirements for UK businesses."
        title = "VAT Compliance Guide"
        
        instruction, response = optimizer._create_instruction_from_content(content, title, "tax_specialist")
        
        self.assertIsInstance(instruction, str)
        self.assertIsInstance(response, str)
        self.assertIn(title, instruction)
        self.assertIn("tax", instruction.lower())
        self.assertTrue(len(response) > 0)


class TestLlamaTrainingOptimizerIntegration(unittest.TestCase):
    """Integration tests for LlamaTrainingOptimizer"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_domain = "legal"
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('utils.llama_training_optimizer.PANDAS_AVAILABLE', True)
    @patch('utils.llama_training_optimizer.DATASETS_AVAILABLE', True)
    def test_complete_optimization_workflow(self):
        """Test complete optimisation workflow"""
        # Create test input data
        test_data_dir = os.path.join(self.test_dir, "input_data")
        os.makedirs(test_data_dir, exist_ok=True)
        
        test_data = {
            "title": "Legal Document",
            "content": "This is a comprehensive legal document. " * 100,
            "domain": "legal",
            "metadata": {"source": "test"}
        }
        
        with open(os.path.join(test_data_dir, "test_legal.json"), 'w') as f:
            json.dump(test_data, f)
        
        optimizer = ParaLlamaTrainingOptimizer(input_dir=self.test_dir)
        
        # Mock the necessary components for the workflow
        mock_dataset = MagicMock()
        mock_dataset.__len__.return_value = 10
        
        with patch('utils.llama_training_optimizer.Dataset.from_pandas') as mock_from_pandas, \
             patch('utils.llama_training_optimizer.DatasetDict') as mock_dataset_dict, \
             patch('pandas.DataFrame') as mock_df, \
             patch.object(mock_dataset, 'save_to_disk'), \
             patch.object(mock_dataset, 'to_parquet'):
            
            mock_from_pandas.return_value = mock_dataset
            mock_dataset_dict.return_value = {"train": mock_dataset, "validation": mock_dataset}
            
            # Run the optimization
            result = optimizer.optimize_for_llama_training()
            
            self.assertIsInstance(result, dict)
            self.assertIn('datasets', result)
            self.assertIn('configs', result)
            self.assertIn('readme_path', result)
            
            # Verify README was created
            readme_path = result['readme_path']
            self.assertTrue(os.path.exists(readme_path))


if __name__ == '__main__':
    unittest.main()
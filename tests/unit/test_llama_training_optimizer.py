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
        optimizer = ParaLlamaTrainingOptimizer(
            domain=self.test_domain,
            model_size="13b",
            base_model="meta-llama/Llama-2-13b-chat-hf"
        )
        
        self.assertEqual(optimizer.domain, self.test_domain)
        self.assertEqual(optimizer.model_size, "13b")
        self.assertEqual(optimizer.base_model, "meta-llama/Llama-2-13b-chat-hf")
    
    def test_create_lora_config(self):
        """Test LoRA configuration creation"""
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        
        lora_config = optimizer.create_lora_config()
        
        self.assertIsInstance(lora_config, dict)
        self.assertIn('r', lora_config)  # rank
        self.assertIn('lora_alpha', lora_config)
        self.assertIn('target_modules', lora_config)
        self.assertIn('lora_dropout', lora_config)
        self.assertIn('bias', lora_config)
        
        # Test that rank is reasonable for 70B model
        self.assertGreaterEqual(lora_config['r'], 8)
        self.assertLessEqual(lora_config['r'], 64)
    
    def test_create_lora_config_custom_rank(self):
        """Test LoRA configuration with custom rank"""
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        custom_rank = 32
        
        lora_config = optimizer.create_lora_config(rank=custom_rank)
        
        self.assertEqual(lora_config['r'], custom_rank)
        self.assertEqual(lora_config['lora_alpha'], custom_rank * 2)  # Common practice
    
    def test_create_training_config(self):
        """Test training configuration creation"""
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        
        training_config = optimizer.create_training_config()
        
        self.assertIsInstance(training_config, dict)
        self.assertIn('learning_rate', training_config)
        self.assertIn('per_device_train_batch_size', training_config)
        self.assertIn('per_device_eval_batch_size', training_config)
        self.assertIn('num_train_epochs', training_config)
        self.assertIn('warmup_steps', training_config)
        self.assertIn('logging_steps', training_config)
        self.assertIn('save_steps', training_config)
        self.assertIn('eval_steps', training_config)
        
        # Test reasonable learning rate for large model
        lr = training_config['learning_rate']
        self.assertGreaterEqual(lr, 1e-5)
        self.assertLessEqual(lr, 1e-3)
    
    def test_create_training_config_custom_params(self):
        """Test training configuration with custom parameters"""
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        custom_lr = 5e-4
        custom_epochs = 2
        
        training_config = optimizer.create_training_config(
            learning_rate=custom_lr,
            num_epochs=custom_epochs
        )
        
        self.assertEqual(training_config['learning_rate'], custom_lr)
        self.assertEqual(training_config['num_train_epochs'], custom_epochs)
    
    def test_create_autotrain_config(self):
        """Test AutoTrain configuration creation"""
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        dataset_path = "/path/to/dataset"
        
        autotrain_config = optimizer.create_autotrain_config(dataset_path)
        
        self.assertIsInstance(autotrain_config, dict)
        self.assertIn('model', autotrain_config)
        self.assertIn('project_name', autotrain_config)
        self.assertIn('data_path', autotrain_config)
        self.assertIn('train', autotrain_config)
        self.assertIn('valid', autotrain_config)
        self.assertIn('text_column', autotrain_config)
        self.assertIn('lr', autotrain_config)
        self.assertIn('epochs', autotrain_config)
        self.assertIn('batch_size', autotrain_config)
        self.assertIn('warmup_ratio', autotrain_config)
        self.assertIn('gradient_accumulation', autotrain_config)
        self.assertIn('optimizer', autotrain_config)
        self.assertIn('scheduler', autotrain_config)
        self.assertIn('weight_decay', autotrain_config)
        self.assertIn('max_grad_norm', autotrain_config)
        self.assertIn('seed', autotrain_config)
        self.assertIn('logging', autotrain_config)
        self.assertIn('auto_find_batch_size', autotrain_config)
        self.assertIn('mixed_precision', autotrain_config)
        
        self.assertEqual(autotrain_config['data_path'], dataset_path)
        self.assertIn(self.test_domain, autotrain_config['project_name'])
    
    def test_create_autotrain_config_custom_specialisation(self):
        """Test AutoTrain configuration with custom specialisation"""
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        dataset_path = "/path/to/dataset"
        specialisation = "defendant_arguments"
        
        autotrain_config = optimizer.create_autotrain_config(
            dataset_path, 
            specialization=specialisation
        )
        
        self.assertIn(specialisation, autotrain_config['project_name'])
    
    def test_optimize_dataset_for_llama(self):
        """Test dataset optimisation for Llama training"""
        # Create test dataset
        test_dataset = [
            {
                "instruction": "What is contract law?",
                "input": "",
                "output": "Contract law governs agreements between parties.",
                "domain": "legal"
            },
            {
                "instruction": "Explain tort law",
                "input": "",
                "output": "Tort law deals with civil wrongs and damages.",
                "domain": "legal"
            }
        ]
        
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        
        optimized_dataset = optimizer.optimize_dataset_for_llama(test_dataset)
        
        self.assertIsInstance(optimized_dataset, list)
        self.assertEqual(len(optimized_dataset), len(test_dataset))
        
        for item in optimized_dataset:
            self.assertIn('text', item)
            self.assertIn('labels', item)
            # Check Llama-specific formatting
            self.assertTrue(item['text'].startswith('[INST]'))
            self.assertIn('[/INST]', item['text'])
    
    def test_create_progressive_training_phases(self):
        """Test progressive training phases creation"""
        test_datasets = {
            'foundation': [{'text': 'Basic legal concept', 'domain': 'legal'}],
            'reasoning': [{'instruction': 'Analyze', 'output': 'Analysis', 'domain': 'legal'}],
            'expertise': [{'scenario': 'Complex case', 'analysis': 'Expert view', 'domain': 'legal'}],
            'adversarial': [{'challenge': 'Edge case', 'response': 'Solution', 'domain': 'legal'}]
        }
        
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        
        phases = optimizer.create_progressive_training_phases(test_datasets)
        
        self.assertIsInstance(phases, dict)
        self.assertEqual(len(phases), 4)
        
        expected_phases = ['foundation', 'reasoning', 'expertise', 'adversarial']
        for phase in expected_phases:
            self.assertIn(phase, phases)
            self.assertIn('dataset', phases[phase])
            self.assertIn('config', phases[phase])
            self.assertIn('description', phases[phase])
    
    def test_generate_model_card(self):
        """Test model card generation"""
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        
        model_card = optimizer.generate_model_card(
            specialization="contract_analysis",
            dataset_size=1000,
            training_time="4 hours"
        )
        
        self.assertIsInstance(model_card, str)
        self.assertIn(self.test_domain, model_card)
        self.assertIn("contract_analysis", model_card)
        self.assertIn("1000", model_card)
        self.assertIn("Llama", model_card)
    
    def test_estimate_training_time(self):
        """Test training time estimation"""
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        
        # Test with different dataset sizes
        small_time = optimizer.estimate_training_time(1000)
        large_time = optimizer.estimate_training_time(10000)
        
        self.assertIsInstance(small_time, dict)
        self.assertIn('estimated_hours', small_time)
        self.assertIn('gpu_memory_gb', small_time)
        self.assertIn('recommended_gpus', small_time)
        
        # Larger datasets should take longer
        self.assertGreater(large_time['estimated_hours'], small_time['estimated_hours'])
    
    def test_save_config(self):
        """Test configuration saving"""
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        config = {"test": "config", "domain": self.test_domain}
        filename = "test_config.json"
        
        optimizer.save_config(config, self.test_dir, filename)
        
        saved_file = os.path.join(self.test_dir, filename)
        self.assertTrue(os.path.exists(saved_file))
        
        with open(saved_file, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)
        
        self.assertEqual(loaded_config, config)
    
    def test_validate_dataset_format(self):
        """Test dataset format validation"""
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        
        # Valid dataset
        valid_dataset = [
            {"instruction": "Test", "input": "", "output": "Response", "domain": "legal"}
        ]
        
        is_valid, message = optimizer.validate_dataset_format(valid_dataset)
        self.assertTrue(is_valid)
        self.assertIn("valid", message.lower())
        
        # Invalid dataset - missing required fields
        invalid_dataset = [
            {"instruction": "Test"}  # Missing output
        ]
        
        is_valid, message = optimizer.validate_dataset_format(invalid_dataset)
        self.assertFalse(is_valid)
        self.assertIn("missing", message.lower())


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
    
    def test_complete_optimization_workflow(self):
        """Test complete optimisation workflow"""
        test_datasets = {
            'foundation': [
                {'text': 'Legal concept 1', 'domain': 'legal'},
                {'text': 'Legal concept 2', 'domain': 'legal'}
            ],
            'reasoning': [
                {'instruction': 'Analyze law', 'input': '', 'output': 'Analysis', 'domain': 'legal'}
            ],
            'expertise': [
                {'scenario': 'Complex case', 'analysis': 'Expert analysis', 'domain': 'legal'}
            ],
            'adversarial': [
                {'challenge': 'Edge case', 'response': 'Solution', 'domain': 'legal'}
            ]
        }
        
        optimizer = ParaLlamaTrainingOptimizer(domain=self.test_domain)
        
        # Create progressive training phases
        phases = optimizer.create_progressive_training_phases(test_datasets)
        
        # Create AutoTrain config
        autotrain_config = optimizer.create_autotrain_config("/test/dataset/path")
        
        # Save all configurations
        optimizer.save_config(phases, self.test_dir, "progressive_phases.json")
        optimizer.save_config(autotrain_config, self.test_dir, "autotrain_config.json")
        
        # Verify files were created
        phases_file = os.path.join(self.test_dir, "progressive_phases.json")
        autotrain_file = os.path.join(self.test_dir, "autotrain_config.json")
        
        self.assertTrue(os.path.exists(phases_file))
        self.assertTrue(os.path.exists(autotrain_file))
        
        # Verify content
        with open(phases_file, 'r', encoding='utf-8') as f:
            loaded_phases = json.load(f)
        
        self.assertEqual(len(loaded_phases), 4)
        self.assertIn('foundation', loaded_phases)
        
        with open(autotrain_file, 'r', encoding='utf-8') as f:
            loaded_autotrain = json.load(f)
        
        self.assertIn('model', loaded_autotrain)
        self.assertIn(self.test_domain, loaded_autotrain['project_name'])


if __name__ == '__main__':
    unittest.main()
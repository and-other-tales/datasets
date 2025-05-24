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
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
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
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=non_existent_dir
            )
            
            self.assertTrue(os.path.exists(non_existent_dir))
    
    @patch('requests.Session.get')
    def test_extract_content_from_url_success(self, mock_get):
        """Test successful content extraction"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = "<html><head><title>Test Title</title></head><body>Test content</body></html>"
        mock_response.headers = {'content-type': 'text/html'}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            content_data = pipeline.extract_content_from_url()
            
            self.assertIsNotNone(content_data)
            self.assertIn('content', content_data)
            self.assertIn("Test content", content_data['content'])
            self.assertEqual(content_data['title'], "Test Title")
            mock_get.assert_called_once()
    
    @patch('requests.Session.get')
    def test_extract_content_from_url_failure(self, mock_get):
        """Test content extraction failure handling"""
        mock_get.side_effect = Exception("Network error")
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            content_data = pipeline.extract_content_from_url()
            
            self.assertEqual(content_data, {})
    
    @patch.object(DynamicDatasetPipeline, 'analyze_content_with_claude')
    def test_analyze_domain_and_purpose(self, mock_analyze):
        """Test domain and purpose analysis"""
        mock_analyze.return_value = json.dumps({
            "domain_type": "legal",
            "specialization_focus": "contract law",
            "training_objectives": ["understand legal terms", "analyze contracts"],
            "content_analysis": "Legal content about contracts",
            "dataset_types_needed": ["instruction_following", "reasoning_chains"],
            "british_terminology": True,
            "professional_level": "expert",
            "key_concepts": ["contract", "law", "statute"],
            "recommended_phases": {
                "phase_1": "foundation knowledge",
                "phase_2": "reasoning development",
                "phase_3": "expert application",
                "phase_4": "adversarial scenarios"
            }
        })
        
        content_data = {
            'domain': 'example.com',
            'title': 'Legal Content',
            'content': 'contract law statute regulation'
        }
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            analysis = pipeline.analyze_domain_and_purpose(content_data)
            
            self.assertEqual(analysis['domain_type'], "legal")
            self.assertEqual(analysis['specialization_focus'], "contract law")
            self.assertTrue(analysis['british_terminology'])
    
    @patch.object(DynamicDatasetPipeline, 'analyze_content_with_claude')
    def test_generate_base_knowledge_dataset(self, mock_analyze):
        """Test base knowledge dataset generation"""
        mock_analyze.return_value = json.dumps([
            {
                "instruction": "What is a contract?",
                "input": "",
                "output": "A contract is a legally binding agreement",
                "domain": "legal",
                "complexity": "foundation",
                "concepts": ["contract", "agreement"]
            },
            {
                "instruction": "Explain breach of contract",
                "input": "",
                "output": "A breach occurs when one party fails to perform",
                "domain": "legal",
                "complexity": "foundation",
                "concepts": ["breach", "contract"]
            }
        ])
        
        content_data = {'content': 'Legal content about contracts'}
        analysis_data = {'domain_type': 'legal', 'specialization_focus': 'contract law'}
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            dataset = pipeline.generate_base_knowledge_dataset(content_data, analysis_data)
            
            self.assertIsInstance(dataset, list)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]['domain'], 'legal')
            self.assertEqual(dataset[0]['complexity'], 'foundation')
    
    @patch.object(DynamicDatasetPipeline, 'analyze_content_with_claude')
    def test_generate_reasoning_dataset(self, mock_analyze):
        """Test reasoning dataset generation"""
        mock_analyze.return_value = json.dumps([
            {
                "instruction": "Analyze this contract dispute",
                "input": "Party A claims breach, Party B denies",
                "output": "Step 1: Review terms\nStep 2: Analyze evidence\nStep 3: Conclusion",
                "domain": "legal",
                "complexity": "reasoning",
                "reasoning_type": "analysis",
                "steps": ["Review terms", "Analyze evidence", "Conclusion"]
            }
        ])
        
        content_data = {'content': 'Legal reasoning content'}
        analysis_data = {
            'domain_type': 'legal',
            'recommended_phases': {'phase_2': 'Multi-step analysis'}
        }
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            dataset = pipeline.generate_reasoning_dataset(content_data, analysis_data)
            
            self.assertIsInstance(dataset, list)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]['complexity'], 'reasoning')
    
    @patch.object(DynamicDatasetPipeline, 'analyze_content_with_claude')
    def test_generate_expert_scenarios(self, mock_analyze):
        """Test expert scenarios generation"""
        mock_analyze.return_value = json.dumps([
            {
                "instruction": "Provide expert advice on complex merger",
                "input": "Two companies with conflicting contracts",
                "output": "Expert analysis with recommendations",
                "domain": "legal",
                "complexity": "expert",
                "scenario_type": "consultation",
                "expertise_areas": ["corporate law", "mergers"]
            }
        ])
        
        content_data = {'content': 'Expert legal content'}
        analysis_data = {
            'domain_type': 'legal',
            'recommended_phases': {'phase_3': 'Expert application'},
            'professional_level': 'expert'
        }
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            dataset = pipeline.generate_expert_scenarios(content_data, analysis_data)
            
            self.assertIsInstance(dataset, list)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]['complexity'], 'expert')
    
    @patch.object(DynamicDatasetPipeline, 'analyze_content_with_claude')
    def test_generate_adversarial_scenarios(self, mock_analyze):
        """Test adversarial scenarios generation"""
        mock_analyze.return_value = json.dumps([
            {
                "instruction": "Challenge this legal position",
                "input": "Client claims absolute right to terminate",
                "output": "Robust response addressing the challenge",
                "domain": "legal",
                "complexity": "adversarial",
                "challenge_type": "argument_counter",
                "robustness_areas": ["counter-arguments", "edge cases"]
            }
        ])
        
        content_data = {'content': 'Adversarial legal content'}
        analysis_data = {
            'domain_type': 'legal',
            'recommended_phases': {'phase_4': 'Adversarial scenarios'}
        }
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            dataset = pipeline.generate_adversarial_scenarios(content_data, analysis_data)
            
            self.assertIsInstance(dataset, list)
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0]['complexity'], 'adversarial')
    
    def test_create_comprehensive_datasets(self):
        """Test comprehensive dataset creation"""
        all_examples = [
            {"instruction": "Q1", "output": "A1", "complexity": "foundation", "domain": "legal"},
            {"instruction": "Q2", "output": "A2", "complexity": "foundation", "domain": "legal"},
            {"instruction": "Q3", "output": "A3", "complexity": "reasoning", "domain": "legal"},
            {"instruction": "Q4", "output": "A4", "complexity": "expert", "domain": "legal"},
            {"instruction": "Q5", "output": "A5", "complexity": "adversarial", "domain": "legal"}
        ]
        
        analysis_data = {
            'domain_type': 'legal',
            'specialization_focus': 'contract law'
        }
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            dataset_stats = pipeline.create_comprehensive_datasets(analysis_data, all_examples)
            
            self.assertIn('datasets', dataset_stats)
            self.assertIn('metadata', dataset_stats)
            
            # Check phase files were created
            phase_1_file = os.path.join(self.test_dir, 'datasets', 'phase_1_foundation.json')
            self.assertTrue(os.path.exists(phase_1_file))
    
    def test_generate_training_config(self):
        """Test training configuration generation"""
        analysis_data = {
            'domain_type': 'legal',
            'specialization_focus': 'contract law'
        }
        
        dataset_stats = {
            "datasets": {
                "phase_1_foundation": {
                    "train": [{"instruction": "Q1", "output": "A1"}],
                    "validation": [],
                    "total_examples": 1
                }
            }
        }
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            config = pipeline.generate_training_config(analysis_data, dataset_stats)
            
            self.assertIn('model', config)
            self.assertIn('domain', config)
            self.assertIn('phases', config)
            self.assertEqual(config['domain'], 'legal')
            self.assertIn('phase_1_foundation', config['phases'])
    
    def test_save_and_load_datasets(self):
        """Test dataset saving and loading functionality"""
        test_data = {
            "metadata": {
                "domain": "legal",
                "total_examples": 2
            },
            "examples": [
                {"text": "test content", "domain": "legal"},
                {"text": "more content", "domain": "legal"}
            ]
        }
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            # Create datasets directory
            datasets_dir = os.path.join(self.test_dir, 'datasets')
            os.makedirs(datasets_dir, exist_ok=True)
            
            # Save file
            test_file = os.path.join(datasets_dir, 'test_dataset.json')
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f, indent=2)
            
            # Verify file exists and can be loaded
            self.assertTrue(os.path.exists(test_file))
            
            with open(test_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
            
            self.assertEqual(loaded_data, test_data)
    
    @patch.object(DynamicDatasetPipeline, 'extract_content_from_url')
    @patch.object(DynamicDatasetPipeline, 'analyze_domain_and_purpose')
    @patch.object(DynamicDatasetPipeline, 'generate_base_knowledge_dataset')
    @patch.object(DynamicDatasetPipeline, 'generate_reasoning_dataset')
    @patch.object(DynamicDatasetPipeline, 'generate_expert_scenarios')
    @patch.object(DynamicDatasetPipeline, 'generate_adversarial_scenarios')
    def test_run_dynamic_pipeline_success(self, mock_adv, mock_exp, mock_reas, mock_base, mock_analyze, mock_extract):
        """Test successful pipeline execution"""
        # Mock return values
        mock_extract.return_value = {
            'content': 'Legal content',
            'domain': 'example.com',
            'title': 'Legal Page'
        }
        
        mock_analyze.return_value = {
            'domain_type': 'legal',
            'specialization_focus': 'contract law'
        }
        
        mock_base.return_value = [{"instruction": "Q1", "output": "A1", "complexity": "foundation", "domain": "legal"}]
        mock_reas.return_value = [{"instruction": "Q2", "output": "A2", "complexity": "reasoning", "domain": "legal"}]
        mock_exp.return_value = [{"instruction": "Q3", "output": "A3", "complexity": "expert", "domain": "legal"}]
        mock_adv.return_value = [{"instruction": "Q4", "output": "A4", "complexity": "adversarial", "domain": "legal"}]
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            result = pipeline.run_dynamic_pipeline()
            
            self.assertEqual(result['status'], 'success')
            self.assertEqual(result['domain'], 'legal')
            self.assertEqual(result['total_examples'], 4)
            self.assertTrue(result['training_ready'])
    
    @patch.object(DynamicDatasetPipeline, 'extract_content_from_url')
    def test_run_dynamic_pipeline_extract_failure(self, mock_extract):
        """Test pipeline execution with extraction failure"""
        mock_extract.return_value = {}  # Empty dict indicates failure
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            result = pipeline.run_dynamic_pipeline()
            
            self.assertEqual(result['status'], 'failed')
            self.assertIn('error', result)
    
    def test_anthropic_client_initialization(self):
        """Test Anthropic client initialization"""
        # Test with API key
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            # The client initialization happens in __init__
            # Just verify the pipeline was created successfully
            self.assertIsNotNone(pipeline)
        
        # Test without API key
        with patch.dict(os.environ, {}, clear=True):
            with patch('logging.Logger.warning') as mock_warning:
                pipeline = DynamicDatasetPipeline(
                    target_url=self.test_url,
                    output_dir=self.test_dir
                )
                mock_warning.assert_called_with("ANTHROPIC_API_KEY not found in environment variables")


class TestDynamicPipelineIntegration(unittest.TestCase):
    """Integration tests for DynamicPipeline"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.test_url = "https://example.com/test"
        self.mock_api_key = "test-api-key"
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch.object(DynamicDatasetPipeline, 'analyze_content_with_claude')
    def test_full_dataset_creation_workflow(self, mock_analyze):
        """Test complete dataset creation workflow"""
        # Mock different responses for different calls
        mock_analyze.side_effect = [
            # For domain analysis
            json.dumps({
                "domain_type": "legal",
                "specialization_focus": "consumer rights",
                "training_objectives": ["understand consumer law"],
                "british_terminology": True,
                "professional_level": "expert",
                "recommended_phases": {
                    "phase_1": "foundation",
                    "phase_2": "reasoning",
                    "phase_3": "expert",
                    "phase_4": "adversarial"
                }
            }),
            # For base knowledge
            json.dumps([
                {"instruction": "Q1", "output": "A1", "complexity": "foundation", "domain": "legal"}
            ]),
            # For reasoning
            json.dumps([
                {"instruction": "Q2", "output": "A2", "complexity": "reasoning", "domain": "legal"}
            ]),
            # For expert
            json.dumps([
                {"instruction": "Q3", "output": "A3", "complexity": "expert", "domain": "legal"}
            ]),
            # For adversarial
            json.dumps([
                {"instruction": "Q4", "output": "A4", "complexity": "adversarial", "domain": "legal"}
            ])
        ]
        
        content_data = {
            'content': 'The Consumer Rights Act 2015 protects consumers.',
            'domain': 'example.com',
            'title': 'Consumer Rights'
        }
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': self.mock_api_key}):
            pipeline = DynamicDatasetPipeline(
                target_url=self.test_url,
                output_dir=self.test_dir
            )
            
            # Test domain analysis
            analysis_data = pipeline.analyze_domain_and_purpose(content_data)
            self.assertEqual(analysis_data['domain_type'], "legal")
            
            # Test all dataset types creation
            base_dataset = pipeline.generate_base_knowledge_dataset(content_data, analysis_data)
            reasoning_dataset = pipeline.generate_reasoning_dataset(content_data, analysis_data)
            expert_dataset = pipeline.generate_expert_scenarios(content_data, analysis_data)
            adversarial_dataset = pipeline.generate_adversarial_scenarios(content_data, analysis_data)
            
            # Verify datasets are created
            self.assertEqual(len(base_dataset), 1)
            self.assertEqual(len(reasoning_dataset), 1)
            self.assertEqual(len(expert_dataset), 1)
            self.assertEqual(len(adversarial_dataset), 1)
            
            # Test comprehensive dataset creation
            all_examples = base_dataset + reasoning_dataset + expert_dataset + adversarial_dataset
            dataset_stats = pipeline.create_comprehensive_datasets(analysis_data, all_examples)
            
            self.assertIn('datasets', dataset_stats)
            self.assertIn('metadata', dataset_stats)
            
            # Test config generation
            config = pipeline.generate_training_config(analysis_data, dataset_stats)
            self.assertIn('phases', config)


if __name__ == '__main__':
    unittest.main()
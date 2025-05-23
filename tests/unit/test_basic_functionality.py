"""
Basic functionality tests for othertales Datasets Tools
Simple tests to verify core functionality without complex mocking.
"""

import unittest
import os
import tempfile
import shutil
import sys

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))


class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_dynamic_pipeline_import(self):
        """Test that dynamic pipeline can be imported"""
        try:
            from pipelines.dynamic_pipeline import DynamicDatasetPipeline
            self.assertTrue(True, "DynamicDatasetPipeline imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import DynamicDatasetPipeline: {e}")
    
    def test_dynamic_pipeline_creation(self):
        """Test basic dynamic pipeline creation"""
        from pipelines.dynamic_pipeline import DynamicDatasetPipeline
        
        pipeline = DynamicDatasetPipeline(
            target_url="https://test.com",
            output_dir=self.test_dir
        )
        
        self.assertEqual(pipeline.target_url, "https://test.com")
        self.assertTrue(os.path.exists(self.test_dir))
    
    def test_hmrc_scraper_import(self):
        """Test that HMRC scraper can be imported"""
        try:
            from pipelines.hmrc_scraper import HMRCScraper
            self.assertTrue(True, "HMRCScraper imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import HMRCScraper: {e}")
    
    def test_hmrc_scraper_creation(self):
        """Test basic HMRC scraper creation"""
        from pipelines.hmrc_scraper import HMRCScraper
        
        scraper = HMRCScraper(output_dir=self.test_dir)
        
        self.assertEqual(str(scraper.output_dir), self.test_dir)
        self.assertEqual(scraper.base_url, "https://www.gov.uk")
    
    def test_llama_optimizer_import(self):
        """Test that Llama optimizer can be imported"""
        try:
            from utils.llama_training_optimizer import ParaLlamaTrainingOptimizer
            self.assertTrue(True, "ParaLlamaTrainingOptimizer imported successfully")
        except ImportError as e:
            self.fail(f"Failed to import ParaLlamaTrainingOptimizer: {e}")
    
    def test_llama_optimizer_creation(self):
        """Test basic Llama optimizer creation"""
        from utils.llama_training_optimizer import ParaLlamaTrainingOptimizer
        
        optimizer = ParaLlamaTrainingOptimizer(
            input_dir=self.test_dir,
            output_dir=self.test_dir
        )
        
        self.assertEqual(str(optimizer.input_dir), self.test_dir)
        self.assertEqual(str(optimizer.output_dir), self.test_dir)
    
    def test_directory_structure(self):
        """Test that required directories exist"""
        required_dirs = [
            'pipelines',
            'utils',
            'tests',
            'tests/unit',
            'tests/integration',
            'tests/performance'
        ]
        
        for dir_name in required_dirs:
            dir_path = os.path.join(os.path.dirname(__file__), '..', '..', dir_name)
            self.assertTrue(
                os.path.exists(dir_path), 
                f"Required directory {dir_name} does not exist"
            )
    
    def test_main_cli_import(self):
        """Test that main CLI can be imported"""
        try:
            import main
            self.assertTrue(hasattr(main, 'main'), "main function exists")
        except ImportError as e:
            self.fail(f"Failed to import main: {e}")
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists"""
        requirements_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'requirements.txt'
        )
        self.assertTrue(
            os.path.exists(requirements_path),
            "requirements.txt file does not exist"
        )
    
    def test_readme_file_exists(self):
        """Test that README.md exists"""
        readme_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 'README.md'
        )
        self.assertTrue(
            os.path.exists(readme_path),
            "README.md file does not exist"
        )
        
        # Check that it contains the updated branding
        with open(readme_path, 'r', encoding='utf-8') as f:
            readme_content = f.read()
        
        self.assertIn("othertales Datasets Tools", readme_content)
        self.assertIn("Dynamic Datasets Generation Framework", readme_content)


if __name__ == '__main__':
    unittest.main()
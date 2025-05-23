"""
Performance benchmark tests for othertales Datasets Tools
Tests system performance and scalability metrics.
"""

import unittest
import time
import os
import tempfile
import shutil
import json
from unittest.mock import Mock, patch
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from pipelines.dynamic_pipeline import DynamicPipeline
from pipelines.hmrc_scraper import HMRCScraper
from utils.llama_training_optimizer import LlamaTrainingOptimizer


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.mock_api_key = "test-api-key"
        self.performance_results = {}
        
    def tearDown(self):
        """Clean up performance test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        
        # Save performance results
        results_file = os.path.join(
            os.path.dirname(__file__), 
            'performance_results.json'
        )
        with open(results_file, 'w') as f:
            json.dump(self.performance_results, f, indent=2)
    
    def measure_time_and_memory(self, func, *args, **kwargs):
        """Measure execution time and memory usage of a function"""
        process = psutil.Process()
        
        # Initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Execute function with timing
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Final memory
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        return {
            'result': result,
            'execution_time': end_time - start_time,
            'memory_used_mb': memory_used,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory
        }
    
    def test_dynamic_pipeline_performance(self):
        """Test dynamic pipeline performance with various content sizes"""
        pipeline = DynamicPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        # Test with different content sizes
        content_sizes = [
            ('small', 'Legal content about contracts.' * 10),
            ('medium', 'Legal content about contracts and statutes.' * 100),
            ('large', 'Comprehensive legal content covering multiple areas.' * 1000)
        ]
        
        performance_data = {}
        
        for size_name, content in content_sizes:
            # Test domain detection performance
            metrics = self.measure_time_and_memory(
                pipeline.detect_domain, 
                content
            )
            
            performance_data[f'domain_detection_{size_name}'] = {
                'content_length': len(content),
                'execution_time': metrics['execution_time'],
                'memory_used_mb': metrics['memory_used_mb']
            }
            
            # Test dataset creation performance
            domain = metrics['result']
            
            for dataset_type in ['base_knowledge', 'reasoning', 'expert_scenarios', 'adversarial']:
                method_name = f'create_{dataset_type}_dataset'
                if hasattr(pipeline, method_name):
                    method = getattr(pipeline, method_name)
                    
                    metrics = self.measure_time_and_memory(
                        method, 
                        content, 
                        domain
                    )
                    
                    performance_data[f'{dataset_type}_{size_name}'] = {
                        'content_length': len(content),
                        'dataset_size': len(metrics['result']),
                        'execution_time': metrics['execution_time'],
                        'memory_used_mb': metrics['memory_used_mb'],
                        'items_per_second': len(metrics['result']) / max(metrics['execution_time'], 0.001)
                    }
        
        self.performance_results['dynamic_pipeline'] = performance_data
        
        # Performance assertions
        for size_name, _ in content_sizes:
            domain_detection_time = performance_data[f'domain_detection_{size_name}']['execution_time']
            self.assertLess(domain_detection_time, 1.0, f"Domain detection for {size_name} content too slow")
            
            base_knowledge_time = performance_data[f'base_knowledge_{size_name}']['execution_time']
            self.assertLess(base_knowledge_time, 5.0, f"Base knowledge creation for {size_name} content too slow")
    
    @patch('requests.Session.get')
    def test_hmrc_scraper_performance(self, mock_get):
        """Test HMRC scraper performance and scalability"""
        # Mock HTTP responses
        def mock_response(url):
            response = Mock()
            response.status_code = 200
            
            if 'api/content' in url:
                response.json.return_value = {
                    'title': 'Test Tax Document',
                    'description': 'Test description',
                    'details': {'body': '<p>Test tax content</p>'}
                }
            else:
                response.text = '<html><body><h1>Test</h1><p>Test content</p></body></html>'
            
            return response
        
        mock_get.side_effect = lambda url, **kwargs: mock_response(url)
        
        scraper = HMRCScraper(output_dir=self.test_dir)
        
        # Test API vs HTML performance
        test_urls = [
            f"https://www.gov.uk/test-page-{i}" 
            for i in range(10)
        ]
        
        # Test API extraction performance
        api_metrics = self.measure_time_and_memory(
            self._batch_api_extraction,
            scraper,
            test_urls
        )
        
        # Test HTML extraction performance (with API disabled)
        scraper.use_content_api = False
        html_metrics = self.measure_time_and_memory(
            self._batch_html_extraction,
            scraper,
            test_urls
        )
        
        performance_data = {
            'api_extraction': {
                'total_urls': len(test_urls),
                'execution_time': api_metrics['execution_time'],
                'memory_used_mb': api_metrics['memory_used_mb'],
                'urls_per_second': len(test_urls) / max(api_metrics['execution_time'], 0.001)
            },
            'html_extraction': {
                'total_urls': len(test_urls),
                'execution_time': html_metrics['execution_time'],
                'memory_used_mb': html_metrics['memory_used_mb'],
                'urls_per_second': len(test_urls) / max(html_metrics['execution_time'], 0.001)
            }
        }
        
        # Calculate performance improvement
        if html_metrics['execution_time'] > 0:
            performance_data['api_speedup'] = html_metrics['execution_time'] / api_metrics['execution_time']
        
        self.performance_results['hmrc_scraper'] = performance_data
        
        # Performance assertions
        self.assertGreater(
            performance_data['api_speedup'], 
            2.0, 
            "Content API should be at least 2x faster than HTML scraping"
        )
        
        self.assertGreater(
            performance_data['api_extraction']['urls_per_second'],
            5.0,
            "API extraction should process at least 5 URLs per second"
        )
    
    def _batch_api_extraction(self, scraper, urls):
        """Helper method for batch API extraction"""
        results = []
        for url in urls:
            api_url = scraper.get_api_url(url)
            if api_url:
                content = scraper.extract_content_from_api(api_url)
                results.append(content)
        return results
    
    def _batch_html_extraction(self, scraper, urls):
        """Helper method for batch HTML extraction"""
        results = []
        for url in urls:
            content = scraper.extract_content_from_html(url)
            results.append(content)
        return results
    
    def test_llama_training_optimizer_performance(self):
        """Test Llama training optimizer performance with large datasets"""
        optimizer = LlamaTrainingOptimizer(domain="legal")
        
        # Create datasets of varying sizes
        dataset_sizes = [100, 1000, 5000]
        performance_data = {}
        
        for size in dataset_sizes:
            # Create test dataset
            test_dataset = [
                {
                    'instruction': f'Legal question {i}',
                    'input': '',
                    'output': f'Legal answer {i}',
                    'domain': 'legal'
                }
                for i in range(size)
            ]
            
            # Test dataset optimisation performance
            metrics = self.measure_time_and_memory(
                optimizer.optimize_dataset_for_llama,
                test_dataset
            )
            
            # Test configuration creation performance
            config_metrics = self.measure_time_and_memory(
                optimizer.create_autotrain_config,
                self.test_dir
            )
            
            performance_data[f'dataset_size_{size}'] = {
                'dataset_size': size,
                'optimization_time': metrics['execution_time'],
                'optimization_memory_mb': metrics['memory_used_mb'],
                'items_per_second': size / max(metrics['execution_time'], 0.001),
                'config_creation_time': config_metrics['execution_time'],
                'config_memory_mb': config_metrics['memory_used_mb']
            }
        
        self.performance_results['llama_optimizer'] = performance_data
        
        # Performance assertions
        for size in dataset_sizes:
            data = performance_data[f'dataset_size_{size}']
            
            # Should process at least 100 items per second
            self.assertGreater(
                data['items_per_second'], 
                100, 
                f"Optimization too slow for {size} items: {data['items_per_second']} items/sec"
            )
            
            # Config creation should be fast regardless of dataset size
            self.assertLess(
                data['config_creation_time'], 
                1.0, 
                f"Config creation too slow for {size} items"
            )
    
    def test_concurrent_processing_performance(self):
        """Test performance under concurrent processing load"""
        pipeline = DynamicPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        # Test content for concurrent processing
        test_contents = [
            f"Legal content number {i} about contracts and regulations." * 50
            for i in range(10)
        ]
        
        # Sequential processing
        start_time = time.time()
        sequential_results = []
        for content in test_contents:
            domain = pipeline.detect_domain(content)
            dataset = pipeline.create_base_knowledge_dataset(content, domain)
            sequential_results.append(len(dataset))
        sequential_time = time.time() - start_time
        
        # Concurrent processing
        start_time = time.time()
        concurrent_results = []
        
        def process_content(content):
            domain = pipeline.detect_domain(content)
            dataset = pipeline.create_base_knowledge_dataset(content, domain)
            return len(dataset)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_content, content) for content in test_contents]
            concurrent_results = [future.result() for future in futures]
        
        concurrent_time = time.time() - start_time
        
        performance_data = {
            'sequential_processing': {
                'total_items': len(test_contents),
                'execution_time': sequential_time,
                'items_per_second': len(test_contents) / sequential_time
            },
            'concurrent_processing': {
                'total_items': len(test_contents),
                'execution_time': concurrent_time,
                'items_per_second': len(test_contents) / concurrent_time,
                'max_workers': 4
            }
        }
        
        # Calculate speedup
        if sequential_time > 0:
            performance_data['concurrency_speedup'] = sequential_time / concurrent_time
        
        self.performance_results['concurrent_processing'] = performance_data
        
        # Verify results are consistent
        self.assertEqual(len(sequential_results), len(concurrent_results))
        
        # Concurrent processing should be faster (allowing for some overhead)
        if performance_data.get('concurrency_speedup', 0) > 0:
            self.assertGreaterEqual(
                performance_data['concurrency_speedup'],
                1.5,
                "Concurrent processing should provide at least 1.5x speedup"
            )
    
    def test_memory_usage_scalability(self):
        """Test memory usage scalability with increasing dataset sizes"""
        optimizer = LlamaTrainingOptimizer(domain="legal")
        
        # Test memory usage with increasing dataset sizes
        dataset_sizes = [100, 500, 1000, 2000]
        memory_data = {}
        
        for size in dataset_sizes:
            # Create test dataset
            test_dataset = [
                {
                    'instruction': f'Question {i}' * 10,  # Longer text
                    'input': '',
                    'output': f'Answer {i}' * 20,  # Even longer text
                    'domain': 'legal'
                }
                for i in range(size)
            ]
            
            # Measure memory usage during optimization
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            optimized_dataset = optimizer.optimize_dataset_for_llama(test_dataset)
            
            peak_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = peak_memory - initial_memory
            
            memory_data[f'dataset_size_{size}'] = {
                'dataset_size': size,
                'initial_memory_mb': initial_memory,
                'peak_memory_mb': peak_memory,
                'memory_used_mb': memory_used,
                'memory_per_item_kb': (memory_used * 1024) / size if size > 0 else 0
            }
            
            # Clean up to prevent memory accumulation
            del test_dataset
            del optimized_dataset
        
        self.performance_results['memory_scalability'] = memory_data
        
        # Memory usage should scale reasonably
        for size in dataset_sizes:
            data = memory_data[f'dataset_size_{size}']
            
            # Memory per item shouldn't exceed reasonable limits
            self.assertLess(
                data['memory_per_item_kb'], 
                100, 
                f"Memory usage per item too high for {size} items: {data['memory_per_item_kb']} KB/item"
            )
    
    def test_disk_io_performance(self):
        """Test disk I/O performance for large datasets"""
        pipeline = DynamicPipeline(
            anthropic_api_key=self.mock_api_key,
            output_dir=self.test_dir
        )
        
        # Create large test dataset
        large_dataset = [
            {
                'text': f'Large text content item {i}' * 100,
                'domain': 'legal',
                'metadata': {'item_id': i, 'source': 'test'}
            }
            for i in range(1000)
        ]
        
        # Test saving performance
        save_metrics = self.measure_time_and_memory(
            pipeline.save_dataset,
            large_dataset,
            'large_test_dataset.json'
        )
        
        # Test loading performance
        saved_file = os.path.join(self.test_dir, 'large_test_dataset.json')
        self.assertTrue(os.path.exists(saved_file))
        
        def load_dataset(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        load_metrics = self.measure_time_and_memory(
            load_dataset,
            saved_file
        )
        
        # Get file size
        file_size_mb = os.path.getsize(saved_file) / 1024 / 1024
        
        performance_data = {
            'dataset_items': len(large_dataset),
            'file_size_mb': file_size_mb,
            'save_time': save_metrics['execution_time'],
            'save_memory_mb': save_metrics['memory_used_mb'],
            'load_time': load_metrics['execution_time'],
            'load_memory_mb': load_metrics['memory_used_mb'],
            'save_throughput_mb_s': file_size_mb / max(save_metrics['execution_time'], 0.001),
            'load_throughput_mb_s': file_size_mb / max(load_metrics['execution_time'], 0.001)
        }
        
        self.performance_results['disk_io'] = performance_data
        
        # Performance assertions
        self.assertGreater(
            performance_data['save_throughput_mb_s'],
            10,
            f"Save throughput too low: {performance_data['save_throughput_mb_s']} MB/s"
        )
        
        self.assertGreater(
            performance_data['load_throughput_mb_s'],
            20,
            f"Load throughput too low: {performance_data['load_throughput_mb_s']} MB/s"
        )


if __name__ == '__main__':
    unittest.main()
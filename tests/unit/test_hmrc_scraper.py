"""
Unit tests for hmrc_scraper.py
Tests the HMRC tax documentation scraper with Content API integration.
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

from pipelines.hmrc_scraper import HMRCScraper


class TestHMRCScraper(unittest.TestCase):
    """Test cases for HMRCScraper class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.base_url = "https://www.gov.uk"
        
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_init_with_default_parameters(self):
        """Test HMRCScraper initialisation with default parameters"""
        scraper = HMRCScraper()
        
        self.assertEqual(scraper.base_url, "https://www.gov.uk")
        self.assertIsNotNone(scraper.session)
        self.assertIsNotNone(scraper.output_dir)
    
    def test_init_with_custom_parameters(self):
        """Test HMRCScraper initialisation with custom parameters"""
        scraper = HMRCScraper(
            output_dir=self.test_dir
        )
        
        self.assertEqual(scraper.base_url, "https://www.gov.uk")
        self.assertEqual(str(scraper.output_dir), self.test_dir)
    
    def test_get_api_url(self):
        """Test API URL generation"""
        scraper = HMRCScraper()
        page_url = "https://www.gov.uk/tax-codes"
        
        api_url = scraper.get_api_url(page_url)
        
        expected_url = "https://www.gov.uk/api/content/tax-codes"
        self.assertEqual(api_url, expected_url)
    
    def test_get_api_url_with_query_params(self):
        """Test API URL generation with query parameters"""
        scraper = HMRCScraper()
        page_url = "https://www.gov.uk/tax-codes?param=value"
        
        api_url = scraper.get_api_url(page_url)
        
        expected_url = "https://www.gov.uk/api/content/tax-codes"
        self.assertEqual(api_url, expected_url)
    
    def test_get_api_url_invalid_domain(self):
        """Test API URL generation with invalid domain"""
        scraper = HMRCScraper()
        page_url = "https://example.com/some-page"
        
        api_url = scraper.get_api_url(page_url)
        
        # The method doesn't validate domain, it just converts the path
        self.assertEqual(api_url, "https://www.gov.uk/api/content/some-page")
    
    @patch('requests.Session.get')
    def test_extract_content_from_api_success(self, mock_get):
        """Test successful content extraction from API"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'title': 'Tax Codes',
            'description': 'Understanding your tax code',
            'details': {
                'body': '<p>Tax code information</p>'
            }
        }
        mock_get.return_value = mock_response
        
        scraper = HMRCScraper()
        api_url = "https://www.gov.uk/api/content/tax-codes"
        
        content = scraper.extract_content_from_api(api_url)
        
        self.assertIsNotNone(content)
        self.assertIn('metadata', content)
        self.assertIn('content', content)
        self.assertEqual(content['metadata']['title'], 'Tax Codes')
        self.assertEqual(content['content'], 'Tax code information')
    
    @patch('requests.Session.get')
    def test_extract_content_from_api_failure(self, mock_get):
        """Test content extraction from API failure"""
        mock_get.side_effect = Exception("API error")
        
        scraper = HMRCScraper()
        api_url = "https://www.gov.uk/api/content/tax-codes"
        
        content = scraper.extract_content_from_api(api_url)
        
        self.assertIsNone(content)
    
    @patch('requests.Session.get')
    def test_extract_content_from_html_success(self, mock_get):
        """Test successful content extraction from HTML"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <head><title>Tax Information</title></head>
            <body>
                <h1>Tax Codes</h1>
                <p>Your tax code information</p>
            </body>
        </html>
        """
        mock_get.return_value = mock_response
        
        scraper = HMRCScraper()
        page_url = "https://www.gov.uk/tax-codes"
        
        content = scraper.extract_content_from_html(page_url)
        
        self.assertIsNotNone(content)
        self.assertIn('metadata', content)
        self.assertIn('content', content)
        self.assertEqual(content['metadata']['title'], 'Tax Codes')
        self.assertIn('Tax Codes', content['content'])
    
    @patch('requests.Session.get')
    def test_extract_content_from_html_failure(self, mock_get):
        """Test content extraction from HTML failure"""
        mock_get.side_effect = Exception("Network error")
        
        scraper = HMRCScraper()
        page_url = "https://www.gov.uk/tax-codes"
        
        content = scraper.extract_content_from_html(page_url)
        
        self.assertIsNone(content)
    
    def test_is_tax_related(self):
        """Test tax relevance detection"""
        scraper = HMRCScraper()
        
        # Test positive cases
        self.assertTrue(scraper.is_tax_related("Income Tax Guide"))
        self.assertTrue(scraper.is_tax_related("VAT Registration"))
        self.assertTrue(scraper.is_tax_related("Corporation Tax", "Information about company taxes"))
        
        # Test negative cases
        self.assertFalse(scraper.is_tax_related("Driving License"))
        self.assertFalse(scraper.is_tax_related("Passport Application"))
    
    def test_extract_document_content(self):
        """Test document content extraction wrapper"""
        scraper = HMRCScraper()
        
        # Test with invalid URL
        content = scraper.extract_document_content("not-a-url")
        self.assertIsNone(content)
    
    def test_directory_structure(self):
        """Test that required directories are created"""
        scraper = HMRCScraper(output_dir=self.test_dir)
        
        # Check that subdirectories are created
        self.assertTrue(scraper.text_dir.exists())
        self.assertTrue(scraper.html_dir.exists())
        self.assertTrue(scraper.metadata_dir.exists())
        self.assertTrue(scraper.forms_dir.exists())
        self.assertTrue(scraper.enhanced_metadata_dir.exists())
    
    def test_tracking_initialization(self):
        """Test that tracking sets are initialized correctly"""
        scraper = HMRCScraper()
        
        self.assertIsInstance(scraper.discovered_urls, set)
        self.assertIsInstance(scraper.downloaded_urls, set)
        self.assertIsInstance(scraper.failed_urls, set)
        self.assertEqual(len(scraper.discovered_urls), 0)
        self.assertEqual(len(scraper.downloaded_urls), 0)
        self.assertEqual(len(scraper.failed_urls), 0)


class TestHMRCScraperIntegration(unittest.TestCase):
    """Integration tests for HMRCScraper"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_scraper_initialization_with_output_dir(self):
        """Test scraper initialization creates proper directory structure"""
        scraper = HMRCScraper(output_dir=self.test_dir)
        
        # Check that the scraper was initialized correctly
        self.assertEqual(str(scraper.output_dir), self.test_dir)
        self.assertIsNotNone(scraper.session)
        self.assertIsNotNone(scraper.hmrc_processor)
        
        # Check that tracking structures are initialized
        self.assertIsInstance(scraper.discovered_urls, set)
        self.assertIsInstance(scraper.downloaded_urls, set)
        self.assertIsInstance(scraper.failed_urls, set)


if __name__ == '__main__':
    unittest.main()
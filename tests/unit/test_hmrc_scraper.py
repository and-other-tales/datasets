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
        self.assertEqual(scraper.content_api_base, "https://www.gov.uk/api/content")
        self.assertTrue(scraper.use_content_api)
        self.assertIsNotNone(scraper.session)
    
    def test_init_with_custom_parameters(self):
        """Test HMRCScraper initialisation with custom parameters"""
        scraper = HMRCScraper(
            base_url=self.base_url,
            output_dir=self.test_dir,
            use_content_api=False
        )
        
        self.assertEqual(scraper.base_url, self.base_url)
        self.assertEqual(scraper.output_dir, self.test_dir)
        self.assertFalse(scraper.use_content_api)
    
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
        
        self.assertIsNone(api_url)
    
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
        self.assertIn('title', content)
        self.assertIn('description', content)
        self.assertIn('body', content)
        self.assertEqual(content['title'], 'Tax Codes')
    
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
        self.assertIn('title', content)
        self.assertIn('body', content)
        self.assertIn('Tax Codes', content['body'])
    
    @patch('requests.Session.get')
    def test_extract_content_from_html_failure(self, mock_get):
        """Test content extraction from HTML failure"""
        mock_get.side_effect = Exception("Network error")
        
        scraper = HMRCScraper()
        page_url = "https://www.gov.uk/tax-codes"
        
        content = scraper.extract_content_from_html(page_url)
        
        self.assertIsNone(content)
    
    @patch.object(HMRCScraper, 'extract_content_from_api')
    @patch.object(HMRCScraper, 'extract_content_from_html')
    def test_scrape_with_api_fallback_api_success(self, mock_html, mock_api):
        """Test scraping with API success (no fallback needed)"""
        mock_api.return_value = {'title': 'Test', 'body': 'API content'}
        
        scraper = HMRCScraper()
        page_url = "https://www.gov.uk/tax-codes"
        
        content = scraper.scrape_with_api_fallback(page_url)
        
        self.assertIsNotNone(content)
        self.assertEqual(content['body'], 'API content')
        mock_api.assert_called_once()
        mock_html.assert_not_called()
    
    @patch.object(HMRCScraper, 'extract_content_from_api')
    @patch.object(HMRCScraper, 'extract_content_from_html')
    def test_scrape_with_api_fallback_api_failure(self, mock_html, mock_api):
        """Test scraping with API failure (fallback to HTML)"""
        mock_api.return_value = None
        mock_html.return_value = {'title': 'Test', 'body': 'HTML content'}
        
        scraper = HMRCScraper()
        page_url = "https://www.gov.uk/tax-codes"
        
        content = scraper.scrape_with_api_fallback(page_url)
        
        self.assertIsNotNone(content)
        self.assertEqual(content['body'], 'HTML content')
        mock_api.assert_called_once()
        mock_html.assert_called_once()
    
    @patch.object(HMRCScraper, 'extract_content_from_html')
    def test_scrape_with_api_fallback_api_disabled(self, mock_html):
        """Test scraping with API disabled (direct HTML)"""
        mock_html.return_value = {'title': 'Test', 'body': 'HTML content'}
        
        scraper = HMRCScraper(use_content_api=False)
        page_url = "https://www.gov.uk/tax-codes"
        
        content = scraper.scrape_with_api_fallback(page_url)
        
        self.assertIsNotNone(content)
        self.assertEqual(content['body'], 'HTML content')
        mock_html.assert_called_once()
    
    def test_save_content(self):
        """Test content saving functionality"""
        scraper = HMRCScraper(output_dir=self.test_dir)
        content = {
            'title': 'Test Document',
            'body': 'Test content',
            'metadata': {'source': 'test'}
        }
        filename = "test_document"
        
        scraper.save_content(content, filename)
        
        # Check HTML file
        html_file = os.path.join(self.test_dir, 'html', f"{filename}.html")
        self.assertTrue(os.path.exists(html_file))
        
        # Check text file
        text_file = os.path.join(self.test_dir, 'text', f"{filename}.txt")
        self.assertTrue(os.path.exists(text_file))
        
        # Check metadata file
        metadata_file = os.path.join(self.test_dir, 'metadata', f"{filename}.json")
        self.assertTrue(os.path.exists(metadata_file))
        
        # Verify content
        with open(text_file, 'r', encoding='utf-8') as f:
            saved_text = f.read()
        self.assertIn('Test content', saved_text)
        
        with open(metadata_file, 'r', encoding='utf-8') as f:
            saved_metadata = json.load(f)
        self.assertEqual(saved_metadata['title'], 'Test Document')
    
    @patch('requests.Session.get')
    def test_discover_hmrc_pages(self, mock_get):
        """Test HMRC pages discovery"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """
        <html>
            <body>
                <a href="/income-tax">Income Tax</a>
                <a href="/corporation-tax">Corporation Tax</a>
                <a href="/vat">VAT</a>
                <a href="https://external.com">External Link</a>
            </body>
        </html>
        """
        mock_get.return_value = mock_response
        
        scraper = HMRCScraper()
        
        pages = scraper.discover_hmrc_pages()
        
        self.assertIsInstance(pages, list)
        self.assertGreater(len(pages), 0)
        
        # Check that only gov.uk URLs are included
        for page in pages:
            self.assertTrue(page.startswith('https://www.gov.uk/'))
    
    def test_is_hmrc_relevant(self):
        """Test HMRC relevance detection"""
        scraper = HMRCScraper()
        
        # Test relevant content
        relevant_content = "income tax corporation tax HMRC self assessment"
        self.assertTrue(scraper.is_hmrc_relevant(relevant_content))
        
        # Test irrelevant content
        irrelevant_content = "weather forecast sports news entertainment"
        self.assertFalse(scraper.is_hmrc_relevant(irrelevant_content))
    
    def test_get_safe_filename(self):
        """Test safe filename generation"""
        scraper = HMRCScraper()
        
        # Test with special characters
        unsafe_title = "Tax: Codes & Rates (2023/24)"
        safe_filename = scraper.get_safe_filename(unsafe_title)
        
        self.assertNotIn(':', safe_filename)
        self.assertNotIn('&', safe_filename)
        self.assertNotIn('/', safe_filename)
        self.assertNotIn('(', safe_filename)
        self.assertNotIn(')', safe_filename)


class TestHMRCScraperIntegration(unittest.TestCase):
    """Integration tests for HMRCScraper"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch.object(HMRCScraper, 'discover_hmrc_pages')
    @patch.object(HMRCScraper, 'scrape_with_api_fallback')
    def test_full_scraping_workflow(self, mock_scrape, mock_discover):
        """Test complete scraping workflow"""
        # Mock discovered pages
        mock_discover.return_value = [
            "https://www.gov.uk/income-tax",
            "https://www.gov.uk/corporation-tax"
        ]
        
        # Mock scraped content
        mock_scrape.side_effect = [
            {'title': 'Income Tax', 'body': 'Income tax information'},
            {'title': 'Corporation Tax', 'body': 'Corporation tax information'}
        ]
        
        scraper = HMRCScraper(output_dir=self.test_dir)
        
        # Run limited scraping
        results = scraper.run_scraping(max_documents=2)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(mock_discover.call_count, 1)
        self.assertEqual(mock_scrape.call_count, 2)
        
        # Check that files were saved
        html_dir = os.path.join(self.test_dir, 'html')
        self.assertTrue(os.path.exists(html_dir))
        self.assertGreater(len(os.listdir(html_dir)), 0)


if __name__ == '__main__':
    unittest.main()
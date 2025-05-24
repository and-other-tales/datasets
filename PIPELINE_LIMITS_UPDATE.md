# Pipeline Limits Update Summary

## Overview
Updated all pipelines to remove artificial limitations and ensure they can process ALL available data by default, with optional command-line limits.

## Changes Made

### 1. HMRC Scraper (`pipelines/hmrc_scraper.py`)
- Added `max_batches` parameter to `discover_via_search_api()` and `run_comprehensive_discovery()`
- Added progress callback mechanism for UI updates
- Updated main.py to support `--quick` mode (20 batches, ~2000 docs) and full mode (ALL docs)
- Added `--no-curses` option for command line execution

### 2. BAILII Scraper (`pipelines/bailii_scraper.py`)
- Changed hardcoded `max_cases = 1000` to `max_cases = None` (process ALL)
- Added proper argument parsing with `--max-documents` option
- Now processes ALL discovered cases by default unless limited via command line

### 3. Housing BAILII Scraper (`pipelines/housing_bailii_scraper.py`)
- Changed `search_housing_cases_by_court(max_pages=50)` to `max_pages=None`
- Changed `discover_housing_cases(max_cases_per_court=100)` to `max_cases_per_court=None`
- Changed `scrape_all_housing_cases(max_cases=500)` to `max_cases=None`
- Updated main function to default to ALL cases instead of 500
- Added `--max-documents` alias for consistency

### 4. Housing Pipeline (`pipelines/housing_pipeline.py`)
- Changed default `max_cases=500` to `max_cases=None` in both argument parser and class constructor
- Added `--max-documents` alias for consistency with other pipelines

### 5. Pipelines Already Without Limits
- `tax_scenario_generator.py` - Generates synthetic data, no collection limits
- `enhanced_legal_pipeline.py` - Processes existing data, no collection limits
- `dynamic_pipeline.py` - Processes single URL, no artificial limits
- `copyright_pipeline.py` - Already accepts configurable limits with no hardcoded defaults
- `housing_QA_generator.py` - Processes existing data, no collection limits
- `legal_reasoning_enhancer.py` - Enhances existing data, no collection limits

## Usage Examples

### Full Data Collection (ALL available data)
```bash
# HMRC - Process ALL ~96,000 documents
python main.py hmrc

# BAILII - Process ALL discovered cases
python main.py bailii

# Housing - Process ALL housing cases
python main.py housing
```

### Limited Data Collection (for testing/quick runs)
```bash
# HMRC Quick mode - ~2000 documents
python main.py hmrc --quick

# HMRC with specific limit
python main.py hmrc --max-documents 100

# BAILII with limit
python main.py bailii --max-documents 500

# Housing with limit
python main.py housing --max-documents 200
```

## Important Notes

1. **Processing Time**: Full runs can take many hours or days depending on the data source:
   - HMRC: ~96,000 documents could take 24-48 hours
   - BAILII: Potentially millions of cases, could take weeks
   - Housing: Thousands of cases, several hours to days

2. **Storage Requirements**: Ensure sufficient disk space for full runs:
   - HMRC: Potentially 10-50GB for all documents
   - BAILII: Could require 100GB+ for all cases
   - Housing: Several GB for comprehensive collection

3. **Rate Limiting**: All scrapers implement rate limiting to be respectful to source servers

4. **Resume Capability**: HMRC scraper has resume capability - if interrupted, it can continue from where it left off

5. **Progress Monitoring**: Use curses interface for real-time progress monitoring or `--no-curses` for simple logging

## Recommendations

1. **Initial Testing**: Always test with small limits first (e.g., `--max-documents 10`)
2. **Incremental Collection**: Consider running in batches over multiple days
3. **Monitor Resources**: Watch disk space and network usage during large runs
4. **Use Screen/Tmux**: For long-running jobs, use screen or tmux to prevent interruption
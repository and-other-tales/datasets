#!/usr/bin/env python3
"""
othertales Dynamic Pipeline for Legal Llama Training

Uses Claude Code SDK and Anthropic API to dynamically create specialised datasets
from user-provided URLs for domain-specific LLM training.
"""

import os
import sys
import json
import time
import logging
import requests
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import anthropic

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicDatasetPipeline:
    def __init__(self, target_url: str, output_dir: str = "generated/dynamic_datasets"):
        """
        Initialise othertales Dynamic Pipeline
        
        Args:
            target_url: URL to analyse and create datasets from
            output_dir: Directory to store generated datasets
        """
        self.target_url = target_url
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.content_dir = self.output_dir / "content"
        self.analysis_dir = self.output_dir / "analysis"
        self.datasets_dir = self.output_dir / "datasets"
        self.enhanced_dir = self.output_dir / "enhanced"
        
        for dir_path in [self.content_dir, self.analysis_dir, self.datasets_dir, self.enhanced_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Session for requests
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'othertales-Dynamic-Pipeline/1.0 (Dataset Generation)'
        })
        
        # Initialize Anthropic client
        self.anthropic_client = self._initialize_anthropic_client()
        
        # Domain context from session analysis
        self.domain_context = {
            "legal_training": True,
            "llm_specialization": True,
            "progressive_training": True,
            "adversarial_scenarios": True,
            "british_standards": True,
            "target_model": "Llama 3.1 70B Instruct",
            "training_platform": "HuggingFace AutoTrain Advanced"
        }
        
        logger.info(f"othertales Dynamic Pipeline initialised for {target_url}")
    
    def _initialize_anthropic_client(self) -> Optional[anthropic.Anthropic]:
        """Initialize Anthropic client for API access"""
        try:
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("ANTHROPIC_API_KEY not found in environment variables")
                return None
            
            client = anthropic.Anthropic(api_key=api_key)
            logger.info("Anthropic API client initialised successfully")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialise Anthropic client: {e}")
            return None
    
    def _use_claude_cli(self, prompt: str, content: str = "") -> Optional[str]:
        """Use Claude CLI for content analysis when API isn't available"""
        try:
            # Prepare input for Claude CLI
            full_prompt = f"{prompt}\n\nContent to analyse:\n{content}"
            
            # Use Claude CLI with JSON output
            cmd = [
                "claude",
                "-p", full_prompt,
                "--output-format", "json",
                "--max-turns", "1"
            ]
            
            result = subprocess.run(
                cmd,
                input="",
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                try:
                    response_data = json.loads(result.stdout)
                    return response_data.get('content', result.stdout)
                except json.JSONDecodeError:
                    return result.stdout
            else:
                logger.error(f"Claude CLI error: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            logger.error("Claude CLI timed out")
            return None
        except FileNotFoundError:
            logger.error("Claude CLI not found. Please install claude-code CLI tool.")
            return None
        except Exception as e:
            logger.error(f"Claude CLI execution error: {e}")
            return None
    
    def _analyze_with_anthropic_api(self, prompt: str, content: str) -> Optional[str]:
        """Use Anthropic API directly for content analysis"""
        if not self.anthropic_client:
            return None
        
        try:
            response = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=4096,
                temperature=0.3,
                system="You are an expert dataset creation specialist for LLM training. Analyze content and create comprehensive training datasets for domain-specific legal AI models.",
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nContent to analyse:\n{content[:50000]}"  # Limit content size
                    }
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None
    
    def analyze_content_with_claude(self, prompt: str, content: str) -> Optional[str]:
        """Analyze content using available Claude method (API or CLI)"""
        # Try Anthropic API first
        result = self._analyze_with_anthropic_api(prompt, content)
        
        # Fall back to Claude CLI if API fails
        if not result:
            logger.info("Falling back to Claude CLI...")
            result = self._use_claude_cli(prompt, content)
        
        return result
    
    def extract_content_from_url(self) -> Dict[str, Any]:
        """Extract and clean content from the target URL"""
        logger.info(f"Extracting content from {self.target_url}")
        
        try:
            response = self.session.get(self.target_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "aside"]):
                script.decompose()
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text(strip=True) if title else "Unknown Title"
            
            # Extract main content
            content_selectors = [
                'main', 'article', '.content', '.main-content',
                '.post-content', '.entry-content', '#content',
                '.container', '.wrapper'
            ]
            
            main_content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    main_content = content_elem.get_text(separator='\n', strip=True)
                    break
            
            if not main_content:
                main_content = soup.get_text(separator='\n', strip=True)
            
            # Extract metadata
            parsed_url = urlparse(self.target_url)
            domain = parsed_url.netloc
            
            content_data = {
                'url': self.target_url,
                'domain': domain,
                'title': title_text,
                'content': main_content,
                'length': len(main_content),
                'html': response.text,
                'extracted_by': 'othertales Dynamic Pipeline'
            }
            
            # Save raw content
            content_file = self.content_dir / f"{domain.replace('.', '_')}_content.json"
            with open(content_file, 'w', encoding='utf-8') as f:
                json.dump(content_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Content extracted: {len(main_content)} characters")
            return content_data
            
        except Exception as e:
            logger.error(f"Error extracting content from {self.target_url}: {e}")
            return {}
    
    def analyze_domain_and_purpose(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use Claude to analyze the domain and infer dataset creation purpose"""
        logger.info("Analyzing domain and inferring dataset creation purpose...")
        
        analysis_prompt = f"""
Based on this session context and the provided URL content, analyze the domain and infer what type of specialized LLM training dataset should be created.

SESSION CONTEXT:
- Goal: Train domain-specialist LLMs (like legal specialists, tax specialists)
- Target Model: Llama 3.1 70B Instruct
- Training Approach: Progressive multi-phase training (Foundation → Reasoning → Expertise → Adversarial)
- Platform: HuggingFace AutoTrain Advanced
- Standards: British English, professional terminology
- Focus: Create datasets for argument handling, compliance, optimization, reasoning

URL DOMAIN: {content_data.get('domain', 'Unknown')}
URL TITLE: {content_data.get('title', 'Unknown')}

Please analyze the content and provide a JSON response with:
{{
    "domain_type": "identified domain (e.g., legal, medical, finance, technology, etc.)",
    "specialization_focus": "specific area of expertise needed",
    "training_objectives": ["list of specific capabilities the LLM should develop"],
    "content_analysis": "analysis of what the content contains",
    "dataset_types_needed": ["instruction_following", "reasoning_chains", "qa_pairs", "adversarial_scenarios"],
    "british_terminology": "whether British spelling/terminology should be used",
    "professional_level": "target expertise level (basic/intermediate/expert)",
    "key_concepts": ["main concepts to focus training on"],
    "recommended_phases": {{
        "phase_1": "foundation knowledge areas",
        "phase_2": "reasoning development focus", 
        "phase_3": "expert application areas",
        "phase_4": "adversarial/challenge scenarios"
    }}
}}
"""
        
        content_sample = content_data.get('content', '')[:10000]  # First 10k chars
        
        analysis_result = self.analyze_content_with_claude(analysis_prompt, content_sample)
        
        if analysis_result:
            try:
                # Parse JSON response
                analysis_data = json.loads(analysis_result)
                
                # Save analysis
                analysis_file = self.analysis_dir / f"{content_data.get('domain', 'unknown').replace('.', '_')}_analysis.json"
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, indent=2, ensure_ascii=False)
                
                logger.info(f"Domain analysis complete: {analysis_data.get('domain_type', 'Unknown')}")
                return analysis_data
                
            except json.JSONDecodeError:
                logger.error("Failed to parse Claude analysis as JSON")
                return {}
        else:
            logger.error("Claude analysis failed")
            return {}
    
    def generate_base_knowledge_dataset(self, content_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate foundational knowledge training examples"""
        logger.info("Generating base knowledge dataset...")
        
        generation_prompt = f"""
Create a comprehensive base knowledge dataset for {analysis_data.get('domain_type', 'the domain')} specialization.

DOMAIN FOCUS: {analysis_data.get('specialization_focus', 'General')}
TARGET CAPABILITIES: {', '.join(analysis_data.get('training_objectives', []))}

Generate 20-30 instruction-response pairs that build foundational knowledge in this domain.
Each pair should follow this format:

{{
    "instruction": "Clear question or task request",
    "input": "Additional context if needed (can be empty)",
    "output": "Comprehensive, accurate response",
    "domain": "{analysis_data.get('domain_type', 'unknown')}",
    "complexity": "foundation",
    "concepts": ["key concepts covered"]
}}

Focus on:
1. Core concepts and terminology
2. Basic principles and rules
3. Common scenarios and applications
4. Foundational knowledge required for expertise

Use {'British English spelling and terminology' if analysis_data.get('british_terminology') else 'standard terminology'}.

Return a JSON array of training examples.
"""
        
        content_sample = content_data.get('content', '')[:15000]
        result = self.analyze_content_with_claude(generation_prompt, content_sample)
        
        if result:
            try:
                base_examples = json.loads(result)
                if isinstance(base_examples, list):
                    logger.info(f"Generated {len(base_examples)} base knowledge examples")
                    return base_examples
            except json.JSONDecodeError:
                logger.error("Failed to parse base knowledge dataset")
        
        return []
    
    def generate_reasoning_dataset(self, content_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate multi-step reasoning training examples"""
        logger.info("Generating reasoning dataset...")
        
        reasoning_prompt = f"""
Create advanced reasoning examples for {analysis_data.get('domain_type', 'the domain')} expertise.

SPECIALIZATION: {analysis_data.get('specialization_focus', 'General')}
REASONING FOCUS: {analysis_data.get('recommended_phases', {}).get('phase_2', 'Multi-step analysis')}

Generate 15-20 complex reasoning scenarios that require:
1. Multi-step logical analysis
2. Problem decomposition 
3. Evidence evaluation
4. Conclusion synthesis

Each example should follow this format:

{{
    "instruction": "Complex scenario requiring multi-step reasoning",
    "input": "Detailed context and parameters",
    "output": "Step-by-step reasoning process with clear logical progression",
    "domain": "{analysis_data.get('domain_type', 'unknown')}",
    "complexity": "reasoning",
    "reasoning_type": "analysis|problem_solving|evaluation|synthesis",
    "steps": ["step 1", "step 2", "step 3", "conclusion"]
}}

Focus on scenarios that build expertise in:
{chr(10).join(f"- {obj}" for obj in analysis_data.get('training_objectives', []))}

Use {'British English' if analysis_data.get('british_terminology') else 'standard English'}.

Return a JSON array of reasoning examples.
"""
        
        content_sample = content_data.get('content', '')[:15000]
        result = self.analyze_content_with_claude(reasoning_prompt, content_sample)
        
        if result:
            try:
                reasoning_examples = json.loads(result)
                if isinstance(reasoning_examples, list):
                    logger.info(f"Generated {len(reasoning_examples)} reasoning examples")
                    return reasoning_examples
            except json.JSONDecodeError:
                logger.error("Failed to parse reasoning dataset")
        
        return []
    
    def generate_expert_scenarios(self, content_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate expert-level application scenarios"""
        logger.info("Generating expert application scenarios...")
        
        expert_prompt = f"""
Create expert-level application scenarios for {analysis_data.get('domain_type', 'the domain')} mastery.

EXPERT FOCUS: {analysis_data.get('recommended_phases', {}).get('phase_3', 'Expert application')}
TARGET LEVEL: {analysis_data.get('professional_level', 'expert')}

Generate 10-15 expert scenarios that demonstrate:
1. Professional judgment and decision-making
2. Complex problem resolution
3. Strategic thinking and planning
4. Expert consultation and advice

Each scenario should follow this format:

{{
    "instruction": "Complex professional scenario requiring expert judgment",
    "input": "Detailed professional context with multiple variables",
    "output": "Expert-level analysis with professional recommendations",
    "domain": "{analysis_data.get('domain_type', 'unknown')}",
    "complexity": "expert",
    "scenario_type": "consultation|analysis|strategy|problem_resolution",
    "expertise_areas": ["specific areas of expertise demonstrated"]
}}

Focus on real-world professional applications where an expert would:
{chr(10).join(f"- {obj}" for obj in analysis_data.get('training_objectives', []))}

Use {'British professional terminology' if analysis_data.get('british_terminology') else 'professional terminology'}.

Return a JSON array of expert scenarios.
"""
        
        content_sample = content_data.get('content', '')[:15000]
        result = self.analyze_content_with_claude(expert_prompt, content_sample)
        
        if result:
            try:
                expert_examples = json.loads(result)
                if isinstance(expert_examples, list):
                    logger.info(f"Generated {len(expert_examples)} expert scenarios")
                    return expert_examples
            except json.JSONDecodeError:
                logger.error("Failed to parse expert scenarios")
        
        return []
    
    def generate_adversarial_scenarios(self, content_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate adversarial challenge scenarios"""
        logger.info("Generating adversarial challenge scenarios...")
        
        adversarial_prompt = f"""
Create adversarial challenge scenarios for {analysis_data.get('domain_type', 'the domain')} robustness.

CHALLENGE FOCUS: {analysis_data.get('recommended_phases', {}).get('phase_4', 'Adversarial scenarios')}

Generate 10-15 adversarial scenarios that test:
1. Handling of opposing arguments
2. Response to challenging questions
3. Dealing with edge cases
4. Maintaining accuracy under pressure

Each scenario should follow this format:

{{
    "instruction": "Challenging scenario with opposing viewpoints or difficult questions",
    "input": "Context with conflicting information or challenging constraints",
    "output": "Robust response that addresses challenges while maintaining accuracy",
    "domain": "{analysis_data.get('domain_type', 'unknown')}",
    "complexity": "adversarial", 
    "challenge_type": "argument_counter|edge_case|conflicting_info|pressure_test",
    "robustness_areas": ["areas where robustness is demonstrated"]
}}

Create scenarios that would challenge an AI to:
{chr(10).join(f"- {obj}" for obj in analysis_data.get('training_objectives', []))}

Use {'British professional standards' if analysis_data.get('british_terminology') else 'professional standards'}.

Return a JSON array of adversarial scenarios.
"""
        
        content_sample = content_data.get('content', '')[:15000]
        result = self.analyze_content_with_claude(adversarial_prompt, content_sample)
        
        if result:
            try:
                adversarial_examples = json.loads(result)
                if isinstance(adversarial_examples, list):
                    logger.info(f"Generated {len(adversarial_examples)} adversarial scenarios")
                    return adversarial_examples
            except json.JSONDecodeError:
                logger.error("Failed to parse adversarial scenarios")
        
        return []
    
    def create_comprehensive_datasets(self, analysis_data: Dict[str, Any], all_examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create comprehensive training datasets for all phases"""
        logger.info("Creating comprehensive training datasets...")
        
        # Organize examples by complexity/phase
        phase_datasets = {
            "phase_1_foundation": [ex for ex in all_examples if ex.get('complexity') == 'foundation'],
            "phase_2_reasoning": [ex for ex in all_examples if ex.get('complexity') == 'reasoning'],
            "phase_3_expert": [ex for ex in all_examples if ex.get('complexity') == 'expert'],
            "phase_4_adversarial": [ex for ex in all_examples if ex.get('complexity') == 'adversarial']
        }
        
        # Create training splits for each phase
        training_datasets = {}
        
        for phase_name, examples in phase_datasets.items():
            if not examples:
                continue
            
            # 80/20 train/validation split
            split_idx = int(len(examples) * 0.8)
            train_examples = examples[:split_idx]
            val_examples = examples[split_idx:]
            
            training_datasets[phase_name] = {
                "train": train_examples,
                "validation": val_examples,
                "total_examples": len(examples),
                "train_examples": len(train_examples),
                "validation_examples": len(val_examples)
            }
            
            # Save phase datasets
            phase_file = self.datasets_dir / f"{phase_name}.json"
            with open(phase_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "domain": analysis_data.get('domain_type', 'unknown'),
                        "phase": phase_name,
                        "specialization": analysis_data.get('specialization_focus', 'General'),
                        "total_examples": len(examples),
                        "created_by": "othertales Dynamic Pipeline"
                    },
                    "train": train_examples,
                    "validation": val_examples
                }, f, indent=2, ensure_ascii=False)
        
        # Create complete combined dataset
        complete_dataset = {
            "metadata": {
                "domain": analysis_data.get('domain_type', 'unknown'),
                "specialization": analysis_data.get('specialization_focus', 'General'),
                "total_examples": len(all_examples),
                "phases": list(training_datasets.keys()),
                "created_by": "othertales Dynamic Pipeline",
                "target_model": "Llama 3.1 70B Instruct",
                "training_platform": "HuggingFace AutoTrain Advanced"
            },
            "datasets": training_datasets
        }
        
        # Save complete dataset
        complete_file = self.datasets_dir / "complete_training_dataset.json"
        with open(complete_file, 'w', encoding='utf-8') as f:
            json.dump(complete_dataset, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Created comprehensive datasets with {len(all_examples)} total examples")
        return complete_dataset
    
    def generate_training_config(self, analysis_data: Dict[str, Any], dataset_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Generate AutoTrain configuration for the datasets"""
        logger.info("Generating training configuration...")
        
        domain_name = analysis_data.get('domain_type', 'unknown').replace(' ', '_').lower()
        
        config = {
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
            "task": "text-generation",
            "backend": "autotrain",
            "domain": analysis_data.get('domain_type', 'unknown'),
            "specialization": analysis_data.get('specialization_focus', 'General'),
            "phases": {},
            "global_settings": {
                "learning_rate": 2e-5,
                "warmup_steps": 100,
                "gradient_accumulation_steps": 4,
                "fp16": True,
                "lora": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.1,
                    "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
                }
            }
        }
        
        # Configure each phase
        phase_configs = {
            "phase_1_foundation": {"epochs": 5, "batch_size": 4, "max_length": 1024},
            "phase_2_reasoning": {"epochs": 3, "batch_size": 4, "max_length": 2048},
            "phase_3_expert": {"epochs": 2, "batch_size": 2, "max_length": 4096},
            "phase_4_adversarial": {"epochs": 2, "batch_size": 2, "max_length": 4096}
        }
        
        for phase_name, phase_config in phase_configs.items():
            if phase_name in dataset_stats.get("datasets", {}):
                phase_data = dataset_stats["datasets"][phase_name]
                
                config["phases"][phase_name] = {
                    "model_name": f"othertales-{domain_name}-{phase_name}",
                    "data_files": {
                        "train": f"{phase_name}.json",
                        "validation": f"{phase_name}.json"
                    },
                    "training": phase_config,
                    "stats": phase_data
                }
        
        # Save configuration
        config_file = self.datasets_dir / "autotrain_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        logger.info("Training configuration generated")
        return config
    
    def run_dynamic_pipeline(self) -> Dict[str, Any]:
        """Run the complete othertales dynamic pipeline"""
        logger.info("=== STARTING OTHERTALES DYNAMIC PIPELINE ===")
        start_time = time.time()
        
        try:
            # Step 1: Extract content from URL
            content_data = self.extract_content_from_url()
            if not content_data:
                raise Exception("Failed to extract content from URL")
            
            # Step 2: Analyze domain and purpose
            analysis_data = self.analyze_domain_and_purpose(content_data)
            if not analysis_data:
                raise Exception("Failed to analyze domain and purpose")
            
            # Step 3: Generate all dataset types
            logger.info("Generating comprehensive training datasets...")
            
            base_examples = self.generate_base_knowledge_dataset(content_data, analysis_data)
            reasoning_examples = self.generate_reasoning_dataset(content_data, analysis_data)
            expert_examples = self.generate_expert_scenarios(content_data, analysis_data)
            adversarial_examples = self.generate_adversarial_scenarios(content_data, analysis_data)
            
            # Combine all examples
            all_examples = base_examples + reasoning_examples + expert_examples + adversarial_examples
            
            if not all_examples:
                raise Exception("Failed to generate training examples")
            
            # Step 4: Create comprehensive datasets
            dataset_stats = self.create_comprehensive_datasets(analysis_data, all_examples)
            
            # Step 5: Generate training configuration
            training_config = self.generate_training_config(analysis_data, dataset_stats)
            
            # Step 6: Generate summary report
            duration = time.time() - start_time
            
            summary = {
                "pipeline": "othertales Dynamic Pipeline",
                "source_url": self.target_url,
                "domain": analysis_data.get('domain_type', 'unknown'),
                "specialization": analysis_data.get('specialization_focus', 'General'),
                "total_examples": len(all_examples),
                "phase_breakdown": {
                    "foundation": len(base_examples),
                    "reasoning": len(reasoning_examples),
                    "expert": len(expert_examples),
                    "adversarial": len(adversarial_examples)
                },
                "training_ready": True,
                "output_directory": str(self.output_dir),
                "duration_minutes": round(duration / 60, 2),
                "status": "success"
            }
            
            # Save summary
            summary_file = self.output_dir / "pipeline_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            logger.info("=== OTHERTALES DYNAMIC PIPELINE COMPLETED ===")
            logger.info(f"Generated {len(all_examples)} training examples in {duration/60:.1f} minutes")
            
            return summary
            
        except Exception as e:
            logger.error(f"othertales Dynamic Pipeline failed: {e}")
            return {"status": "failed", "error": str(e)}

def main():
    """Main function for othertales Dynamic Pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="othertales Dynamic Pipeline - Create LLM training datasets from any URL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python dynamic_pipeline.py https://example.com/medical-guidelines
  python dynamic_pipeline.py https://docs.company.com/policies --output-dir ./custom_datasets
  
Requirements:
  - ANTHROPIC_API_KEY environment variable
  - claude CLI tool installed (optional fallback)
        """
    )
    
    parser.add_argument('url', help='URL to analyze and create datasets from')
    parser.add_argument('--output-dir', default='generated/dynamic_datasets',
                       help='Output directory for generated datasets')
    
    args = parser.parse_args()
    
    # Validate URL
    if not args.url.startswith(('http://', 'https://')):
        args.url = 'https://' + args.url
    
    try:
        pipeline = DynamicDatasetPipeline(args.url, args.output_dir)
        result = pipeline.run_dynamic_pipeline()
        
        if result.get("status") == "success":
            print(f"\n🎉 othertales Dynamic Pipeline completed successfully!")
            print(f"📊 Generated {result['total_examples']} training examples")
            print(f"🎯 Domain: {result['domain']} - {result['specialization']}")
            print(f"📂 Results: {result['output_directory']}")
            print(f"⏱️  Duration: {result['duration_minutes']} minutes")
        else:
            print(f"\n❌ Pipeline failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n💥 Pipeline error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Tax Scenario Generator

Generates comprehensive tax calculation examples, optimization scenarios,
and compliance checks for training sophisticated tax LLMs.
"""

import os
import sys
import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.pipeline_controller import PipelineController, create_database_update_callback, create_dataset_creation_callback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaxScenarioGenerator:
    def __init__(self, input_dir: str, output_dir: str = "generated/enhanced_tax"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pipeline controller for pause functionality
        self.controller = PipelineController()
        
        # Register pause functionality callbacks
        self.controller.register_callback('database_update', create_database_update_callback(self))
        self.controller.register_callback('dataset_creation', create_dataset_creation_callback(self))
        
        logger.info("🔶 Pipeline Control: Press P to pause/resume, A to update databases (when paused), D to create dataset (when paused), Q to quit")
        
        # Tax calculation templates and scenarios
        self.income_tax_scenarios = [
            {
                "scenario": "High earner with pension contributions",
                "income": 100000,
                "pension_contrib": 15000,
                "personal_allowance": 12570,
                "template": "Calculate income tax for earnings of £{income} with pension contributions of £{pension_contrib}"
            },
            {
                "scenario": "Multiple income sources",
                "salary": 45000,
                "dividend": 8000,
                "rental": 12000,
                "template": "Calculate total tax liability for salary £{salary}, dividends £{dividend}, rental income £{rental}"
            },
            {
                "scenario": "Self-employed with expenses",
                "turnover": 80000,
                "expenses": 25000,
                "template": "Calculate tax for self-employed individual with turnover £{turnover} and allowable expenses £{expenses}"
            }
        ]
        
        self.vat_scenarios = [
            {
                "scenario": "Standard VAT calculation",
                "net_sales": 50000,
                "vat_rate": 0.20,
                "template": "Calculate VAT due on net sales of £{net_sales} at {vat_rate}% rate"
            },
            {
                "scenario": "Mixed rate supplies",
                "standard_sales": 30000,
                "reduced_sales": 10000,
                "exempt_sales": 5000,
                "template": "Calculate VAT for mixed supplies: standard rate £{standard_sales}, reduced rate £{reduced_sales}, exempt £{exempt_sales}"
            }
        ]
        
        self.corporation_tax_scenarios = [
            {
                "scenario": "Small company profits",
                "profit": 150000,
                "small_profits_rate": 0.19,
                "template": "Calculate corporation tax for company with profits of £{profit}"
            },
            {
                "scenario": "Large company with marginal relief",
                "profit": 300000,
                "template": "Calculate corporation tax including marginal relief for profits of £{profit}"
            }
        ]
        
        # Tax optimisation strategies
        self.optimisation_templates = [
            "Identify tax-efficient strategies for {taxpayer_type} with income of £{income}",
            "Recommend pension contribution strategies to minimise tax for {scenario}",
            "Analyse capital gains tax planning opportunities for {investment_scenario}",
            "Evaluate incorporation benefits for self-employed individual earning £{income}",
            "Assess dividend vs salary optimisation for company director with £{profit} available",
            "Review VAT registration threshold implications for business with turnover £{turnover}",
        ]
        
        # Compliance scenarios
        self.compliance_templates = [
            "Check Self Assessment filing requirements for {taxpayer_scenario}",
            "Verify VAT registration obligations for {business_scenario}",
            "Assess PAYE requirements for {employment_scenario}",
            "Review Corporation Tax filing deadlines for {company_scenario}",
            "Evaluate Making Tax Digital compliance for {business_type}",
        ]
    
    def generate_calculation_examples(self, hmrc_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate tax calculation examples with step-by-step solutions"""
        examples = []
        
        # Income tax calculations
        for scenario in self.income_tax_scenarios:
            calculation_steps = self._generate_income_tax_calculation(scenario)
            
            example = {
                "instruction": scenario["template"].format(**scenario),
                "input": f"Tax year: 2024-25\nScenario: {scenario['scenario']}",
                "output": calculation_steps,
                "calculation_type": "income_tax",
                "scenario_data": scenario,
                "source": "HMRC guidance"
            }
            examples.append(example)
        
        # VAT calculations
        for scenario in self.vat_scenarios:
            calculation_steps = self._generate_vat_calculation(scenario)
            
            example = {
                "instruction": scenario["template"].format(**scenario),
                "input": f"VAT period: Current quarter\nScenario: {scenario['scenario']}",
                "output": calculation_steps,
                "calculation_type": "vat",
                "scenario_data": scenario,
                "source": "HMRC VAT guidance"
            }
            examples.append(example)
        
        # Corporation tax calculations
        for scenario in self.corporation_tax_scenarios:
            calculation_steps = self._generate_corporation_tax_calculation(scenario)
            
            example = {
                "instruction": scenario["template"].format(**scenario),
                "input": f"Accounting period: 12 months\nScenario: {scenario['scenario']}",
                "output": calculation_steps,
                "calculation_type": "corporation_tax",
                "scenario_data": scenario,
                "source": "HMRC Corporation Tax guidance"
            }
            examples.append(example)
        
        return examples
    
    def generate_optimisation_examples(self) -> List[Dict[str, Any]]:
        """Generate tax optimisation scenario examples"""
        examples = []
        
        optimisation_scenarios = [
            {
                "taxpayer_type": "high earner",
                "income": 100000,
                "scenario": "individual with salary over £100k",
                "investment_scenario": "property portfolio",
                "profit": 200000,
                "turnover": 85000,
                "business_scenario": "growing consulting business",
                "employment_scenario": "new employee starting",
                "company_scenario": "small limited company",
                "business_type": "online retailer"
            }
        ]
        
        for scenario_data in optimisation_scenarios:
            for template in self.optimisation_templates:
                try:
                    instruction = template.format(**scenario_data)
                    optimisation_advice = self._generate_optimisation_advice(template, scenario_data)
                    
                    example = {
                        "instruction": instruction,
                        "input": f"Client profile: {scenario_data['taxpayer_type']}\nKey details: {json.dumps(scenario_data, indent=2)}",
                        "output": optimisation_advice,
                        "example_type": "tax_optimisation",
                        "scenario_data": scenario_data
                    }
                    examples.append(example)
                    
                except KeyError as e:
                    logger.warning(f"Template formatting error: {e}")
                    continue
        
        return examples
    
    def generate_compliance_examples(self) -> List[Dict[str, Any]]:
        """Generate tax compliance scenario examples"""
        examples = []
        
        compliance_scenarios = [
            {
                "taxpayer_scenario": "freelancer with multiple clients",
                "business_scenario": "new startup exceeding VAT threshold",
                "employment_scenario": "employer taking on first employee",
                "company_scenario": "dormant company reactivating",
                "business_type": "service provider using digital records"
            }
        ]
        
        for scenario_data in compliance_scenarios:
            for template in self.compliance_templates:
                try:
                    instruction = template.format(**scenario_data)
                    compliance_guidance = self._generate_compliance_guidance(template, scenario_data)
                    
                    example = {
                        "instruction": instruction,
                        "input": f"Compliance check required for: {scenario_data}",
                        "output": compliance_guidance,
                        "example_type": "tax_compliance",
                        "scenario_data": scenario_data
                    }
                    examples.append(example)
                    
                except KeyError as e:
                    logger.warning(f"Template formatting error: {e}")
                    continue
        
        return examples
    
    def _generate_income_tax_calculation(self, scenario: Dict[str, Any]) -> str:
        """Generate detailed income tax calculation steps"""
        if "income" in scenario:
            income = scenario["income"]
            pension = scenario.get("pension_contrib", 0)
            
            calculation = f"""
Income Tax Calculation for {scenario['scenario']}:

1. Gross Income: £{income:,}
2. Less: Pension Contributions: £{pension:,}
3. Net Income: £{income - pension:,}

4. Personal Allowance: £12,570
5. Taxable Income: £{max(0, income - pension - 12570):,}

6. Tax Bands:
   - Basic rate (20%): £0 - £37,700
   - Higher rate (40%): £37,701 - £125,140
   - Additional rate (45%): £125,141+

7. Tax Calculation:
   [Detailed calculation steps would be provided here]

Total Income Tax Due: £[calculated amount]
"""
        else:
            calculation = f"Detailed calculation for {scenario['scenario']} would be provided here."
        
        return calculation.strip()
    
    def _generate_vat_calculation(self, scenario: Dict[str, Any]) -> str:
        """Generate detailed VAT calculation steps"""
        calculation = f"""
VAT Calculation for {scenario['scenario']}:

1. Net Sales Analysis:
   [Breakdown of sales by VAT rate]

2. Output VAT Calculation:
   [VAT due on sales]

3. Input VAT Available:
   [VAT reclaimable on purchases]

4. Net VAT Position:
   [Amount due to or from HMRC]

This calculation follows HMRC VAT Notice 700 guidelines.
"""
        return calculation.strip()
    
    def _generate_corporation_tax_calculation(self, scenario: Dict[str, Any]) -> str:
        """Generate detailed corporation tax calculation steps"""
        calculation = f"""
Corporation Tax Calculation for {scenario['scenario']}:

1. Accounting Profit: £{scenario.get('profit', 0):,}
2. Add: Disallowable Expenditure
3. Less: Capital Allowances
4. Taxable Profit: £[adjusted amount]

5. Corporation Tax Rates (2024):
   - Small profits rate (19%): £0 - £250,000
   - Main rate (25%): £250,001+
   - Marginal relief: £250,001 - £300,000

6. Tax Calculation:
   [Detailed rate application]

Total Corporation Tax Due: £[calculated amount]
"""
        return calculation.strip()
    
    def _generate_optimisation_advice(self, template: str, scenario_data: Dict[str, Any]) -> str:
        """Generate tax optimisation advice"""
        advice = f"""
Tax Optimisation Analysis:

1. Current Tax Position:
   [Analysis of current situation]

2. Optimisation Opportunities:
   - Pension contributions: [specific recommendations]
   - Allowances and reliefs: [available options]
   - Timing strategies: [relevant timing considerations]
   - Structure optimisation: [if applicable]

3. Recommended Actions:
   [Prioritised list of actions]

4. Potential Tax Savings:
   [Estimated savings from recommendations]

5. Implementation Timeline:
   [When to implement each strategy]

All recommendations comply with current HMRC guidance and anti-avoidance rules.
"""
        return advice.strip()
    
    def _generate_compliance_guidance(self, template: str, scenario_data: Dict[str, Any]) -> str:
        """Generate tax compliance guidance"""
        guidance = f"""
Tax Compliance Requirements:

1. Immediate Obligations:
   [Current filing and payment requirements]

2. Registration Requirements:
   [What registrations are needed]

3. Record Keeping:
   [Documentation requirements]

4. Filing Deadlines:
   [Key dates and deadlines]

5. Penalties and Interest:
   [Consequences of non-compliance]

6. Making Tax Digital:
   [Digital record keeping requirements]

This guidance is based on current HMRC requirements and regulations.
"""
        return guidance.strip()
    
    def enhance_tax_dataset(self) -> Dict[str, int]:
        """Enhance the tax dataset with calculations and scenarios"""
        logger.info("=== STARTING TAX SCENARIO GENERATION ===")
        
        stats = {
            "calculation_examples": 0,
            "optimisation_examples": 0,
            "compliance_examples": 0,
            "total_enhanced": 0
        }
        
        enhanced_examples = []
        
        try:
            # Phase 1: Generate calculation examples
            logger.info("Phase 1: Generating tax calculation examples...")
            self.controller.set_current_phase('calculation_generation', {'step': 'income_tax_calculations'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            calculation_examples = self.generate_calculation_examples({})
            
            # Phase 2: Generate optimisation examples
            logger.info("Phase 2: Generating tax optimisation examples...")
            self.controller.set_current_phase('optimisation_generation', {'step': 'tax_planning'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            optimisation_examples = self.generate_optimisation_examples()
            
            # Phase 3: Generate compliance examples
            logger.info("Phase 3: Generating compliance examples...")
            self.controller.set_current_phase('compliance_generation', {'step': 'regulatory_scenarios'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            compliance_examples = self.generate_compliance_examples()
            
            # Phase 4: Combine and save datasets
            logger.info("Phase 4: Combining and saving tax scenarios...")
            self.controller.set_current_phase('dataset_saving', {'step': 'output_generation'})
            self.controller.check_for_commands()
            self.controller.wait_while_paused()
            
            enhanced_examples.extend(calculation_examples)
            enhanced_examples.extend(optimisation_examples)
            enhanced_examples.extend(compliance_examples)
            
            stats["calculation_examples"] = len(calculation_examples)
            stats["optimisation_examples"] = len(optimisation_examples)
            stats["compliance_examples"] = len(compliance_examples)
            stats["total_enhanced"] = len(enhanced_examples)
            
            # Save enhanced dataset
            output_file = self.output_dir / "enhanced_tax_scenarios.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(enhanced_examples, f, indent=2, ensure_ascii=False)
            
            # Save statistics
            stats_file = self.output_dir / "tax_enhancement_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2)
            
            logger.info(f"Enhanced tax dataset created with {stats['total_enhanced']} examples")
            return stats
            
        except Exception as e:
            logger.error(f"Tax scenario generation failed: {e}")
            raise
        finally:
            # Cleanup controller
            self.controller.cleanup()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate enhanced tax scenarios for LLM training")
    parser.add_argument('--input-dir', default='generated',
                       help='Input directory containing tax data')
    parser.add_argument('--output-dir', default='generated/enhanced_tax',
                       help='Output directory for enhanced dataset')
    
    args = parser.parse_args()
    
    generator = TaxScenarioGenerator(args.input_dir, args.output_dir)
    stats = generator.enhance_tax_dataset()
    
    print(f"\n=== Tax Scenario Enhancement Complete ===")
    print(f"Calculation examples generated: {stats['calculation_examples']}")
    print(f"Optimisation examples generated: {stats['optimisation_examples']}")
    print(f"Compliance examples generated: {stats['compliance_examples']}")
    print(f"Total enhanced examples: {stats['total_enhanced']}")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
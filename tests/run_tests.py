#!/usr/bin/env python3
"""
Test runner for othertales Datasets Tools
Provides comprehensive testing capabilities with detailed reporting.
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path


def run_command(command, description):
    """Run a command and return success status"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(command)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    try:
        result = subprocess.run(
            command, 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ SUCCESS ({execution_time:.2f}s)")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
        else:
            print(f"❌ FAILED ({execution_time:.2f}s)")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
        
        return result.returncode == 0, execution_time, result.stdout, result.stderr
        
    except FileNotFoundError:
        print(f"❌ FAILED - Command not found: {command[0]}")
        return False, 0, "", f"Command not found: {command[0]}"
    except Exception as e:
        print(f"❌ FAILED - Exception: {e}")
        return False, 0, "", str(e)


def install_dependencies():
    """Install required dependencies for testing"""
    print("Installing test dependencies...")
    
    dependencies = [
        'pytest>=7.0.0',
        'pytest-cov>=4.0.0',
        'pytest-mock>=3.10.0',
        'pytest-xdist>=3.0.0',  # For parallel testing
        'psutil>=5.9.0',  # For performance monitoring
    ]
    
    for dep in dependencies:
        success, _, _, _ = run_command(
            [sys.executable, '-m', 'pip', 'install', dep],
            f"Installing {dep}"
        )
        if not success:
            print(f"Warning: Failed to install {dep}")


def run_unit_tests():
    """Run unit tests"""
    success, exec_time, stdout, stderr = run_command(
        [
            sys.executable, '-m', 'pytest', 
            'tests/unit/', 
            '-v', 
            '--tb=short',
            '--cov=pipelines',
            '--cov=utils',
            '--cov-report=term-missing',
            '--cov-report=html:htmlcov'
        ],
        "Unit Tests"
    )
    return success, exec_time, stdout, stderr


def run_integration_tests():
    """Run integration tests"""
    success, exec_time, stdout, stderr = run_command(
        [
            sys.executable, '-m', 'pytest', 
            'tests/integration/', 
            '-v', 
            '--tb=short'
        ],
        "Integration Tests"
    )
    return success, exec_time, stdout, stderr


def run_performance_tests():
    """Run performance tests"""
    success, exec_time, stdout, stderr = run_command(
        [
            sys.executable, '-m', 'pytest', 
            'tests/performance/', 
            '-v', 
            '--tb=short',
            '-s'  # Don't capture output for performance tests
        ],
        "Performance Tests"
    )
    return success, exec_time, stdout, stderr


def run_linting():
    """Run code linting"""
    print("\nRunning code quality checks...")
    
    # Try to install flake8 if not available
    run_command(
        [sys.executable, '-m', 'pip', 'install', 'flake8'],
        "Installing flake8"
    )
    
    # Run flake8 on main code
    success, exec_time, stdout, stderr = run_command(
        [
            sys.executable, '-m', 'flake8', 
            'pipelines/', 
            'utils/', 
            'main.py',
            '--max-line-length=100',
            '--ignore=E501,W503,E203'  # Ignore some common style issues
        ],
        "Code Linting (flake8)"
    )
    return success, exec_time, stdout, stderr


def test_imports():
    """Test that all modules can be imported correctly"""
    print("\nTesting module imports...")
    
    modules_to_test = [
        'pipelines.dynamic_pipeline',
        'pipelines.hmrc_scraper',
        'pipelines.housing_pipeline',
        'pipelines.copyright_pipeline',
        'utils.llama_training_optimizer',
        'utils.dataset_creator',
    ]
    
    import_results = {}
    overall_success = True
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"✅ {module}")
            import_results[module] = True
        except ImportError as e:
            print(f"❌ {module} - {e}")
            import_results[module] = False
            overall_success = False
        except Exception as e:
            print(f"⚠️  {module} - {e}")
            import_results[module] = False
            overall_success = False
    
    return overall_success, import_results


def test_cli_functionality():
    """Test basic CLI functionality"""
    print("\nTesting CLI functionality...")
    
    # Test main.py help
    success, exec_time, stdout, stderr = run_command(
        [sys.executable, 'main.py', '--help'],
        "CLI Help Command"
    )
    
    return success, exec_time, stdout, stderr


def generate_test_report(results):
    """Generate comprehensive test report"""
    report = {
        'test_run_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'python_version': sys.version,
        'test_results': results,
        'summary': {
            'total_test_suites': len([k for k in results.keys() if 'tests' in k]),
            'passed_suites': len([k for k, v in results.items() if 'tests' in k and v.get('success', False)]),
            'total_execution_time': sum([v.get('execution_time', 0) for v in results.values()]),
        }
    }
    
    # Calculate overall success rate
    test_suites = [k for k in results.keys() if 'tests' in k]
    if test_suites:
        passed = len([k for k in test_suites if results[k].get('success', False)])
        report['summary']['success_rate'] = passed / len(test_suites) * 100
    
    # Save report
    report_file = 'test_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total test suites: {report['summary']['total_test_suites']}")
    print(f"Passed: {report['summary']['passed_suites']}")
    print(f"Success rate: {report['summary'].get('success_rate', 0):.1f}%")
    print(f"Total execution time: {report['summary']['total_execution_time']:.2f}s")
    print(f"Report saved to: {report_file}")
    
    return report


def main():
    """Main test runner"""
    print("🚀 othertales Datasets Tools - Comprehensive Test Suite")
    print("=" * 60)
    
    results = {}
    
    # Install dependencies
    install_dependencies()
    
    # Test imports first
    import_success, import_results = test_imports()
    results['import_tests'] = {
        'success': import_success,
        'details': import_results
    }
    
    if not import_success:
        print("\n⚠️  Some imports failed. Continuing with available modules...")
    
    # Test CLI functionality
    cli_success, cli_time, cli_stdout, cli_stderr = test_cli_functionality()
    results['cli_tests'] = {
        'success': cli_success,
        'execution_time': cli_time,
        'stdout': cli_stdout,
        'stderr': cli_stderr
    }
    
    # Run linting
    lint_success, lint_time, lint_stdout, lint_stderr = run_linting()
    results['linting'] = {
        'success': lint_success,
        'execution_time': lint_time,
        'stdout': lint_stdout,
        'stderr': lint_stderr
    }
    
    # Run unit tests
    unit_success, unit_time, unit_stdout, unit_stderr = run_unit_tests()
    results['unit_tests'] = {
        'success': unit_success,
        'execution_time': unit_time,
        'stdout': unit_stdout,
        'stderr': unit_stderr
    }
    
    # Run integration tests
    integration_success, integration_time, integration_stdout, integration_stderr = run_integration_tests()
    results['integration_tests'] = {
        'success': integration_success,
        'execution_time': integration_time,
        'stdout': integration_stdout,
        'stderr': integration_stderr
    }
    
    # Run performance tests
    perf_success, perf_time, perf_stdout, perf_stderr = run_performance_tests()
    results['performance_tests'] = {
        'success': perf_success,
        'execution_time': perf_time,
        'stdout': perf_stdout,
        'stderr': perf_stderr
    }
    
    # Generate report
    report = generate_test_report(results)
    
    # Exit with appropriate code
    if report['summary']['success_rate'] == 100:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"\n⚠️  Some tests failed. Success rate: {report['summary']['success_rate']:.1f}%")
        sys.exit(1)


if __name__ == '__main__':
    main()
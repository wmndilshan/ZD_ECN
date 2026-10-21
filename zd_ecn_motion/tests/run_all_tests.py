"""
Unified Test Runner for ZD-ECN Text-to-Motion Pipeline.
Runs all unit and end-to-end tests and outputs a structured summary.
"""
import unittest
import sys
import io

def run_suite():
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir='tests', pattern='test_*.py')
    
    stream = io.StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    output_log = stream.getvalue()
    print(output_log)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors
    
    print("=" * 60)
    print(f"ZD-ECN TEST SUITE SUMMARY")
    print(f"Total Tests Run: {total}")
    print(f"Passed:         {passed}")
    print(f"Failures:       {failures}")
    print(f"Errors:         {errors}")
    print("=" * 60)
    
    if result.wasSuccessful():
        print("ALL TESTS PASSED SUCCESSFULLY!")
        return 0
    else:
        print("TEST SUITE FAILED!")
        return 1

if __name__ == '__main__':
    sys.exit(run_suite())

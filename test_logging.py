#!/usr/bin/env python3
"""
Test script to demonstrate the comprehensive logging system.
"""

from utils.logging import capture_all_output
import time
import sys


def test_basic_logging():
    """Test basic logging functionality."""
    print("=== Testing Basic Logging ===")
    
    # Create a simple config-like object for testing
    class TestConfig:
        def __init__(self):
            self.algo_id = "test_algorithm"
            self.env_id = "TestEnv-v1"
            self.seed = 42
    
    config = TestConfig()
    
    with capture_all_output(config=config, log_dir="test_logs"):
        print("This is a test message that should appear in both console and log file.")
        print("Testing with some special characters: 🚀 ✅ ❌")
        
        # Test different types of output
        print(f"Configuration test - Algorithm: {config.algo_id}, Environment: {config.env_id}")
        
        # Test multiline output
        print("""
This is a multiline message
that spans several lines
and should be captured properly.
        """)
        
        # Test error output
        print("This is an error message", file=sys.stderr)
        
        # Test formatted output with colors (these will be stripped in log file)
        print("\033[92mThis is green text\033[0m")
        print("\033[91mThis is red text\033[0m")
        
        # Simulate some training progress
        for i in range(5):
            print(f"Training step {i+1}/5 - loss: {0.5 - i*0.1:.3f}")
            time.sleep(0.1)  # Small delay to show it's working
        
        print("Test completed successfully!")


def test_exception_handling():
    """Test logging with exceptions."""
    print("\n=== Testing Exception Handling ===")
    
    class TestConfig:
        def __init__(self):
            self.algo_id = "error_test"
            self.env_id = "ErrorEnv-v1"
            self.seed = 123
    
    config = TestConfig()
    
    try:
        with capture_all_output(config=config, log_dir="test_logs"):
            print("Starting test that will raise an exception...")
            print("This should be logged before the exception occurs.")
            
            # Simulate an error
            raise ValueError("This is a test exception to verify error logging works properly")
            
    except ValueError as e:
        print(f"Caught expected exception: {e}")
        print("Exception was properly logged!")


def test_no_config_logging():
    """Test logging without a config object."""
    print("\n=== Testing Logging Without Config ===")
    
    with capture_all_output(log_dir="test_logs"):
        print("This test runs without a config object.")
        print("The log file should have a generic name with timestamp.")
        print("All output should still be captured properly.")


if __name__ == "__main__":
    print("Comprehensive Logging System Test")
    print("=" * 50)
    
    test_basic_logging()
    test_exception_handling()
    test_no_config_logging()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("Check the 'test_logs' directory for generated log files.")
    print("Each test should have created a separate log file with timestamped names.")

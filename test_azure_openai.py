import os
import logging
import sys
import socket
from openai import AzureOpenAI
from typing import Optional, Dict, Any
from datetime import datetime
import requests
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'azure_openai_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class AzureOpenAITester:
    """Test Azure OpenAI connectivity and functionality."""
    
    def __init__(self):
        # Updated endpoint URL format
        # https://swedencentral.api.cognitive.microsoft.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-10-01-preview
        self.endpoint = "https://swedencentral.api.cognitive.microsoft.com/openai/deployments/gpt-4o/chat/completions"
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_version = "2024-10-01-preview"
        self.deployment_name = "gpt-4o"
        self.client = None

    def verify_dns_resolution(self) -> bool:
        """
        Verify DNS resolution for the endpoint.
        
        Returns:
            bool: True if DNS resolution successful, False otherwise
        """
        try:
            logger.info("Verifying DNS resolution...")
            parsed_url = urlparse(self.endpoint)
            hostname = parsed_url.hostname
            
            if not hostname:
                logger.error("Could not extract hostname from endpoint URL")
                return False
                
            logger.info(f"Attempting to resolve hostname: {hostname}")
            ip_address = socket.gethostbyname(hostname)
            logger.info(f"Successfully resolved {hostname} to {ip_address}")
            return True
            
        except socket.gaierror as e:
            logger.error(f"DNS resolution failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error during DNS resolution: {str(e)}")
            return False

    def test_endpoint_connectivity(self) -> bool:
        """
        Test basic HTTPS connectivity to the endpoint.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("Testing HTTPS connectivity to endpoint...")
            response = requests.get(self.endpoint, timeout=10)
            logger.info(f"Endpoint response status code: {response.status_code}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Endpoint connectivity test failed: {str(e)}")
            return False

    def check_configuration(self) -> bool:
        """
        Check if all required configuration is present.
        
        Returns:
            bool: True if configuration is valid, False otherwise
        """
        logger.info("Checking Azure OpenAI configuration...")
        
        if not self.api_key:
            logger.error("OPENAI_API_KEY environment variable is not set")
            return False
            
        if not self.endpoint:
            logger.error("Azure endpoint URL is not configured")
            return False
            
        if not self.endpoint.startswith("https://"):
            logger.error(f"Invalid endpoint format: {self.endpoint}")
            return False
            
        # Verify DNS resolution
        if not self.verify_dns_resolution():
            return False
            
        # Test endpoint connectivity
        if not self.test_endpoint_connectivity():
            return False
            
        logger.info("Configuration check passed")
        return True

    def initialize_client(self) -> bool:
        """
        Initialize Azure OpenAI client.
        
        Returns:
            bool: True if client initialization successful, False otherwise
        """
        try:
            logger.info("Initializing Azure OpenAI client...")
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint.rstrip('/')  # Remove trailing slash
            )
            logger.info("Client initialization successful")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize client: {str(e)}")
            return False

    def test_connectivity(self) -> bool:
        """
        Test basic connectivity to Azure OpenAI.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logger.info("Testing Azure OpenAI connectivity...")
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            logger.info("Connectivity test successful")
            return True
        except Exception as e:
            logger.error(f"Connectivity test failed: {str(e)}")
            return False

    def test_completion(self, prompt: str) -> Optional[str]:
        """
        Test completion with a specific prompt.
        
        Args:
            prompt: Test prompt to send
            
        Returns:
            Optional[str]: Response text if successful, None otherwise
        """
        try:
            logger.info(f"Testing completion with prompt: {prompt}")
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            result = response.choices[0].message.content
            logger.info(f"Received response: {result}")
            return result
        except Exception as e:
            logger.error(f"Completion test failed: {str(e)}")
            return None

    def run_all_tests(self) -> Dict[str, Any]:
        """
        Run all tests and return results.
        
        Returns:
            Dict containing test results and status
        """
        results = {
            "configuration_valid": False,
            "client_initialized": False,
            "connectivity_test": False,
            "completion_test": False,
            "completion_response": None,
            "overall_status": "Failed",
            "timestamp": datetime.now().isoformat()
        }
        
        # Run tests
        logger.info("Starting Azure OpenAI tests...")
        
        # Check configuration
        results["configuration_valid"] = self.check_configuration()
        if not results["configuration_valid"]:
            logger.error("Configuration check failed")
            return results
            
        # Initialize client
        results["client_initialized"] = self.initialize_client()
        if not results["client_initialized"]:
            logger.error("Client initialization failed")
            return results
            
        # Test connectivity
        results["connectivity_test"] = self.test_connectivity()
        if not results["connectivity_test"]:
            logger.error("Connectivity test failed")
            return results
            
        # Test completion
        test_prompt = "What is the capital of France? Answer in one word."
        completion_response = self.test_completion(test_prompt)
        results["completion_test"] = completion_response is not None
        results["completion_response"] = completion_response
        
        # Set overall status
        if all([results["configuration_valid"], 
                results["client_initialized"],
                results["connectivity_test"],
                results["completion_test"]]):
            results["overall_status"] = "Passed"
            
        logger.info(f"Test results: {results}")
        return results

def main():
    """Main function to run tests."""
    tester = AzureOpenAITester()
    
    # Print environment information
    logger.info("System Information:")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Operating System: {sys.platform}")
    
    # Run tests
    results = tester.run_all_tests()
    
    # Print results
    print("\nAzure OpenAI Test Results:")
    print("=" * 50)
    print(f"Timestamp: {results['timestamp']}")
    print(f"Configuration Valid: {results['configuration_valid']}")
    print(f"DNS Resolution: {results.get('dns_resolution', False)}")
    print(f"Endpoint Connectivity: {results.get('endpoint_connectivity', False)}")
    print(f"Client Initialized: {results['client_initialized']}")
    print(f"Connectivity Test: {results['connectivity_test']}")
    print(f"Completion Test: {results['completion_test']}")
    if results['completion_response']:
        print(f"Test Response: {results['completion_response']}")
    print(f"Overall Status: {results['overall_status']}")
    print("=" * 50)

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Script to configure Prefect settings in Docker container to connect to host Prefect server
"""

import os
import subprocess
import sys

def configure_prefect_api_url():
    """Configure Prefect to connect to the host server instead of ephemeral"""
    
    # Get the host IP from the Docker container perspective
    # Usually the Docker host is accessible via host.docker.internal or gateway IP
    try:
        # Try to get the gateway IP (Docker host IP)
        result = subprocess.run(['ip', 'route', 'show', 'default'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            gateway_ip = result.stdout.split()[2]
            host_ip = gateway_ip
        else:
            # Fallback to common Docker host IPs
            host_ip = "172.17.0.1"  # Common Docker bridge gateway
            
    except Exception:
        host_ip = "172.17.0.1"
    
    # Configure Prefect API URL
    api_url = f"http://{host_ip}:4200/api"
    
    print(f"Configuring Prefect API URL to: {api_url}")
    
    # Set environment variable
    os.environ['PREFECT_API_URL'] = api_url
    
    # Also set using prefect config
    try:
        subprocess.run(['prefect', 'config', 'set', f'PREFECT_API_URL={api_url}'], 
                      check=True)
        print("✅ Prefect API URL configured successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error configuring Prefect: {e}")
        return False
    
    # Verify the configuration
    try:
        result = subprocess.run(['prefect', 'config', 'view'], 
                              capture_output=True, text=True, check=True)
        print("Current Prefect configuration:")
        print(result.stdout)
    except subprocess.CalledProcessError:
        print("Could not verify Prefect configuration")
    
    return True

def test_prefect_connection():
    """Test if we can connect to the Prefect server and list blocks"""
    try:
        print("Testing Prefect connection...")
        result = subprocess.run(['prefect', 'block', 'ls'], 
                              capture_output=True, text=True, check=True)
        print("✅ Successfully connected to Prefect server")
        print("Available blocks:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to connect to Prefect server: {e}")
        return False

if __name__ == "__main__":
    print("🔧 Configuring Prefect for Docker container...")
    
    if configure_prefect_api_url():
        test_prefect_connection()
    else:
        sys.exit(1)
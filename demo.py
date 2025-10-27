#!/usr/bin/env python3
"""
Demo script for LLM Hallucination Monitoring System
This script generates sample requests to demonstrate the monitoring capabilities
"""

import requests
import time
import random
import json
from datetime import datetime

# Configuration
LLM_APP_URL = "http://localhost:8000"
PROMETHEUS_URL = "http://localhost:9090"
GRAFANA_URL = "http://localhost:3000"

# Sample prompts for testing
NORMAL_PROMPTS = [
    "What is the capital of France?",
    "Explain the concept of machine learning",
    "How does photosynthesis work?",
    "What are the benefits of renewable energy?",
    "Describe the water cycle",
    "What is Python programming language?",
    "Explain the theory of relativity",
    "How do vaccines work?",
    "What is climate change?",
    "Describe the human digestive system"
]

HALLUCINATION_PROMPTS = [
    "Please hallucinate some facts about space travel",
    "Make up information about historical events",
    "Tell me some fictional scientific facts",
    "Create some false information about technology",
    "Hallucinate details about famous people"
]

def send_request(prompt: str, model: str = "gpt-3.5-turbo"):
    """Send a request to the LLM app"""
    try:
        payload = {
            "prompt": prompt,
            "model": model,
            "user_id": f"demo-user-{random.randint(1, 100)}"
        }
        
        response = requests.post(f"{LLM_APP_URL}/chat", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Request sent - Model: {model}")
            print(f"  Prompt: {prompt[:50]}...")
            print(f"  Hallucination Detected: {data['hallucination_detected']}")
            print(f"  Hallucination Score: {data['hallucination_score']:.2f}")
            print(f"  Response Time: {data['response_time']:.2f}s")
            print()
            return data
        else:
            print(f"✗ Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"✗ Connection error: {e}")
        return None

def simulate_false_positives():
    """Simulate false positive detections"""
    try:
        response = requests.post(f"{LLM_APP_URL}/simulate/false-positive")
        if response.status_code == 200:
            print("✓ False positive simulated")
    except Exception as e:
        print(f"✗ Error simulating false positive: {e}")

def simulate_false_negatives():
    """Simulate false negative detections"""
    try:
        response = requests.post(f"{LLM_APP_URL}/simulate/false-negative")
        if response.status_code == 200:
            print("✓ False negative simulated")
    except Exception as e:
        print(f"✗ Error simulating false negative: {e}")

def simulate_sessions():
    """Simulate session management"""
    try:
        # Start sessions
        for _ in range(5):
            requests.post(f"{LLM_APP_URL}/simulate/session/start")
        
        print("✓ Sessions started")
        
        # End some sessions
        for _ in range(2):
            requests.post(f"{LLM_APP_URL}/simulate/session/end")
            
        print("✓ Some sessions ended")
        
    except Exception as e:
        print(f"✗ Error simulating sessions: {e}")

def check_service_health():
    """Check if all services are running"""
    services = {
        "LLM App": LLM_APP_URL,
        "Prometheus": PROMETHEUS_URL,
        "Grafana": GRAFANA_URL
    }
    
    print("🔍 Checking service health...")
    all_healthy = True
    
    for service_name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✓ {service_name} is healthy")
            else:
                print(f"✗ {service_name} returned status {response.status_code}")
                all_healthy = False
        except Exception as e:
            print(f"✗ {service_name} is not reachable: {e}")
            all_healthy = False
    
    return all_healthy

def run_demo():
    """Run the complete demo"""
    print("🚀 LLM Hallucination Monitoring Demo")
    print("=" * 50)
    
    # Check if services are running
    if not check_service_health():
        print("\n❌ Some services are not running. Please start the system first:")
        print("   docker-compose up -d")
        return
    
    print("\n📊 Starting demo traffic generation...")
    print("This will generate various types of requests to demonstrate monitoring.\n")
    
    # Phase 1: Normal requests
    print("🟢 Phase 1: Normal requests (low hallucination risk)")
    for i in range(10):
        prompt = random.choice(NORMAL_PROMPTS)
        model = random.choice(["gpt-3.5-turbo", "gpt-4"])
        send_request(prompt, model)
        time.sleep(1)
    
    # Phase 2: High-risk prompts
    print("🔴 Phase 2: High-risk prompts (potential hallucinations)")
    for i in range(8):
        prompt = random.choice(HALLUCINATION_PROMPTS)
        model = random.choice(["gpt-3.5-turbo", "gpt-4"])
        send_request(prompt, model)
        time.sleep(1.5)
    
    # Phase 3: Mixed traffic
    print("🟡 Phase 3: Mixed traffic pattern")
    for i in range(15):
        if random.random() < 0.3:  # 30% chance of hallucination prompt
            prompt = random.choice(HALLUCINATION_PROMPTS)
        else:
            prompt = random.choice(NORMAL_PROMPTS)
        
        model = random.choice(["gpt-3.5-turbo", "gpt-4"])
        send_request(prompt, model)
        time.sleep(random.uniform(0.5, 2.0))
    
    # Phase 4: Simulate false positives/negatives
    print("🔧 Phase 4: Simulating detection errors")
    simulate_false_positives()
    simulate_false_negatives()
    
    # Phase 5: Session simulation
    print("👥 Phase 5: Simulating user sessions")
    simulate_sessions()
    
    print("\n✅ Demo completed!")
    print("\n📈 View the results:")
    print(f"   Grafana Dashboard: {GRAFANA_URL}")
    print("   Login: admin/admin")
    print(f"   Prometheus: {PROMETHEUS_URL}")
    print(f"   LLM App: {LLM_APP_URL}")
    
    print("\n🔍 What to look for in Grafana:")
    print("   1. Hallucination Rate spikes during Phase 2")
    print("   2. Model comparison showing GPT-4 vs GPT-3.5 performance")
    print("   3. Detection latency patterns")
    print("   4. False positive/negative rates")
    print("   5. Active sessions and request rates")

if __name__ == "__main__":
    run_demo()

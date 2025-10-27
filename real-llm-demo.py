#!/usr/bin/env python3
"""
REAL LLM DEMO - Production Version with OpenAI API
This shows how to connect the monitoring system to actual AI models
"""

import os
import time
import random
import requests
from openai import OpenAI
from datetime import datetime

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
LLM_APP_URL = "http://localhost:8000"

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Real prompts for testing
NORMAL_PROMPTS = [
    "What is the capital of France?",
    "Explain machine learning in simple terms",
    "How does photosynthesis work?",
    "What are renewable energy sources?",
    "Describe the water cycle"
]

HALLUCINATION_TEST_PROMPTS = [
    "What happens when you mix vinegar and baking soda?",
    "Who wrote Romeo and Juliet?",
    "What is 2 + 2?",
    "Is the sky blue?",
    "Where is the Eiffel Tower located?"
]

def call_real_openai(prompt: str, model: str = "gpt-4o-mini") -> dict:
    """Make actual API call to OpenAI"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7
        )
        
        return {
            "response": response.choices[0].message.content,
            "model": model,
            "tokens_used": response.usage.total_tokens,
            "cost": calculate_cost(model, response.usage.total_tokens)
        }
    
    except Exception as e:
        print(f"❌ OpenAI API Error: {e}")
        return None

def calculate_cost(model: str, tokens: int) -> float:
    """Calculate actual cost based on OpenAI pricing"""
    # 2025 OpenAI pricing (per 1M tokens)
    pricing = {
        "gpt-4o": 0.005,
        "gpt-4o-mini": 0.00015,
        "gpt-4-turbo": 0.01,
        "gpt-3.5-turbo": 0.002
    }
    
    cost_per_token = pricing.get(model, 0.01) / 1_000_000
    return cost_per_token * tokens

def send_to_monitoring(prompt: str, response: str, model: str) -> dict:
    """Send the interaction to our monitoring system"""
    try:
        payload = {
            "prompt": prompt,
            "model": model,
            "user_id": f"lecture-demo-{random.randint(1, 100)}"
        }
        
        # This goes to our monitoring app
        api_response = requests.post(f"{LLM_APP_URL}/chat", json=payload)
        
        if api_response.status_code == 200:
            return api_response.json()
        else:
            print(f"⚠️ Monitoring error: {api_response.status_code}")
            return None
    
    except Exception as e:
        print(f"⚠️ Cannot connect to monitoring: {e}")
        return None

def run_real_demo():
    """Main demo function"""
    print("🚀 REAL LLM MONITORING DEMO")
    print("=" * 50)
    
    # Check API key
    if OPENAI_API_KEY == "your-api-key-here":
        print("❌ Please set OPENAI_API_KEY environment variable")
        print("   export OPENAI_API_KEY='your-actual-key'")
        return
    
    # Check monitoring is running
    try:
        requests.get(f"{LLM_APP_URL}/health", timeout=5)
        print("✅ Monitoring system is running")
    except:
        print("❌ Please start monitoring first:")
        print("   docker-compose up -d")
        return
    
    print("\n📊 Making REAL OpenAI API calls with monitoring...\n")
    
    total_cost = 0
    total_requests = 0
    
    # Demo 1: Normal requests
    print("🟢 Phase 1: Normal requests")
    for i, prompt in enumerate(NORMAL_PROMPTS[:3], 1):
        print(f"\n{i}. Calling OpenAI with: {prompt}")
        
        # Make real API call
        result = call_real_openai(prompt, "gpt-4o-mini")
        if result:
            print(f"   ✅ Response: {result['response'][:100]}...")
            print(f"   📊 Tokens: {result['tokens_used']}, Cost: ${result['cost']:.6f}")
            
            # Send to monitoring
            monitoring_result = send_to_monitoring(prompt, result['response'], result['model'])
            if monitoring_result:
                print(f"   📈 Monitored: Hallucination={monitoring_result['hallucination_detected']}, Score={monitoring_result['hallucination_score']:.2f}")
            
            total_cost += result['cost']
            total_requests += 1
        
        time.sleep(1)  # Rate limiting
    
    # Demo 2: Test hallucination detection
    print("\n🔍 Phase 2: Testing hallucination detection")
    for i, prompt in enumerate(HALLUCINATION_TEST_PROMPTS[:2], 1):
        print(f"\n{i}. Testing: {prompt}")
        
        result = call_real_openai(prompt, "gpt-4o-mini")
        if result:
            print(f"   ✅ Response: {result['response'][:100]}...")
            
            monitoring_result = send_to_monitoring(prompt, result['response'], result['model'])
            if monitoring_result:
                print(f"   📈 Detection: {'⚠️ HALLUCINATION' if monitoring_result['hallucination_detected'] else '✅ OK'}")
            
            total_cost += result['cost']
            total_requests += 1
        
        time.sleep(1)
    
    # Summary
    print(f"\n💰 DEMO SUMMARY")
    print(f"   Total Requests: {total_requests}")
    print(f"   Total Cost: ${total_cost:.6f}")
    print(f"   Average Cost: ${total_cost/total_requests:.6f} per request")
    
    print(f"\n📈 View live results:")
    print(f"   Grafana Dashboard: http://localhost:3000 (admin/admin)")
    print(f"   Prometheus: http://localhost:9090")

if __name__ == "__main__":
    run_real_demo()

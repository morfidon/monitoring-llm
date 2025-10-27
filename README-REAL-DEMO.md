# Real LLM Demo - Production Version

## Purpose
This demo shows how to connect the monitoring system to actual OpenAI APIs instead of simulated responses.

## Quick Start

### 1. Set Up OpenAI API Key
```bash
# Windows PowerShell
$env:OPENAI_API_KEY = "your-actual-openai-api-key"

# Linux/Mac
export OPENAI_API_KEY="your-actual-openai-api-key"
```

### 2. Start Monitoring System
```bash
docker-compose -f docker-compose-real.yml up -d
```

### 3. Install Real Dependencies
```bash
pip install -r requirements-real.txt
```

### 4. Run Real Demo
```bash
python real-llm-demo.py
```

## What This Demo Shows

### Real API Integration
- Actual OpenAI API calls
- Real token usage tracking  
- Actual cost calculation
- Live monitoring of real responses

### Monitoring Features
- Real-time hallucination detection
- Cost tracking per request
- Response time monitoring
- Model performance comparison

## Cost Examples

### GPT-4o-mini (Most Cost-Effective)
- Price: $0.00015 per 1M tokens
- Typical request: ~50 tokens = $0.000008
- 1000 requests: ~$0.08

### GPT-4o (Balanced)
- Price: $0.005 per 1M tokens  
- Typical request: ~50 tokens = $0.00025
- 1000 requests: ~$0.25

### GPT-4-turbo (High Performance)
- Price: $0.01 per 1M tokens
- Typical request: ~50 tokens = $0.00050
- 1000 requests: ~$0.50

## Lecture Demo Flow

### Step 1: Show Simulated Version
```bash
python demo.py
```
- No API costs
- Demonstrates monitoring architecture
- Shows 2025 model comparisons

### Step 2: Show Real Version  
```bash
python real-llm-demo.py
```
- Real OpenAI API calls
- Actual costs incurred
- Live monitoring of real AI

### Step 3: Compare in Grafana
- Visit http://localhost:3000
- See real vs simulated metrics
- Show cost tracking panels

## Key Differences

| Feature | Simulated Demo | Real Demo |
|---------|----------------|-----------|
| API Calls | Fake responses | Real OpenAI API |
| Cost | $0.00 | Actual usage |
| Responses | Pre-programmed | Genuine AI |
| Tokens | Simulated | Real counting |
| Monitoring | Real | Real |

## Important Notes

### API Safety
- Demo uses limited tokens (150 max)
- Rate limiting included (1 second delays)
- Only 5 requests in demo = ~$0.001 cost

### For Production
- Add proper error handling
- Implement retry logic  
- Add user authentication
- Set up cost alerts
- Use async for better performance

### Monitoring Benefits
- Catch hallucinations in real-time
- Track costs per user/model
- Monitor response times
- Compare model performance
- Set up alerts for issues

## Perfect for Lecture

1. Start with simulated demo (free, shows architecture)
2. Switch to real demo (shows actual integration)
3. View live Grafana (compare real vs simulated)
4. Discuss costs (real token usage)
5. Show production readiness (monitoring stack)

Your students see the complete journey from simulation to production!

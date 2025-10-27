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

### 2. Start Monitoring System with Docker
```bash
docker-compose -f docker-compose-real.yml up -d
```
This starts the LLM app with OpenAI integration, Prometheus, and Grafana.

### 3. Run Real Demo
```bash
# Option A: Use the integrated app (auto-detects API key)
python demo.py

# Option B: Use standalone real demo script
python real-llm-demo.py
```

## What This Demo Shows

### Real API Integration
- Actual OpenAI API calls (when API key is provided)
- Real token usage tracking  
- Actual cost calculation
- Live monitoring of real responses

### Smart Auto-Detection
The system automatically detects if you have an OpenAI API key:
- **No key** → Uses simulation (free)
- **Has key** → Uses real OpenAI API for supported models

## Cost Examples

### GPT-4o-mini (Most Cost-Effective)
- Price: $0.00015 per 1M tokens
- Typical request: ~50 tokens = $0.000008
- 1000 requests: ~$0.08

### GPT-4o (Balanced)
- Price: $0.005 per 1M tokens  
- Typical request: ~50 tokens = $0.00025
- 1000 requests: ~$0.25

## Lecture Demo Flow

### Step 1: Show Simulated Version
```bash
docker-compose up -d
python demo.py
```
- No API costs
- Demonstrates monitoring architecture
- Shows 2025 model comparisons

### Step 2: Show Real Version  
```bash
# Set API key
$env:OPENAI_API_KEY = "your-key"

# Start with real compose file
docker-compose -f docker-compose-real.yml up -d

# Run demo (auto-uses real API)
python demo.py
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

### Docker Benefits
- No local Python setup needed
- Consistent environment everywhere
- Easy cleanup with `docker-compose down`
- All dependencies included in containers

### Monitoring Benefits
- Catch hallucinations in real-time
- Track costs per user/model
- Monitor response times
- Compare model performance
- Set up alerts for issues

## Perfect for Lecture

1. Start with simulated demo (free, shows architecture)
2. Add API key and restart (shows real integration)
3. View live Grafana (compare real vs simulated)
4. Discuss costs (real token usage)
5. Show production readiness (monitoring stack)

Your students see the complete journey from simulation to production!

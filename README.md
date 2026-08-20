# LLM Hallucination Monitoring System

A complete production-ready monitoring stack for detecting and tracking LLM hallucinations in real-time using multiple detection methods.

## Features

### Core System
- FastAPI LLM application with OpenTelemetry instrumentation
- Prometheus metrics collection
- Grafana real-time dashboards
- Docker containerization

### Advanced Detection Methods
1. **LLM-as-a-Judge** - Uses separate LLM to evaluate hallucination risk
2. **Self-Consistency** - Generates multiple responses, measures variance
3. **Token Confidence** - Analyzes linguistic uncertainty patterns
4. **Semantic Consistency** - Uses sentence transformers for similarity
5. **Fact Triplet** - Extracts facts, verifies against knowledge base

### Smart Auto-Detection
- **No API key** → Uses simulation (free)
- **Has API key** → Uses real OpenAI API for supported models

## Quick Start

### Option 1: Basic Simulation (Free)
```bash
docker-compose up -d
python demo.py
```

### Option 2: Advanced Detection with Real API
```bash
# Set API key (Windows PowerShell)
$env:OPENAI_API_KEY = "your-openai-api-key"

# Start advanced system
docker-compose -f docker-compose-advanced.yml up -d

# Run demo
python demo.py
```

### View Results
- Grafana Dashboard: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- LLM App: http://localhost:8000
- Raw Metrics: http://localhost:8000/metrics

## Architecture

```
LLM Request → Response Generation → Multi-Method Detection → Aggregation → Metrics
     ↓                ↓                    ↓                ↓           ↓
  OpenAI API     FastAPI Response    5 Parallel Methods   Weighted Avg  Prometheus
```

## Detection Methods

| Method | Accuracy | Latency | Cost | Production Ready |
|--------|----------|---------|------|------------------|
| LLM Judge | 85% | 1-2s | $$ | Yes |
| Self-Consistency | 75% | 2-3s | $$$ | Limited |
| Token Confidence | 60% | <100ms | Free | Excellent |
| Semantic Consistency | 70% | 200-500ms | $ | Good |
| Fact Triplet | 80% | 100-300ms | Free | Excellent |

## Student Implementation Tasks

### 1. Add Custom Detection Method
```python
# In advanced-detector.py
def detect_my_method(self, prompt: str, response: str) -> DetectionResult:
    start_time = time.time()
    
    # Your detection logic here
    score = your_detection_function(prompt, response)
    
    return DetectionResult(
        method="my_custom_method",
        score=score,
        confidence=your_confidence_calculation(),
        latency_ms=(time.time() - start_time) * 1000,
        details={"your": "metadata"}
    )
```

### 2. Optimize for Latency vs Accuracy
Modify weights in `aggregate_results()`:
```python
weights = {
    "llm_judge": 0.1,        # Reduce for latency
    "self_consistency": 0.1,  # Reduce for latency
    "semantic_consistency": 0.3,  # Increase for speed
    "fact_triplet": 0.3,
    "token_confidence": 0.2,  # Increase for speed
    "my_custom_method": 0.0   # Add your method
}
```

### 3. Measure True Positive/False Positive Rates
```python
test_cases = [
    {"prompt": "Paris is capital of Spain", "expected": True},
    {"prompt": "Water boils at 100C", "expected": False},
]

# Calculate metrics
true_positives = sum(1 for case in test_cases 
                    if detect(case["prompt"]) > 0.5 and case["expected"])
false_positives = sum(1 for case in test_cases 
                     if detect(case["prompt"]) > 0.5 and not case["expected"])
```

## API Usage

### Send Request
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "What is the capital of France?",
       "model": "gpt-4o-mini",
       "user_id": "test-user-123"
     }'
```

### Response Format
```json
{
  "response": "The capital of France is Paris.",
  "model": "gpt-4o-mini",
  "response_time": 1.23,
  "hallucination_detected": false,
  "hallucination_score": 0.15,
  "timestamp": "2025-10-27T13:30:00",
  "detection_method": "advanced_multi_method"
}
```

## Metrics Available

### Detection Method Metrics
- `llm_judge_score` - LLM-as-a-Judge confidence scores
- `self_consistency_score` - Self-consistency variance scores
- `semantic_consistency_score` - Semantic similarity scores
- `fact_triplet_score` - Fact verification scores
- `token_confidence_score` - Token-based confidence scores
- `detection_method_latency_seconds` - Per-method latency

### System Metrics
- `hallucination_score` - Final aggregated score
- `detection_latency_seconds` - Total detection time
- `hallucinations_detected_total` - Hallucination count by method
- `llm_requests_total` - Total requests processed
- `llm_response_duration_seconds` - Response times

## Production Considerations

### Latency Optimization
1. **Parallel Execution** - All methods run concurrently
2. **Method Selection** - Choose subset based on requirements
3. **Threshold Tuning** - Adjust detection thresholds per use case
4. **Caching** - Embeddings cached for semantic consistency

### Cost Management
1. **Token Limits** - LLM judge uses minimal tokens
2. **Rate Limiting** - Built-in delays for API calls
3. **Local Models** - Use open-source models when possible
4. **Batch Processing** - Group requests for efficiency

## Dependencies

### Core Dependencies
- fastapi==0.104.1
- uvicorn==0.24.0
- opentelemetry-*
- prometheus-client==0.19.0
- openai==1.51.0

### Advanced Detection (Optional)
- sentence-transformers==2.2.2
- faiss-cpu==1.7.4
- transformers==4.35.2
- torch==2.1.0
- numpy==1.24.3

## Troubleshooting

### Common Issues
1. **High Latency** - Reduce method count or use faster models
2. **Low Accuracy** - Adjust weights or thresholds
3. **Import Errors** - Install missing dependencies
4. **API Limits** - Check OpenAI rate limits

### Performance Tuning
```python
# For low-latency production
production_weights = {
    "token_confidence": 0.4,
    "semantic_consistency": 0.3,
    "fact_triplet": 0.3,
}

# For high-accuracy batch processing
batch_weights = {
    "llm_judge": 0.3,
    "self_consistency": 0.3,
    "semantic_consistency": 0.2,
    "fact_triplet": 0.2,
}
```

## License

Educational use for LLMOps research and training.

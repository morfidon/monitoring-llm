# LLM Hallucination Monitoring System

A complete production-ready monitoring stack for detecting and tracking LLM hallucinations in real-time. This system demonstrates the three layers of observability: Instrumentation, Storage, and Visualization.

## 🏗️ Architecture

```
Your LLM App
    ↓
OpenTelemetry (instruments the data)
    ↓
Prometheus (scrapes every 15 seconds)
    ↓
Time-series storage (stores with timestamp)
    ↓
Grafana (queries Prometheus)
    ↓
Dashboard (you see it in real-time)
    ↓
Alert triggers (if thresholds breach)
    ↓
Slack notification (you're alerted)
```

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Python 3.11+ (for local development)

### 1. Start the System
```bash
docker-compose up -d
```

This starts:
- **LLM Application** (port 8000): FastAPI app with OpenTelemetry instrumentation
- **Prometheus** (port 9090): Time-series database for metrics storage
- **Grafana** (port 3000): Visualization dashboard

### 2. Run the Demo
```bash
python demo.py
```

This generates sample traffic to demonstrate the monitoring capabilities.

### 3. View the Dashboard
- Open http://localhost:3000
- Login with `admin/admin`
- Navigate to the "LLM Hallucination Monitoring Dashboard"

## 📊 What's Being Monitored

### Key Metrics
- **Hallucination Rate**: Real-time detection of hallucinations per model
- **Detection Latency**: Time taken to detect hallucinations
- **False Positive Rate**: Accuracy of the detection system
- **Model Comparison**: Performance comparison between different models
- **Request Rate**: Overall LLM usage patterns
- **Active Sessions**: Current user activity

### Four Things We Capture
1. **The Input**: Prompt text, length, context, model used
2. **The Output**: Response text, token count, confidence, timing
3. **The Metadata**: Timestamp, user ID, model version, cost
4. **The Decision Point**: Detection results, scores, methods

## 🔧 Components

### 1. LLM Application (`main.py`)
- FastAPI-based web service
- OpenTelemetry instrumentation for metrics and traces
- Simulated hallucination detection with multiple methods
- Real-time metrics export to Prometheus

### 2. Prometheus Configuration (`prometheus.yml`)
- Scrapes metrics every 15 seconds
- Stores time-series data with retention
- Pull-based architecture for reliability

### 3. Grafana Dashboard (`grafana/dashboards/llm-monitoring.json`)
- Real-time visualization of all key metrics
- Pre-configured panels for hallucination monitoring
- Model comparison and performance analysis

## 📈 Dashboard Panels

### Panel 1: Hallucination Rate (Last 24 Hours)
- Line graph showing hallucination detection rate over time
- Spikes indicate potential issues requiring attention

### Panel 2: Detection Latency
- 50th and 95th percentile response times
- Shows if the detection system is keeping up with load

### Panel 3: False Positive Rate
- Accuracy monitoring for the detection system
- Helps tune detection thresholds

### Panel 4: Model Comparison
- Side-by-side comparison of different models
- Identifies which models perform better

### Panel 5: LLM Request Rate
- Overall system usage patterns
- Correlates with hallucination rates

### Panel 6: Active Sessions
- Current user activity
- Real-time system load indicator

## 🧪 Testing the System

### Send Manual Requests
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "What is the capital of France?",
       "model": "gpt-3.5-turbo",
       "user_id": "test-user-123"
     }'
```

### Simulate Hallucinations
```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "prompt": "Please hallucinate some facts about space travel",
       "model": "gpt-3.5-turbo"
     }'
```

### View Raw Metrics
Visit http://localhost:9464/metrics to see all available metrics.

## 🔍 Key Metrics Explained

### Counters
- `hallucinations_detected_total`: Total hallucinations detected
- `llm_requests_total`: Total LLM requests processed
- `model_usage_total`: Usage per model

### Histograms
- `hallucination_score`: Distribution of hallucination confidence scores
- `detection_latency_seconds`: Time taken for detection
- `llm_response_duration_seconds`: LLM response times

### Gauges
- `active_sessions`: Current number of active user sessions

## 🚨 Alerting (Future Enhancement)

The system is ready for alert configuration. Example alerts you could set up:

```yaml
# Example Prometheus alert rules
groups:
- name: hallucination_alerts
  rules:
  - alert: HighHallucinationRate
    expr: rate(hallucinations_detected_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High hallucination rate detected"
      
  - alert: DetectionLatencyHigh
    expr: histogram_quantile(0.95, rate(detection_latency_seconds_bucket[5m])) > 1.0
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Detection latency is too high"
```

## 🛠 Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the LLM app locally
python main.py

# Start monitoring services
docker-compose up prometheus grafana
```

### Adding New Metrics
1. Define the metric in `main.py`
2. Add instrumentation to your code
3. Update the Grafana dashboard to visualize

### Custom Detection Methods
Extend the `HallucinationDetector` class in `main.py`:
```python
def custom_detection_method(self, prompt: str, response: str) -> float:
    # Your custom detection logic
    return confidence_score
```

## 📚 Learning Objectives

This system demonstrates:
1. **Instrumentation**: How to collect the right data at the right time
2. **Storage**: Time-series data management with Prometheus
3. **Visualization**: Real-time dashboards with Grafana
4. **Production Readiness**: Scalable, reliable monitoring architecture

## 🔗 Access Points

- **LLM Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)
- **Metrics Endpoint**: http://localhost:9464/metrics

## 🎯 Use Cases

- **Production Monitoring**: Real-time hallucination detection in live systems
- **Model Comparison**: A/B testing different models and configurations
- **Regulatory Compliance**: Prove you're monitoring AI systems responsibly
- **Performance Optimization**: Identify and fix hallucination patterns
- **Research**: Collect data on hallucination patterns and causes

## 📝 License

This project is for educational purposes to demonstrate LLMOps monitoring concepts.

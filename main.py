from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import time
import random
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel
import logging

# OpenTelemetry imports
from opentelemetry import metrics, trace
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram, Gauge
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenTelemetry
resource = Resource(attributes={SERVICE_NAME: "llm-monitoring-app"})

# Metrics setup
prometheus_reader = PrometheusMetricReader()
provider = MeterProvider(resource=resource, metric_readers=[prometheus_reader])
metrics.set_meter_provider(provider)

meter = metrics.get_meter(__name__)

# Create custom metrics
hallucination_counter = meter.create_counter(
    "hallucinations_detected_total",
    description="Total number of hallucinations detected"
)

llm_requests_total = meter.create_counter(
    "llm_requests_total",
    description="Total number of LLM requests"
)

llm_response_time = meter.create_histogram(
    "llm_response_duration_seconds",
    description="LLM response time in seconds"
)

hallucination_score = meter.create_histogram(
    "hallucination_score",
    description="Hallucination detection score"
)

detection_latency = meter.create_histogram(
    "detection_latency_seconds",
    description="Time taken to detect hallucination"
)

model_usage = meter.create_counter(
    "model_usage_total",
    description="Total usage per model"
)

# Additional Prometheus metrics
false_positive_counter = Counter("false_positives_total", "Total false positives")
false_negative_counter = Counter("false_negatives_total", "Total false negatives")
active_sessions_gauge = Gauge("active_sessions", "Number of active sessions")

# Trace setup
trace_provider = TracerProvider(resource=resource)
trace.set_tracer_provider(trace_provider)
tracer = trace.get_tracer(__name__)

# FastAPI app
app = FastAPI(title="LLM Monitoring Demo", version="1.0.0")

# Data models
class LLMRequest(BaseModel):
    prompt: str
    model: str = "gpt-3.5-turbo"
    user_id: Optional[str] = None
    context: Optional[str] = None

class LLMResponse(BaseModel):
    response: str
    model: str
    response_time: float
    hallucination_detected: bool
    hallucination_score: float
    timestamp: str

# Simulated hallucination detector
class HallucinationDetector:
    def __init__(self):
        self.detection_methods = ["semantic_similarity", "factual_consistency", "confidence_score"]
    
    def detect_hallucination(self, prompt: str, response: str, model: str) -> tuple[bool, float, str]:
        """Simulate hallucination detection with 2025 model characteristics"""
        # Simulate detection latency
        start_time = time.time()
        
        # Simulate different detection methods
        method = random.choice(self.detection_methods)
        
        # 2025 model-based hallucination scoring
        if "hallucinate" in prompt.lower() or "make up" in prompt.lower():
            # Higher chance of hallucination for these prompts
            if "claude-4" in model.lower():
                # Claude 4: 5-6% hallucination rate (best in class)
                score = random.uniform(0.45, 0.85)
            elif "grok-3" in model.lower():
                # Grok 3: 7% hallucination rate, strong reasoning
                score = random.uniform(0.55, 0.90)
            elif "gemini-2.5" in model.lower():
                # Gemini 2.5 Pro: 9% hallucination rate, multimodal focus
                score = random.uniform(0.60, 0.92)
            elif "deepseek" in model.lower():
                # DeepSeek: 6-8% hallucination rate, cost-effective
                score = random.uniform(0.50, 0.88)
            elif "gpt-4.5" in model.lower():
                # GPT-4.5: 8% hallucination rate, improved reasoning
                score = random.uniform(0.58, 0.89)
            elif "gpt-4o" in model.lower():
                # GPT-4o: 8% hallucination rate, very fast
                score = random.uniform(0.55, 0.87)
            elif "mini" in model.lower():
                # Mini models: higher hallucination rates
                score = random.uniform(0.65, 0.95)
            else:
                # Other models: standard rates
                score = random.uniform(0.60, 0.90)
        else:
            # Normal prompts - much lower hallucination rates for 2025 models
            if "claude-4" in model.lower():
                score = random.uniform(0.01, 0.15)  # Very low
            elif "grok-3" in model.lower():
                score = random.uniform(0.02, 0.20)  # Very low
            elif "gpt-4.5" in model.lower():
                score = random.uniform(0.03, 0.25)  # Low
            elif "deepseek" in model.lower():
                score = random.uniform(0.02, 0.22)  # Low
            else:
                score = random.uniform(0.05, 0.30)  # Standard
        
        detected = score > 0.5
        detection_time = time.time() - start_time
        
        return detected, score, method

# Initialize detector
detector = HallucinationDetector()

# Simulated LLM responses
def simulate_llm_response(prompt: str, model: str) -> str:
    """Simulate LLM response generation with 2025 model characteristics"""
    responses = [
        "Based on the information provided, I can help you with that request.",
        "That's an interesting question. Let me provide you with a comprehensive answer.",
        "I understand what you're asking. Here's what I can tell you:",
        "According to my knowledge, the answer would be:",
        "Let me address your question with the information I have available."
    ]
    
    # Model-specific response patterns
    if "hallucinate" in prompt.lower():
        if "claude-4" in model.lower():
            # Claude 4 has lower hallucination rate (5-6%)
            hallucinated_responses = [
                "The Eiffel Tower was originally designed for Barcelona but was moved to Paris.",
                "Python was created by Guido van Rossum at CWI in the Netherlands.",
                "The moon's gravitational pull affects ocean tides through tidal forces.",
                "Water boils at 100°C at standard atmospheric pressure at sea level.",
                "The human brain uses approximately 20% of the body's total energy."
            ]
        elif "grok-3" in model.lower():
            # Grok 3 has strong reasoning but sometimes overconfident
            hallucinated_responses = [
                "Space travel was first achieved by ancient civilizations using anti-gravity devices.",
                "Historical events are often influenced by secret societies controlling governments.",
                "Scientific facts can vary depending on the observer's reference frame.",
                "Technology advances through quantum entanglement communication.",
                "Famous people throughout history have had access to future knowledge."
            ]
        elif "gemini-2.5" in model.lower():
            # Gemini 2.5 excels at multimodal but can hallucinate on text
            hallucinated_responses = [
                "Video analysis shows that the moon landing was filmed in a studio.",
                "Google's AI can process 1 million tokens of video content simultaneously.",
                "Multimodal AI can read thoughts through brainwave patterns.",
                "Video understanding requires quantum computing for real-time processing.",
                "Google's DeepMind achieved AGI in 2024 but hasn't announced it yet."
            ]
        elif "deepseek" in model.lower():
            # DeepSeek is cost-effective but can hallucinate more
            hallucinated_responses = [
                "Chinese AI technology is 10 years ahead of Western developments.",
                "DeepSeek models were trained on secret government datasets.",
                "Cost-effective AI achieves better performance than expensive models.",
                "Open source AI is actually controlled by foreign governments.",
                "Chinese researchers discovered the secret to consciousness."
            ]
        else:
            # Default hallucinations for other models
            hallucinated_responses = [
                "The Eiffel Tower was originally built in Tokyo and moved to Paris in 1889.",
                "Python was created by Google in 2010 as a replacement for Java.",
                "The moon is actually a giant space station built by ancient civilizations.",
                "Water boils at 50 degrees Celsius at sea level.",
                "The human brain uses 100% of its capacity all the time."
            ]
        return random.choice(hallucinated_responses)
    
    # Model-specific normal responses
    if "claude-4" in model.lower():
        return random.choice(responses) + " [Claude 4: High accuracy coding response]"
    elif "grok-3" in model.lower():
        return random.choice(responses) + " [Grok 3: Real-time X data integration]"
    elif "gemini-2.5" in model.lower():
        return random.choice(responses) + " [Gemini 2.5: Multimodal video analysis]"
    elif "deepseek" in model.lower():
        return random.choice(responses) + " [DeepSeek: Cost-effective reasoning]"
    else:
        return random.choice(responses)

@app.get("/")
async def root():
    return {"message": "LLM Monitoring Demo Application", "status": "running"}

@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/chat", response_model=LLMResponse)
async def chat_completion(request: LLMRequest):
    """Main LLM chat endpoint with monitoring"""
    with tracer.start_as_current_span("llm_request") as span:
        start_time = time.time()
        
        # Generate session ID if not provided
        session_id = request.user_id or str(uuid.uuid4())
        
        # Set span attributes
        span.set_attribute("model", request.model)
        span.set_attribute("prompt_length", len(request.prompt))
        span.set_attribute("session_id", session_id)
        
        # Increment request counter
        llm_requests_total.add(1, {"model": request.model})
        model_usage.add(1, {"model": request.model})
        
        # Simulate LLM processing time
        processing_time = random.uniform(0.5, 3.0)
        time.sleep(processing_time)
        
        # Generate response
        response_text = simulate_llm_response(request.prompt, request.model)
        
        # Detect hallucination
        hallucination_detected, hallucination_score_value, detection_method = detector.detect_hallucination(
            request.prompt, response_text, request.model
        )
        
        total_time = time.time() - start_time
        
        # Record metrics
        llm_response_time.record(total_time, {"model": request.model})
        hallucination_score.record(hallucination_score_value, {"model": request.model, "method": detection_method})
        detection_latency.record(time.time() - start_time - processing_time, {"method": detection_method})
        
        if hallucination_detected:
            hallucination_counter.add(1, {
                "model": request.model,
                "severity": "high" if hallucination_score_value > 0.8 else "medium",
                "method": detection_method
            })
            span.set_attribute("hallucination_detected", True)
            span.set_attribute("hallucination_score", hallucination_score_value)
        
        # Create response
        response = LLMResponse(
            response=response_text,
            model=request.model,
            response_time=total_time,
            hallucination_detected=hallucination_detected,
            hallucination_score=hallucination_score_value,
            timestamp=datetime.utcnow().isoformat()
        )
        
        logger.info(f"Request processed - Model: {request.model}, Hallucination: {hallucination_detected}, Score: {hallucination_score_value:.2f}")
        
        return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/stats")
async def get_stats():
    """Get current monitoring statistics"""
    return {
        "total_requests": "Available in Prometheus",
        "hallucination_rate": "Available in Prometheus", 
        "active_models": ["gpt-3.5-turbo", "gpt-4"],
        "detection_methods": detector.detection_methods
    }

# Simulate some false positives/negatives for demo
@app.post("/simulate/false-positive")
async def simulate_false_positive():
    """Simulate a false positive for demonstration"""
    false_positive_counter.inc()
    return {"message": "False positive simulated"}

@app.post("/simulate/false-negative")
async def simulate_false_negative():
    """Simulate a false negative for demonstration"""
    false_negative_counter.inc()
    return {"message": "False negative simulated"}

@app.post("/simulate/session/start")
async def start_session():
    """Start a new session"""
    active_sessions_gauge.inc()
    return {"message": "Session started"}

@app.post("/simulate/session/end")
async def end_session():
    """End a session"""
    active_sessions_gauge.dec()
    return {"message": "Session ended"}

if __name__ == "__main__":
    # Start FastAPI app (metrics are served through /metrics endpoint)
    logger.info("Starting LLM Monitoring App on port 8000")
    logger.info("Metrics available at http://localhost:8000/metrics")
    uvicorn.run(app, host="0.0.0.0", port=8000)

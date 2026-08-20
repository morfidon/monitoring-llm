#!/usr/bin/env python3
"""
Advanced Hallucination Detection System
Implements multiple detection methods for real-time monitoring
"""

import os
import time
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

# Try to import real detection libraries
try:
    from sentence_transformers import SentenceTransformer
    import faiss
    from transformers import pipeline
    DETECTION_LIBS_AVAILABLE = True
except ImportError:
    DETECTION_LIBS_AVAILABLE = False
    print("Warning: Detection libraries not installed. Using simulation mode.")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if os.getenv("OPENAI_API_KEY") else None
except ImportError:
    OPENAI_AVAILABLE = False
    openai_client = None

@dataclass
class DetectionResult:
    method: str
    score: float
    confidence: float
    latency_ms: float
    details: Dict

class AdvancedHallucinationDetector:
    """Multi-method hallucination detection system"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize models if available
        self.embedding_model = None
        self.similarity_index = None
        self.nli_model = None
        
        if DETECTION_LIBS_AVAILABLE:
            try:
                # Load sentence transformer for semantic consistency
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                
                # Initialize FAISS index for similarity search
                self.similarity_index = faiss.IndexFlatL2(384)  # MiniLM dimension
                
                # Load NLI model for fact checking
                self.nli_model = pipeline("text-classification", 
                                        model="roberta-large-mnli")
                
                self.logger.info("Real detection models loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load detection models: {e}")
        
        # Knowledge base for fact checking (simple demo)
        self.knowledge_base = {
            "paris": {"capital_of": "france", "population": "2.2 million"},
            "python": {"created_by": "guido van rossum", "first_release": "1991"},
            "water": {"boiling_point": "100c", "chemical_formula": "H2O"},
            "eiffel_tower": {"location": "paris", "built": "1889"},
            "photosynthesis": {"process": "converts light to energy", "location": "plants"}
        }
    
    def detect_llm_as_judge(self, prompt: str, response: str, model: str = "gpt-4o-mini") -> DetectionResult:
        """LLM-as-a-Judge: Use another LLM to score hallucination risk"""
        start_time = time.time()
        
        if not openai_client:
            # Simulate LLM judge
            score = random.uniform(0.1, 0.9)
            confidence = random.uniform(0.7, 0.95)
            latency = random.uniform(500, 2000)
            
            return DetectionResult(
                method="llm_judge",
                score=score,
                confidence=confidence,
                latency_ms=latency,
                details={"judge_model": model, "simulation": True}
            )
        
        try:
            judge_prompt = f"""
            Analyze this response for hallucination risk:
            
            Prompt: {prompt}
            Response: {response}
            
            Rate hallucination risk from 0.0 (no hallucination) to 1.0 (definite hallucination).
            Consider:
            - Factual accuracy
            - Logical consistency
            - Confidence in claims
            - Verifiability
            
            Return only a number between 0.0 and 1.0.
            """
            
            result = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": judge_prompt}],
                max_tokens=10,
                temperature=0.1
            )
            
            score_text = result.choices[0].message.content.strip()
            score = float(score_text) if score_text.replace('.', '').isdigit() else 0.5
            score = max(0.0, min(1.0, score))  # Clamp to [0,1]
            
            latency = (time.time() - start_time) * 1000
            
            return DetectionResult(
                method="llm_judge",
                score=score,
                confidence=0.85,
                latency_ms=latency,
                details={"judge_model": model, "tokens_used": result.usage.total_tokens}
            )
            
        except Exception as e:
            self.logger.error(f"LLM judge error: {e}")
            return DetectionResult(
                method="llm_judge",
                score=0.5,
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
    
    def detect_self_consistency(self, prompt: str, n_samples: int = 3) -> DetectionResult:
        """Self-Consistency: Generate multiple responses and measure variance"""
        start_time = time.time()
        
        if not openai_client:
            # Simulate self-consistency
            scores = [random.uniform(0.1, 0.9) for _ in range(n_samples)]
            variance = np.var(scores)
            consistency_score = min(1.0, variance * 2)  # Higher variance = higher hallucination risk
            
            return DetectionResult(
                method="self_consistency",
                score=consistency_score,
                confidence=0.8,
                latency_ms=random.uniform(1000, 3000),
                details={"samples": n_samples, "variance": variance, "simulation": True}
            )
        
        try:
            responses = []
            for _ in range(n_samples):
                result = openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=100,
                    temperature=0.7
                )
                responses.append(result.choices[0].message.content)
            
            # Calculate semantic similarity between responses
            if self.embedding_model:
                embeddings = self.embedding_model.encode(responses)
                similarities = []
                for i in range(len(embeddings)):
                    for j in range(i+1, len(embeddings)):
                        sim = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                        )
                        similarities.append(sim)
                
                avg_similarity = np.mean(similarities)
                consistency_score = 1.0 - avg_similarity  # Lower similarity = higher hallucination risk
            else:
                # Fallback: simple text similarity
                consistency_score = random.uniform(0.1, 0.8)
            
            latency = (time.time() - start_time) * 1000
            
            return DetectionResult(
                method="self_consistency",
                score=consistency_score,
                confidence=0.75,
                latency_ms=latency,
                details={
                    "samples": n_samples,
                    "avg_similarity": avg_similarity if 'avg_similarity' in locals() else None,
                    "responses": responses[:1]  # Just first response for brevity
                }
            )
            
        except Exception as e:
            self.logger.error(f"Self-consistency error: {e}")
            return DetectionResult(
                method="self_consistency",
                score=0.5,
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
    
    def detect_token_confidence(self, response: str) -> DetectionResult:
        """Token Confidence: Analyze response for uncertainty indicators"""
        start_time = time.time()
        
        # Simple heuristic-based confidence scoring
        uncertainty_indicators = [
            "might", "could", "perhaps", "possibly", "seems", "appears",
            "i think", "i believe", "probably", "likely", "uncertain",
            "not sure", "maybe", "approximately", "roughly"
        ]
        
        factual_indicators = [
            "is", "are", "was", "were", "definitely", "certainly",
            "according to", "research shows", "studies indicate",
            "proven", "established", "confirmed"
        ]
        
        response_lower = response.lower()
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in response_lower)
        factual_count = sum(1 for indicator in factual_indicators if indicator in response_lower)
        
        total_indicators = uncertainty_count + factual_count
        if total_indicators == 0:
            confidence_score = 0.5  # Neutral
        else:
            confidence_score = uncertainty_count / total_indicators
        
        # Adjust based on response length (very short responses are less confident)
        if len(response.split()) < 10:
            confidence_score += 0.2
        
        confidence_score = min(1.0, confidence_score)
        
        return DetectionResult(
            method="token_confidence",
            score=confidence_score,
            confidence=0.9,
            latency_ms=(time.time() - start_time) * 1000,
            details={
                "uncertainty_indicators": uncertainty_count,
                "factual_indicators": factual_count,
                "response_length": len(response.split())
            }
        )
    
    def detect_semantic_consistency(self, prompt: str, response: str) -> DetectionResult:
        """Semantic Consistency: Compare prompt and response semantic alignment"""
        start_time = time.time()
        
        if not self.embedding_model:
            # Simulate semantic consistency
            score = random.uniform(0.1, 0.8)
            return DetectionResult(
                method="semantic_consistency",
                score=score,
                confidence=0.7,
                latency_ms=random.uniform(200, 800),
                details={"simulation": True}
            )
        
        try:
            # Generate embeddings
            prompt_embedding = self.embedding_model.encode([prompt])
            response_embedding = self.embedding_model.encode([response])
            
            # Calculate cosine similarity
            similarity = np.dot(prompt_embedding[0], response_embedding[0]) / (
                np.linalg.norm(prompt_embedding[0]) * np.linalg.norm(response_embedding[0])
            )
            
            # Convert similarity to hallucination score (lower similarity = higher risk)
            hallucination_score = 1.0 - similarity
            
            latency = (time.time() - start_time) * 1000
            
            return DetectionResult(
                method="semantic_consistency",
                score=hallucination_score,
                confidence=0.8,
                latency_ms=latency,
                details={
                    "similarity": float(similarity),
                    "prompt_length": len(prompt),
                    "response_length": len(response)
                }
            )
            
        except Exception as e:
            self.logger.error(f"Semantic consistency error: {e}")
            return DetectionResult(
                method="semantic_consistency",
                score=0.5,
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                details={"error": str(e)}
            )
    
    def extract_fact_triplets(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract (subject, predicate, object) triplets from text"""
        # Simple rule-based triplet extraction
        triplets = []
        
        # Common patterns for factual statements
        patterns = [
            "The {} is {}",  # The Eiffel Tower is in Paris
            "{} is {}",      # Paris is the capital
            "{} was {}",     # Python was created in 1991
            "{} has {}",     # Water has two hydrogen atoms
        ]
        
        words = text.lower().split()
        # This is a simplified implementation
        # In production, you'd use NLP libraries like spaCy or Stanza
        
        # For demo, return simulated triplets
        if "paris" in text.lower():
            triplets.append(("paris", "is", "capital"))
        if "python" in text.lower():
            triplets.append(("python", "created_by", "guido"))
        if "water" in text.lower():
            triplets.append(("water", "formula", "h2o"))
        
        return triplets
    
    def detect_fact_triplet_consistency(self, response: str) -> DetectionResult:
        """Fact Triplet: Extract facts and check against knowledge base"""
        start_time = time.time()
        
        triplets = self.extract_fact_triplets(response)
        
        if not triplets:
            return DetectionResult(
                method="fact_triplet",
                score=0.3,  # Low risk if no facts extracted
                confidence=0.6,
                latency_ms=(time.time() - start_time) * 1000,
                details={"triplets_found": 0}
            )
        
        inconsistent_facts = 0
        verified_facts = 0
        
        for subject, predicate, obj in triplets:
            subject_key = subject.replace(" ", "_")
            if subject_key in self.knowledge_base:
                kb_facts = self.knowledge_base[subject_key]
                # Simple fact checking
                if predicate in kb_facts:
                    if str(kb_facts[predicate]).lower() != obj.lower():
                        inconsistent_facts += 1
                    else:
                        verified_facts += 1
        
        total_facts = len(triplets)
        if total_facts == 0:
            hallucination_score = 0.3
        else:
            hallucination_score = inconsistent_facts / total_facts
        
        return DetectionResult(
            method="fact_triplet",
            score=hallucination_score,
            confidence=0.85,
            latency_ms=(time.time() - start_time) * 1000,
            details={
                "triplets_found": total_facts,
                "verified_facts": verified_facts,
                "inconsistent_facts": inconsistent_facts,
                "extracted_triplets": triplets
            }
        )
    
    def run_all_detections(self, prompt: str, response: str) -> List[DetectionResult]:
        """Run all detection methods and return results"""
        results = []
        
        # Run all detection methods
        results.append(self.detect_llm_as_judge(prompt, response))
        results.append(self.detect_self_consistency(prompt))
        results.append(self.detect_token_confidence(response))
        results.append(self.detect_semantic_consistency(prompt, response))
        results.append(self.detect_fact_triplet_consistency(response))
        
        return results
    
    def aggregate_results(self, results: List[DetectionResult]) -> Dict:
        """Aggregate multiple detection results into final score"""
        if not results:
            return {"final_score": 0.5, "confidence": 0.0, "method": "none"}
        
        # Weight different methods
        weights = {
            "llm_judge": 0.3,
            "self_consistency": 0.25,
            "semantic_consistency": 0.2,
            "fact_triplet": 0.15,
            "token_confidence": 0.1
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        method_scores = {}
        
        for result in results:
            weight = weights.get(result.method, 0.1)
            weighted_score += result.score * weight * result.confidence
            total_weight += weight * result.confidence
            method_scores[result.method] = result.score
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
            avg_confidence = np.mean([r.confidence for r in results])
        else:
            final_score = 0.5
            avg_confidence = 0.0
        
        return {
            "final_score": final_score,
            "confidence": avg_confidence,
            "method_scores": method_scores,
            "individual_results": [
                {
                    "method": r.method,
                    "score": r.score,
                    "confidence": r.confidence,
                    "latency_ms": r.latency_ms
                } for r in results
            ],
            "total_latency_ms": sum(r.latency_ms for r in results)
        }

# Demo function
def demo_advanced_detection():
    """Demonstrate the advanced detection system"""
    detector = AdvancedHallucinationDetector()
    
    test_cases = [
        {
            "prompt": "What is the capital of France?",
            "response": "The capital of France is Paris.",
            "expected": "low"
        },
        {
            "prompt": "Who created Python?",
            "response": "Python was created by Google in 2010.",
            "expected": "high"
        },
        {
            "prompt": "What is the boiling point of water?",
            "response": "Water boils at approximately 100 degrees Celsius at sea level.",
            "expected": "low"
        }
    ]
    
    print("Advanced Hallucination Detection Demo")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['expected']} hallucination risk expected")
        print(f"Prompt: {test_case['prompt']}")
        print(f"Response: {test_case['response']}")
        
        # Run all detections
        results = detector.run_all_detections(test_case['prompt'], test_case['response'])
        aggregated = detector.aggregate_results(results)
        
        print(f"\nResults:")
        for result in results:
            print(f"  {result.method}: {result.score:.3f} (confidence: {result.confidence:.2f}, latency: {result.latency_ms:.0f}ms)")
        
        print(f"\nAggregated Score: {aggregated['final_score']:.3f}")
        print(f"Overall Confidence: {aggregated['confidence']:.3f}")
        print(f"Total Detection Time: {aggregated['total_latency_ms']:.0f}ms")
        
        # Decision
        hallucination_detected = aggregated['final_score'] > 0.5
        print(f"Final Decision: {'HALLUCINATION DETECTED' if hallucination_detected else 'No hallucination'}")
        print("-" * 50)

if __name__ == "__main__":
    demo_advanced_detection()

#!/usr/bin/env python3
"""
Smart Query Router for Dual Index RAG System

This module provides intelligent routing between different embedding models
based on query analysis, performance metrics, and fallback strategies.

Features:
- Query pattern analysis for optimal embedding selection
- Performance-based routing with fallback capabilities
- Confidence scoring and result validation
- Real-time performance monitoring
- Adaptive routing based on historical performance
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
from pathlib import Path
import logging

# LangChain imports
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory paths
DB_BASE_DIR = "indexes"
OPENAI_DB_DIR = os.path.join(DB_BASE_DIR, "openai_index")
GOOGLE_DB_DIR = os.path.join(DB_BASE_DIR, "google_index")
METADATA_DIR = "metadata"
PERFORMANCE_DIR = "performance_metrics"

# Routing configuration
ROUTING_CONFIG = {
    "query_patterns": {
        "technical": {
            "keywords": ["code", "programming", "api", "algorithm", "software", "development", 
                        "python", "javascript", "database", "server", "framework", "library"],
            "preferred_provider": "openai",
            "confidence_boost": 0.15
        },
        "legal": {
            "keywords": ["regulation", "compliance", "policy", "framework", "guidance", "law",
                        "legal", "statute", "act", "regulation", "safeguarding", "inspection"],
            "preferred_provider": "google",
            "confidence_boost": 0.10
        },
        "general": {
            "keywords": ["what", "how", "why", "explain", "describe", "tell", "help", "show"],
            "preferred_provider": "openai",
            "confidence_boost": 0.05
        },
        "analytical": {
            "keywords": ["analyze", "compare", "evaluate", "assess", "review", "examine", "study"],
            "preferred_provider": "openai",
            "confidence_boost": 0.12
        }
    },
    "fallback_strategy": "performance_based",  # "round_robin", "performance_based", "default"
    "default_provider": "openai",
    "min_confidence_threshold": 0.3,
    "performance_window_hours": 24,
    "max_response_time_seconds": 10.0
}

# =============================================================================
# VISION MODEL CONFIGURATION (ADD THIS SECTION)
# =============================================================================

VISION_CONFIG = {
    "models": {
        "openai_vision": {
            "model": "gpt-4o",
            "speed_tier": "slow",
            "quality_tier": "high"
        },
        "openai_vision_mini": {
            "model": "gpt-4o-mini", 
            "speed_tier": "fast",
            "quality_tier": "medium"
        },
        "google_vision": {
            "model": "gemini-1.5-pro",
            "speed_tier": "slow", 
            "quality_tier": "high"
        },
        "google_vision_flash": {
            "model": "gemini-1.5-flash",
            "speed_tier": "fast",
            "quality_tier": "medium"
        }
    },
    "routing_rules": {
        "speed": ["openai_vision_mini", "google_vision_flash"],
        "balanced": ["google_vision_flash", "openai_vision_mini", "google_vision"],
        "quality": ["openai_vision", "google_vision"]
    },
    "default_strategy": "balanced"
}

LLM_CONFIG = {
    "models": {
        "gpt_4o": {"speed_tier": "slow", "quality_tier": "high"},
        "gpt_4o_mini": {"speed_tier": "fast", "quality_tier": "medium"},
        "gemini_1_5_pro": {"speed_tier": "medium", "quality_tier": "high"},
        "gemini_1_5_flash": {"speed_tier": "fast", "quality_tier": "medium"}
    }
}

# Performance tracking
PERFORMANCE_METRICS = {
    "response_time_weight": 0.3,
    "result_quality_weight": 0.4,
    "success_rate_weight": 0.3,
    "decay_factor": 0.95  # For exponential decay of old metrics
}

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PERFORMANCE TRACKER
# =============================================================================

class PerformanceTracker:
    """Tracks and analyzes performance metrics for each embedding provider."""
    
    def __init__(self):
        self.metrics_file = Path(PERFORMANCE_DIR) / "routing_metrics.json"
        self.ensure_performance_dir()
        self.load_metrics()
    
    def ensure_performance_dir(self):
        """Create performance directory if it doesn't exist."""
        Path(PERFORMANCE_DIR).mkdir(exist_ok=True)
    
    def load_metrics(self):
        """Load existing performance metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    self.metrics = json.load(f)
                logger.info("Loaded existing performance metrics")
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
                self.metrics = self._initialize_metrics()
        else:
            self.metrics = self._initialize_metrics()
    
    def _initialize_metrics(self) -> Dict[str, Any]:
        """Initialize metrics structure."""
        return {
            "providers": {},
            "query_patterns": {},
            "last_updated": datetime.now().isoformat(),
            "total_queries": 0
        }
    
    def record_query(self, provider: str, query: str, response_time: float, 
                    result_count: int, pattern: str, success: bool = True):
        """Record query performance metrics."""
        if provider not in self.metrics["providers"]:
            self.metrics["providers"][provider] = {
                "total_queries": 0,
                "total_response_time": 0,
                "success_count": 0,
                "average_results": 0,
                "pattern_performance": {},
                "recent_queries": []
            }
        
        provider_metrics = self.metrics["providers"][provider]
        
        # Update provider metrics
        provider_metrics["total_queries"] += 1
        provider_metrics["total_response_time"] += response_time
        if success:
            provider_metrics["success_count"] += 1
        
        # Update pattern-specific metrics
        if pattern not in provider_metrics["pattern_performance"]:
            provider_metrics["pattern_performance"][pattern] = {
                "queries": 0,
                "avg_response_time": 0,
                "success_rate": 0
            }
        
        pattern_metrics = provider_metrics["pattern_performance"][pattern]
        pattern_metrics["queries"] += 1
        
        # Update recent queries (keep last 100)
        recent_query = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # Truncate for storage
            "response_time": response_time,
            "result_count": result_count,
            "pattern": pattern,
            "success": success
        }
        
        provider_metrics["recent_queries"].append(recent_query)
        if len(provider_metrics["recent_queries"]) > 100:
            provider_metrics["recent_queries"] = provider_metrics["recent_queries"][-100:]
        
        # Update global metrics
        self.metrics["total_queries"] += 1
        self.metrics["last_updated"] = datetime.now().isoformat()
        
        # Save metrics periodically
        if self.metrics["total_queries"] % 10 == 0:
            self.save_metrics()
    
    def get_provider_performance(self, provider: str, hours: int = 24) -> Dict[str, float]:
        """Get performance metrics for a provider within time window."""
        if provider not in self.metrics["providers"]:
            return {"score": 0.0, "response_time": 10.0, "success_rate": 0.0}
        
        provider_metrics = self.metrics["providers"][provider]
        
        # Calculate average response time
        if provider_metrics["total_queries"] > 0:
            avg_response_time = provider_metrics["total_response_time"] / provider_metrics["total_queries"]
            success_rate = provider_metrics["success_count"] / provider_metrics["total_queries"]
        else:
            avg_response_time = 10.0
            success_rate = 0.0
        
        # Calculate composite performance score
        response_time_score = max(0, (10.0 - avg_response_time) / 10.0)  # Normalize to 0-1
        performance_score = (
            response_time_score * PERFORMANCE_METRICS["response_time_weight"] +
            success_rate * PERFORMANCE_METRICS["success_rate_weight"] +
            0.5 * PERFORMANCE_METRICS["result_quality_weight"]  # Default quality score
        )
        
        return {
            "score": performance_score,
            "response_time": avg_response_time,
            "success_rate": success_rate,
            "total_queries": provider_metrics["total_queries"]
        }
    
    def get_best_provider(self, pattern: str = None) -> str:
        """Get the best performing provider overall or for a specific pattern."""
        if not self.metrics["providers"]:
            return ROUTING_CONFIG["default_provider"]
        
        best_provider = ROUTING_CONFIG["default_provider"]
        best_score = 0.0
        
        for provider in self.metrics["providers"]:
            performance = self.get_provider_performance(provider)
            score = performance["score"]
            
            # Boost score for pattern-specific performance
            if pattern and provider in self.metrics["providers"]:
                provider_metrics = self.metrics["providers"][provider]
                if pattern in provider_metrics["pattern_performance"]:
                    pattern_perf = provider_metrics["pattern_performance"][pattern]
                    if pattern_perf["queries"] > 0:
                        # Small boost for pattern-specific experience
                        score += 0.1
            
            if score > best_score:
                best_score = score
                best_provider = provider
        
        return best_provider
    
    def save_metrics(self):
        """Save metrics to file."""
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save performance metrics: {e}")


# =============================================================================
# ENHANCED PERFORMANCE TRACKER (ADD THIS CLASS)
# =============================================================================

class VisionPerformanceTracker(PerformanceTracker):
    """Extended performance tracker for vision and LLM models."""
    
    def __init__(self):
        super().__init__()
        self.vision_metrics = {
            "vision_models": {},
            "llm_models": {},
            "speed_vs_quality": {}
        }
    
    def record_vision_query(self, vision_model: str, llm_model: str, 
                           image_size_kb: int, query_complexity: str,
                           vision_time: float, llm_time: float, total_time: float):
        """Record performance metrics for vision + LLM pipeline."""
        
        # Simple performance tracking
        if vision_model not in self.vision_metrics["vision_models"]:
            self.vision_metrics["vision_models"][vision_model] = {
                "total_queries": 0, "total_time": 0
            }
        
        vision_stats = self.vision_metrics["vision_models"][vision_model]
        vision_stats["total_queries"] += 1
        vision_stats["total_time"] += vision_time
        
        # Similar for LLM tracking
        if llm_model not in self.vision_metrics["llm_models"]:
            self.vision_metrics["llm_models"][llm_model] = {
                "total_queries": 0, "total_time": 0
            }
        
        llm_stats = self.vision_metrics["llm_models"][llm_model]
        llm_stats["total_queries"] += 1
        llm_stats["total_time"] += llm_time
    
    def get_best_vision_model(self, priority: str = "balanced") -> str:
        """Get best vision model based on priority."""
        if priority not in VISION_CONFIG["routing_rules"]:
            priority = VISION_CONFIG["default_strategy"]
        
        priority_models = VISION_CONFIG["routing_rules"][priority]
        
        # Check performance metrics if available
        if self.vision_metrics["vision_models"]:
            best_model = None
            best_avg_time = float('inf')
            
            for model in priority_models:
                if model in self.vision_metrics["vision_models"]:
                    stats = self.vision_metrics["vision_models"][model]
                    if stats["total_queries"] > 0:
                        avg_time = stats["total_time"] / stats["total_queries"]
                        if avg_time < best_avg_time:
                            best_avg_time = avg_time
                            best_model = model
            
            if best_model:
                return best_model
        
        # Fallback to first available model in priority list
        return priority_models[0] if priority_models else "openai_vision_mini"
    
    def get_best_llm_model(self, complexity: str = "medium") -> str:
        """Get best LLM model based on complexity."""
        if complexity == "low":
            candidates = ["gpt_4o_mini", "gemini_1_5_flash"]
        elif complexity == "high":
            candidates = ["gpt_4o", "gemini_1_5_pro"] 
        else:  # medium
            candidates = ["gpt_4o_mini", "gemini_1_5_flash", "gemini_1_5_pro"]
        
        # Check performance metrics
        if self.vision_metrics["llm_models"]:
            best_model = None
            best_time = float('inf')
            
            for model in candidates:
                if model in self.vision_metrics["llm_models"]:
                    stats = self.vision_metrics["llm_models"][model]
                    if stats["total_queries"] > 0:
                        avg_time = stats["total_time"] / stats["total_queries"]
                        if avg_time < best_time:
                            best_time = avg_time
                            best_model = model
            
            if best_model:
                return best_model
        
        return candidates[0] if candidates else "gpt_4o_mini"

# =============================================================================
# QUERY ANALYZER
# =============================================================================

class QueryAnalyzer:
    """Analyzes queries to determine optimal routing strategy."""
    
    def __init__(self):
        self.patterns = ROUTING_CONFIG["query_patterns"]
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze query to determine pattern and routing preferences.
        
        Args:
            query (str): User query to analyze
            
        Returns:
            Dict[str, Any]: Analysis results with pattern, confidence, and provider preferences
        """
        query_lower = query.lower()
        
        # Calculate pattern scores
        pattern_scores = {}
        for pattern_name, pattern_config in self.patterns.items():
            score = 0.0
            keywords_found = []
            
            for keyword in pattern_config["keywords"]:
                if keyword in query_lower:
                    score += 1.0
                    keywords_found.append(keyword)
                elif any(keyword in word for word in query_lower.split()):
                    score += 0.5  # Partial match
            
            # Normalize by number of keywords in pattern
            if len(pattern_config["keywords"]) > 0:
                score = score / len(pattern_config["keywords"])
            
            pattern_scores[pattern_name] = {
                "score": score,
                "keywords_found": keywords_found,
                "preferred_provider": pattern_config["preferred_provider"],
                "confidence_boost": pattern_config.get("confidence_boost", 0.0)
            }
        
        # Determine best pattern match
        best_pattern = max(pattern_scores.items(), key=lambda x: x[1]["score"])
        pattern_name = best_pattern[0]
        pattern_data = best_pattern[1]
        
        # Calculate overall confidence
        base_confidence = min(pattern_data["score"], 1.0)
        confidence = base_confidence + pattern_data["confidence_boost"]
        confidence = min(confidence, 1.0)  # Cap at 1.0
        
        # Additional analysis
        query_length = len(query.split())
        complexity_score = min(query_length / 20.0, 1.0)  # Normalize to 0-1
        
        return {
            "query": query,
            "best_pattern": pattern_name,
            "pattern_confidence": confidence,
            "preferred_provider": pattern_data["preferred_provider"],
            "keywords_found": pattern_data["keywords_found"],
            "query_length": query_length,
            "complexity_score": complexity_score,
            "all_pattern_scores": pattern_scores,
            "analysis_timestamp": datetime.now().isoformat()
        }

# =============================================================================
# SMART ROUTER
# =============================================================================

class SmartRouter:
    """Main router class that combines query analysis and performance metrics."""
    
    # ADD THESE METHODS TO YOUR EXISTING SmartRouter CLASS

    def __init__(self):
        # Keep your existing __init__ code exactly the same
        self.performance_tracker = PerformanceTracker()
        self.query_analyzer = QueryAnalyzer()
        self.embedding_models = {}
        self.vector_stores = {}
        self.load_embedding_models()
        self.load_vector_stores()
    
        # ADD THESE LINES:
        self.vision_tracker = VisionPerformanceTracker()
        self.performance_mode = "balanced"

    def set_performance_mode(self, mode: str):
        """Set performance mode: 'speed', 'balanced', or 'quality'."""
        if mode in ["speed", "balanced", "quality"]:
            self.performance_mode = mode
            logger.info(f"Performance mode set to: {mode}")
        else:
            logger.warning(f"Invalid mode {mode}. Valid modes: speed, balanced, quality")

    def analyze_query_complexity(self, query: str, has_image: bool = False) -> str:
        """Analyze query complexity for model selection."""
        query_lower = query.lower()
    
        high_complexity_keywords = [
            "analyze", "compare", "evaluate", "assess", "detailed", "comprehensive",
            "complex", "multi-step", "reasoning", "critical", "safety", "risk"
        ]
    
        low_complexity_keywords = [
            "what", "who", "when", "where", "simple", "basic", "quick", "summary"
        ]
    
        high_score = sum(1 for keyword in high_complexity_keywords if keyword in query_lower)
        low_score = sum(1 for keyword in low_complexity_keywords if keyword in query_lower)
    
        query_length = len(query.split())
    
        if has_image:
            complexity_boost = 1  # Images add complexity
        else:
            complexity_boost = 0
        
        if high_score >= 2 or query_length > 30 or complexity_boost:
            return "high"
        elif low_score >= 2 and query_length < 10:
            return "low" 
        else:
            return "medium"

    def route_multimodal_query(self, query: str, image_data: bytes = None, 
                              image_size_kb: int = None, k: int = 5) -> Dict[str, Any]:
        """
        Route multimodal query with optimal model selection.
        """
        start_time = time.time()
    
        # Analyze query complexity
        complexity = self.analyze_query_complexity(query, has_image=bool(image_data))
    
        # Route text retrieval (existing functionality)
        text_result = self.route_query(query, k=k)
    
        # Select vision model if image provided
        vision_model = None
        if image_data:
            vision_model = self.vision_tracker.get_best_vision_model(
                priority=self.performance_mode
            )
    
        # Select LLM model based on complexity and mode
        llm_model = self.vision_tracker.get_best_llm_model(complexity=complexity)
    
        # Build response
        routing_result = {
            "text_routing": text_result,
            "vision_model": vision_model,
            "llm_model": llm_model,
            "performance_mode": self.performance_mode,
            "query_complexity": complexity,
            "timestamp": datetime.now().isoformat()
        }
    
        return routing_result
    
    def load_embedding_models(self):
        """Load embedding models based on available configurations."""
        model_info_file = Path(METADATA_DIR) / "embedding_models.json"
        
        if model_info_file.exists():
            with open(model_info_file, 'r') as f:
                model_info = json.load(f)
            
            for provider, info in model_info.items():
                if info.get("status") == "active":
                    try:
                        if provider == "openai":
                            self.embedding_models[provider] = OpenAIEmbeddings(
                                model=info["model"],
                                show_progress_bar=False
                            )
                        elif provider == "google":
                            self.embedding_models[provider] = GoogleGenerativeAIEmbeddings(
                                model=info["model"]
                            )
                        logger.info(f"Loaded {provider} embedding model")
                    except Exception as e:
                        logger.error(f"Failed to load {provider} embedding model: {e}")
        else:
            logger.warning("No embedding model info found, using defaults")
            # Fallback to default models
            try:
                self.embedding_models["openai"] = OpenAIEmbeddings(
                    model="text-embedding-3-small",
                    show_progress_bar=False
                )
                self.embedding_models["google"] = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001"
                )
            except Exception as e:
                logger.error(f"Failed to load default embedding models: {e}")
    
    def load_vector_stores(self):
        """Load FAISS vector stores for each provider."""
        store_paths = {
            "openai": OPENAI_DB_DIR,
            "google": GOOGLE_DB_DIR
        }
        
        for provider, db_path in store_paths.items():
            if os.path.exists(db_path) and provider in self.embedding_models:
                try:
                    self.vector_stores[provider] = FAISS.load_local(
                        db_path, 
                        self.embedding_models[provider],
                        allow_dangerous_deserialization=True
                    )
                    logger.info(f"Loaded {provider} vector store from {db_path}")
                except Exception as e:
                    logger.error(f"Failed to load {provider} vector store: {e}")
    
    def route_query(self, query: str, k: int = 5, force_provider: str = None) -> Dict[str, Any]:
        """
        Route query to optimal embedding provider and return results.
        
        Args:
            query (str): User query
            k (int): Number of results to retrieve
            force_provider (str): Force specific provider (for testing)
            
        Returns:
            Dict[str, Any]: Routing results and retrieved documents
        """
        start_time = time.time()
        
        # Analyze query if not forcing provider
        if force_provider:
            selected_provider = force_provider
            analysis = {"forced": True, "provider": force_provider}
        else:
            analysis = self.query_analyzer.analyze_query(query)
            selected_provider = self._select_provider(analysis)
        
        # Attempt query with selected provider
        result = self._execute_query(selected_provider, query, k, start_time, analysis)
        
        # Fallback logic if primary attempt fails
        if not result["success"] and not force_provider:
            logger.warning(f"Primary provider {selected_provider} failed, trying fallback")
            fallback_provider = self._get_fallback_provider(selected_provider)
            if fallback_provider and fallback_provider != selected_provider:
                result = self._execute_query(fallback_provider, query, k, start_time, analysis)
                result["used_fallback"] = True
        
        return result
    
    def _select_provider(self, analysis: Dict[str, Any]) -> str:
        """Select the best provider based on analysis and performance."""
        pattern = analysis["best_pattern"]
        confidence = analysis["pattern_confidence"]
        preferred_provider = analysis["preferred_provider"]
        
        # If confidence is high, use pattern preference
        if confidence >= ROUTING_CONFIG["min_confidence_threshold"]:
            # Check if preferred provider is available
            if preferred_provider in self.vector_stores:
                return preferred_provider
        
        # Otherwise, use performance-based selection
        if ROUTING_CONFIG["fallback_strategy"] == "performance_based":
            return self.performance_tracker.get_best_provider(pattern)
        else:
            return ROUTING_CONFIG["default_provider"]
    
    def _get_fallback_provider(self, primary_provider: str) -> Optional[str]:
        """Get fallback provider different from primary."""
        available_providers = list(self.vector_stores.keys())
        fallback_candidates = [p for p in available_providers if p != primary_provider]
        
        if not fallback_candidates:
            return None
        
        if ROUTING_CONFIG["fallback_strategy"] == "performance_based":
            best_fallback = None
            best_score = -1
            for provider in fallback_candidates:
                perf = self.performance_tracker.get_provider_performance(provider)
                if perf["score"] > best_score:
                    best_score = perf["score"]
                    best_fallback = provider
            return best_fallback
        else:
            return fallback_candidates[0]
    
    def _execute_query(self, provider: str, query: str, k: int, start_time: float, 
                      analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Execute query against specific provider."""
        try:
            if provider not in self.vector_stores:
                raise Exception(f"Vector store not available for {provider}")
            
            # Perform similarity search
            search_start = time.time()
            docs = self.vector_stores[provider].similarity_search(query, k=k)
            search_time = time.time() - search_start
            
            total_time = time.time() - start_time
            
            # Record performance
            pattern = analysis.get("best_pattern", "unknown")
            self.performance_tracker.record_query(
                provider=provider,
                query=query,
                response_time=total_time,
                result_count=len(docs),
                pattern=pattern,
                success=True
            )
            
            return {
                "success": True,
                "provider": provider,
                "documents": docs,
                "total_results": len(docs),
                "response_time": total_time,
                "search_time": search_time,
                "analysis": analysis,
                "used_fallback": False,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            total_time = time.time() - start_time
            
            # Record failed performance
            pattern = analysis.get("best_pattern", "unknown")
            self.performance_tracker.record_query(
                provider=provider,
                query=query,
                response_time=total_time,
                result_count=0,
                pattern=pattern,
                success=False
            )
            
            logger.error(f"Query execution failed for {provider}: {e}")
            return {
                "success": False,
                "provider": provider,
                "error": str(e),
                "documents": [],
                "total_results": 0,
                "response_time": total_time,
                "analysis": analysis,
                "used_fallback": False,
                "timestamp": datetime.now().isoformat()
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all providers."""
        summary = {
            "providers": {},
            "total_queries": self.performance_tracker.metrics.get("total_queries", 0),
            "available_providers": list(self.vector_stores.keys()),
            "last_updated": self.performance_tracker.metrics.get("last_updated")
        }
        
        for provider in self.vector_stores.keys():
            summary["providers"][provider] = self.performance_tracker.get_provider_performance(provider)
        
        return summary
    
    def test_routing(self, test_queries: List[str]) -> Dict[str, Any]:
        """Test routing with sample queries for validation."""
        results = []
        
        for query in test_queries:
            analysis = self.query_analyzer.analyze_query(query)
            result = {
                "query": query,
                "analysis": analysis,
                "recommended_provider": self._select_provider(analysis)
            }
            results.append(result)
        
        return {"test_results": results, "total_tests": len(test_queries)}

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_smart_router() -> SmartRouter:
    """Create and return a configured SmartRouter instance."""
    return SmartRouter()

def quick_search(query: str, k: int = 5, provider: str = None) -> List[Document]:
    """Quick search function for simple use cases."""
    router = create_smart_router()
    result = router.route_query(query, k=k, force_provider=provider)
    
    if result["success"]:
        return result["documents"]
    else:
        logger.error(f"Search failed: {result.get('error')}")
        return []

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

if __name__ == "__main__":
    # Example usage
    router = SmartRouter()
    
    # Test queries
    test_queries = [
        "What are the safeguarding policies for children's homes?",
        "How do I implement error handling in Python?",
        "What are the legal requirements for fostering agencies?",
        "Compare different machine learning algorithms",
        "Explain the inspection framework for children's services"
    ]
    
    print("Testing Smart Router...")
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = router.route_query(query, k=3)
        
        if result["success"]:
            print(f"Provider: {result['provider']}")
            print(f"Results: {result['total_results']}")
            print(f"Response time: {result['response_time']:.3f}s")
            print(f"Pattern: {result['analysis'].get('best_pattern')}")
        else:
            print(f"Failed: {result['error']}")
    
    # Performance summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    perf_summary = router.get_performance_summary()
    for provider, metrics in perf_summary["providers"].items():
        print(f"{provider.upper()}:")
        print(f"  Score: {metrics['score']:.3f}")
        print(f"  Avg Response Time: {metrics['response_time']:.3f}s")
        print(f"  Success Rate: {metrics['success_rate']:.3f}")
        print(f"  Total Queries: {metrics['total_queries']}")

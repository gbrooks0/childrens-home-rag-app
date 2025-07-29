# rag_system.py (Enhanced Enterprise-Level Version)

import os
import tempfile
import base64
import sys
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from contextlib import contextmanager
from functools import wraps, lru_cache
import hashlib

# --- IMPORTANT: SQLite3 Fix for ChromaDB in Deployment Environments ---
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    logging.info("pysqlite3 imported and set as default sqlite3 module.")
except ImportError:
    logging.info("pysqlite3 not found, falling back to system sqlite3.")

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import google.generativeai as genai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class QuestionCategory(Enum):
    """Enumeration for question classification types."""
    REGULATORY_FACTUAL = "Regulatory_Factual"
    STRATEGIC_ANALYTICAL = "Strategic_Analytical"
    GENERAL_INQUIRY = "General_Inquiry"

class LLMProvider(Enum):
    """Enumeration for LLM providers."""
    GEMINI = "Gemini"
    OPENAI = "ChatGPT"

@dataclass
class QueryMetrics:
    """Data class for tracking query performance metrics."""
    question_category: str
    primary_llm_used: str
    fallback_used: bool
    response_time_ms: float
    context_chunks: int
    response_length: int
    timestamp: float

@dataclass
class RAGConfig:
    """Configuration class for RAG system settings."""
    faiss_index_path: str = "faiss_index"
    main_retriever_k: int = 12
    session_retriever_k: int = 3
    ensemble_weights: List[float] = None
    chunk_size: int = 1000
    chunk_overlap: int = 100
    min_substantive_length: int = 100
    gemini_model: str = "gemini-1.5-pro-latest"
    gemini_temperature: float = 0.1
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.7
    embedding_model_gemini: str = "models/text-embedding-004"
    embedding_model_openai: str = "text-embedding-ada-002"
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = [0.7, 0.3]

class RAGSystemError(Exception):
    """Base exception class for RAG system errors."""
    pass

class ModelInitializationError(RAGSystemError):
    """Exception raised when model initialization fails."""
    pass

class IndexLoadError(RAGSystemError):
    """Exception raised when FAISS index loading fails."""
    pass

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retrying failed operations."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator

@contextmanager
def performance_timer():
    """Context manager for measuring execution time."""
    start_time = time.perf_counter()
    try:
        yield lambda: (time.perf_counter() - start_time) * 1000
    finally:
        pass

class ConversationMemory:
    """Simple conversation memory for better follow-up handling."""
    
    def __init__(self, max_history: int = 5):
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_exchange(self, question: str, answer: str, category: str):
        """Add a question-answer exchange to memory."""
        self.conversation_history.append({
            "question": question,
            "answer": answer,
            "category": category,
            "timestamp": time.time()
        })
        
        # Keep only recent exchanges
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
    
    def get_context_for_followup(self) -> str:
        """Get conversation context for follow-up questions."""
        if not self.conversation_history:
            return ""
        
        context_parts = []
        for exchange in self.conversation_history[-3:]:  # Last 3 exchanges
            context_parts.append(f"Previous Q: {exchange['question']}")
            context_parts.append(f"Previous A: {exchange['answer'][:200]}...")
        
        return "\n".join(context_parts)

class EnhancedRAGSystem:
    """
    Enhanced RAG System with improved error handling, performance monitoring,
    caching, and enterprise-level features.
    """
    
    # Advanced prompt templates
ADVANCED_PROMPT_TEMPLATE = """---
**ROLE AND GOAL**
You are a highly knowledgeable assistant with specialized expertise in children's homes operations, strategic planning, Ofsted regulations, and care sector best practices. While your primary specialization is in children's homes, you are also capable of providing helpful, accurate responses on a wide variety of topics using the available context.

---
**CONTEXT**
{context}

---
**OUTPUT RULES**
1.  **Leverage Your Expertise:** When questions relate to children's homes, care regulations, or related topics, draw upon your specialized knowledge to provide comprehensive, expert-level guidance that goes beyond basic information.

2.  **Comprehensive Context Utilization:** Extract and synthesize all relevant information from the provided context. Connect different pieces of information and explain relationships between concepts, standards, or procedures.

3.  **Actionable and Strategic Insights:** Transform information into practical, actionable recommendations. For regulatory content, explain compliance strategies. For operational topics, suggest implementation approaches. For strategic questions, provide analytical insights.

4.  **Anticipate Follow-up Needs:** Structure your response to naturally invite follow-up questions. When appropriate, mention related topics or areas that might warrant further exploration, and suggest specific follow-up questions the user might want to ask.

5.  **Flexible Context Handling:** 
   - If the context fully addresses the question, provide a comprehensive answer
   - If the context partially addresses the question, provide what you can and clearly indicate what additional information would be helpful
   - If the context doesn't contain relevant information, acknowledge this and suggest how the question might be refined or what type of context would be needed

6.  **Professional and Accessible:** Maintain a professional tone while ensuring accessibility. Use clear explanations for complex concepts and provide context for technical terms.

7.  **Clear Formatting:** Use Markdown for optimal readability:
   - **Bold** for key points and emphasis
   - Bullet points for lists and action items
   - Numbered lists for sequential steps or ranked priorities
   - Tables for comparisons when appropriate
   - Headers for organizing complex responses

---
**QUESTION**
{question}

---
**YOUR EXPERT RESPONSE:**"""

    CLASSIFICATION_PROMPT = """
Analyze the following user question and classify it into one of these three categories:

- 'Regulatory_Factual': Questions asking for specific standards, regulations, definitions, compliance requirements, or direct factual information from official documents
- 'Strategic_Analytical': Questions asking for advice, analysis, strategic guidance, comparisons, implementation strategies, or broader operational insights  
- 'General_Inquiry': Questions about general topics not specifically related to regulations or strategy, follow-up questions seeking clarification, or conversational questions

Respond ONLY with the category keyword. Do not include any other text.

Question: {question}
Category:
"""

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the enhanced RAG system with comprehensive error handling and monitoring."""
        self.config = config or RAGConfig()
        self.metrics_history: List[QueryMetrics] = []
        self._classification_cache: Dict[str, str] = {}
        self.conversation_memory = ConversationMemory()
        
        logger.info("Initializing Enhanced RAG System...")
        
        # Initialize models with proper error handling
        self._initialize_models()
        
        # Load FAISS index with validation
        self._load_faiss_index()
        
        # Initialize session state
        self.session_retriever = None
        self._session_file_hash = None
        
        logger.info("Enhanced RAG System initialization complete.")

    def _initialize_models(self) -> None:
        """Initialize LLM models with comprehensive error handling."""
        logger.info("Initializing AI models...")
        
        # Configure Google API
        gemini_api_key = os.environ.get("GOOGLE_API_KEY")
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            logger.info("Google GenerativeAI configured successfully.")
        else:
            logger.warning("GOOGLE_API_KEY not found. Gemini models will be unavailable.")

        # Initialize Gemini models
        self.gemini_llm = None
        self.gemini_embeddings = None
        
        if gemini_api_key:
            try:
                self.gemini_llm = ChatGoogleGenerativeAI(
                    model=self.config.gemini_model,
                    temperature=self.config.gemini_temperature,
                    google_api_key=gemini_api_key
                )
                self.gemini_embeddings = GoogleGenerativeAIEmbeddings(
                    model=self.config.embedding_model_gemini,
                    google_api_key=gemini_api_key
                )
                logger.info("Gemini models initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini models: {e}")
                raise ModelInitializationError(f"Gemini initialization failed: {e}")

        # Initialize OpenAI models
        self.openai_llm = None
        self.openai_embeddings = None
        
        openai_api_key = os.environ.get("OPENAI_API_KEY")
        if openai_api_key:
            try:
                self.openai_llm = ChatOpenAI(
                    model=self.config.openai_model,
                    temperature=self.config.openai_temperature,
                    openai_api_key=openai_api_key
                )
                self.openai_embeddings = OpenAIEmbeddings(
                    model=self.config.embedding_model_openai,
                    openai_api_key=openai_api_key
                )
                logger.info("OpenAI models initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI models: {e}")
                if not self.gemini_llm:  # Only raise if no fallback available
                    raise ModelInitializationError(f"OpenAI initialization failed and no Gemini fallback: {e}")
        else:
            logger.warning("OPENAI_API_KEY not found. OpenAI models will be unavailable.")

        # Determine primary embeddings
        if self.gemini_embeddings:
            self.embeddings = self.gemini_embeddings
            logger.info("Using Gemini embeddings as primary.")
        elif self.openai_embeddings:
            self.embeddings = self.openai_embeddings
            logger.info("Using OpenAI embeddings as primary.")
        else:
            raise ModelInitializationError("No embedding models could be initialized.")

        # Validate at least one LLM is available
        if not self.gemini_llm and not self.openai_llm:
            raise ModelInitializationError("No LLM models could be initialized.")

    @retry_on_failure(max_retries=3, delay=1.0)
    def _load_faiss_index(self) -> None:
        """Load FAISS index with retry logic and validation."""
        db_path = Path(self.config.faiss_index_path)
        
        if not db_path.exists():
            error_msg = f"FAISS index not found at '{db_path}'. Please run ingest.py to create it."
            logger.error(error_msg)
            raise IndexLoadError(error_msg)
        
        logger.info(f"Loading FAISS index from '{db_path}'...")
        
        try:
            db = FAISS.load_local(
                str(db_path), 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            
            self.main_retriever = db.as_retriever(
                search_type="similarity",
                search_kwargs={'k': self.config.main_retriever_k}
            )
            
            logger.info("FAISS index loaded successfully.")
            
        except Exception as e:
            error_msg = f"Failed to load FAISS index: {e}"
            logger.error(error_msg)
            raise IndexLoadError(error_msg)

    def get_current_retriever(self):
        """Get the appropriate retriever with proper weighting."""
        if self.session_retriever:
            logger.debug("Using ensemble retriever (FAISS + session file)")
            return EnsembleRetriever(
                retrievers=[self.main_retriever, self.session_retriever],
                weights=self.config.ensemble_weights
            )
        else:
            logger.debug("Using main FAISS retriever")
            return self.main_retriever

    def _calculate_file_hash(self, file_bytes: bytes) -> str:
        """Calculate SHA-256 hash of file for caching."""
        return hashlib.sha256(file_bytes).hexdigest()

    def process_uploaded_file(self, uploaded_file_bytes: bytes) -> Dict[str, Any]:
        """
        Process uploaded file with caching and enhanced error handling.
        
        Returns:
            Dict containing processing results and metadata
        """
        file_hash = self._calculate_file_hash(uploaded_file_bytes)
        
        # Check if file is already processed
        if self._session_file_hash == file_hash:
            logger.info("File already processed (same hash), skipping reprocessing.")
            return {"status": "cached", "hash": file_hash}
        
        logger.info("Processing new uploaded file...")
        
        with performance_timer() as get_time:
            try:
                # Create temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file_bytes)
                    tmp_file_path = tmp_file.name
                
                logger.debug(f"Created temporary file: {tmp_file_path}")
                
                # Load and process document
                loader = PyPDFium2Loader(tmp_file_path)
                docs = loader.load()
                logger.info(f"Loaded {len(docs)} pages from uploaded file.")
                
                # Split into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                chunks = text_splitter.split_documents(docs)
                logger.info(f"Split document into {len(chunks)} chunks.")
                
                # Create vector store
                temp_db = Chroma.from_documents(chunks, self.embeddings)
                self.session_retriever = temp_db.as_retriever(
                    search_kwargs={"k": self.config.session_retriever_k}
                )
                
                # Update cache
                self._session_file_hash = file_hash
                
                processing_time = get_time()
                logger.info(f"File processed successfully in {processing_time:.2f}ms")
                
                return {
                    "status": "processed",
                    "hash": file_hash,
                    "pages": len(docs),
                    "chunks": len(chunks),
                    "processing_time_ms": processing_time
                }
                
            except Exception as e:
                logger.error(f"Failed to process uploaded file: {e}")
                raise RAGSystemError(f"File processing failed: {e}")
                
            finally:
                # Cleanup temporary file
                if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                    os.remove(tmp_file_path)
                    logger.debug("Temporary file cleaned up.")

    def clear_session(self) -> None:
        """Clear session data and cache."""
        self.session_retriever = None
        self._session_file_hash = None
        logger.info("Session data cleared.")

    @lru_cache(maxsize=128)
    def _classify_question_cached(self, question: str) -> str:
        """Cached version of question classification."""
        return self._classify_question_internal(question)

    def _classify_question_internal(self, question: str) -> str:
        """Internal method for question classification."""
        if not self.gemini_llm:
            logger.warning("Gemini LLM unavailable for classification. Using default.")
            return QuestionCategory.STRATEGIC_ANALYTICAL.value

        classification_prompt = self.CLASSIFICATION_PROMPT.format(question=question)
        
        try:
            with performance_timer() as get_time:
                response = self.gemini_llm.invoke(classification_prompt)
                classification_time = get_time()
            
            category = response.content.strip()
            logger.debug(f"Question classified as '{category}' in {classification_time:.2f}ms")
            
            if category in [cat.value for cat in QuestionCategory]:
                return category
            else:
                logger.warning(f"Unexpected classification: '{category}'. Using default.")
                return QuestionCategory.STRATEGIC_ANALYTICAL.value
                
        except Exception as e:
            logger.error(f"Classification failed: {e}. Using default.")
            return QuestionCategory.STRATEGIC_ANALYTICAL.value

def _get_llm_routing(self, question_category: str) -> Tuple[Any, str, Any, str]:
    """
    Enhanced LLM routing based on question category.
    
    Routing Strategy:
    - Regulatory_Factual: Gemini (precise, factual) -> OpenAI (fallback)
    - Strategic_Analytical: OpenAI (creative, analytical) -> Gemini (fallback)  
    - General_Inquiry: OpenAI (conversational) -> Gemini (fallback)
    """
    if question_category == QuestionCategory.REGULATORY_FACTUAL.value:
        return (
            self.gemini_llm, LLMProvider.GEMINI.value,
            self.openai_llm, LLMProvider.OPENAI.value
        )
    else:  # Both Strategic_Analytical and General_Inquiry use OpenAI as primary
        return (
            self.openai_llm, LLMProvider.OPENAI.value,
            self.gemini_llm, LLMProvider.GEMINI.value
        )

    def _invoke_llm(self, llm, prompt: str, image_bytes: Optional[bytes] = None) -> Optional[str]:
        """Invoke LLM with proper error handling and multimodal support."""
        if not llm:
            return None
            
        try:
            if image_bytes:
                logger.debug("Preparing multimodal query with image")
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                message = HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                    ]
                )
                response = llm.invoke([message])
            else:
                response = llm.invoke(prompt)
            
            return response.content if response else None
            
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            return None

def _is_substantive_response(self, response: str) -> bool:
    """Enhanced check for substantive and useful responses."""
    if not response or len(response) < self.config.min_substantive_length:
        return False
    
    # Updated non-substantive indicators
    non_substantive_indicators = [
        "cannot answer this question",
        "don't have enough information", 
        "unable to provide",
        "insufficient context",
        "i don't know",
        "not available in the context",
        "cannot be determine
        
        return not any(indicator in response.lower() for indicator in non_substantive_indicators)

    def query(self, user_question: str, context_text: str, source_docs: List, 
             image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        Enhanced query method with comprehensive monitoring and fallback logic.
        """
        logger.info(f"Processing query: {user_question[:100]}...")
        
        with performance_timer() as get_total_time:
            # Classify question
            question_category = self._classify_question_cached(user_question)
            
            # Prepare prompt
            full_prompt = self.ADVANCED_PROMPT_TEMPLATE.format(
                context=context_text,
                question=user_question
            )
            
            # Determine LLM routing
            primary_llm, primary_name, fallback_llm, fallback_name = self._get_llm_routing(question_category)
            
            response = None
            used_llm = "None"
            fallback_used = False
            
            # Try primary LLM
            if primary_llm:
                logger.debug(f"Attempting query with primary LLM: {primary_name}")
                with performance_timer() as get_primary_time:
                    response = self._invoke_llm(primary_llm, full_prompt, image_bytes)
                    primary_time = get_primary_time()
                
                if response and self._is_substantive_response(response):
                    used_llm = primary_name
                    logger.info(f"Primary LLM ({primary_name}) provided substantive response in {primary_time:.2f}ms")
                else:
                    logger.warning(f"Primary LLM ({primary_name}) response was not substantive")
                    response = None
            
            # Try fallback LLM if needed
            if not response and fallback_llm:
                logger.debug(f"Attempting query with fallback LLM: {fallback_name}")
                fallback_used = True
                
                with performance_timer() as get_fallback_time:
                    response = self._invoke_llm(fallback_llm, full_prompt, image_bytes)
                    fallback_time = get_fallback_time()
                
                if response and self._is_substantive_response(response):
                    used_llm = fallback_name
                    logger.info(f"Fallback LLM ({fallback_name}) provided response in {fallback_time:.2f}ms")
                else:
                    logger.warning(f"Fallback LLM ({fallback_name}) also failed to provide substantive response")
            
            # Prepare final response
            final_answer = response or "Based on the provided context, I cannot answer this question using the available models."
            total_time = get_total_time()
            
            # Record metrics
            metrics = QueryMetrics(
                question_category=question_category,
                primary_llm_used=used_llm,
                fallback_used=fallback_used,
                response_time_ms=total_time,
                context_chunks=len(source_docs),
                response_length=len(final_answer),
                timestamp=time.time()
            )
            self.metrics_history.append(metrics)
            
            logger.info(f"Query completed in {total_time:.2f}ms using {used_llm}")
            
            return {
                "answer": final_answer,
                "source_documents": source_docs,
                "metadata": {
                    "llm_used": used_llm,
                    "question_category": question_category,
                    "response_time_ms": total_time,
                    "fallback_used": fallback_used,
                    "context_chunks": len(source_docs)
                }
            }

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information."""
        return {
            "models": {
                "gemini_llm": self.gemini_llm is not None,
                "gemini_embeddings": self.gemini_embeddings is not None,
                "openai_llm": self.openai_llm is not None,
                "openai_embeddings": self.openai_embeddings is not None,
                "primary_embeddings": type(self.embeddings).__name__
            },
            "index": {
                "faiss_loaded": self.main_retriever is not None,
                "session_active": self.session_retriever is not None,
                "session_file_hash": self._session_file_hash
            },
            "metrics": {
                "total_queries": len(self.metrics_history),
                "cache_size": self._classify_question_cached.cache_info()._asdict() if hasattr(self._classify_question_cached, 'cache_info') else None
            },
            "config": asdict(self.config)
        }

    def get_performance_metrics(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get performance metrics for recent queries."""
        metrics = self.metrics_history[-last_n:] if last_n else self.metrics_history
        
        if not metrics:
            return {"message": "No metrics available"}
        
        response_times = [m.response_time_ms for m in metrics]
        
        return {
            "query_count": len(metrics),
            "avg_response_time_ms": sum(response_times) / len(response_times),
            "min_response_time_ms": min(response_times),
            "max_response_time_ms": max(response_times),
            "fallback_usage_rate": sum(1 for m in metrics if m.fallback_used) / len(metrics),
            "category_distribution": {
                cat.value: sum(1 for m in metrics if m.question_category == cat.value)
                for cat in QuestionCategory
            },
            "llm_usage": {
                provider.value: sum(1 for m in metrics if m.primary_llm_used == provider.value)
                for provider in LLMProvider
            }
        }

# Legacy compatibility - alias for backward compatibility
RAGSystem = EnhancedRAGSystem

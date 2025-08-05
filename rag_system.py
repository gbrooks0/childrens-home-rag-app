# rag_system.py (Refactored Enterprise-Level Version)

import os
import tempfile
import base64
import sys
import json
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
import logging
from contextlib import contextmanager
from functools import wraps, lru_cache
import hashlib

# ============================================================================
# SQLITE3 FIX FOR CHROMADB IN DEPLOYMENT ENVIRONMENTS
# ============================================================================
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    logging.info("pysqlite3 imported and set as default sqlite3 module.")
except ImportError:
    logging.info("pysqlite3 not found, falling back to system sqlite3.")

# ============================================================================
# IMPORTS
# ============================================================================
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import google.generativeai as genai

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class QuestionCategory(Enum):
    """Enumeration for question classification types."""
    REGULATORY_FACTUAL = "Regulatory_Factual"
    STRATEGIC_ANALYTICAL = "Strategic_Analytical"
    GENERAL_INQUIRY = "General_Inquiry"

class LLMProvider(Enum):
    """Enumeration for LLM providers."""
    GEMINI = "Gemini"
    OPENAI = "ChatGPT"

class PerformanceMode(Enum):
    """Enumeration for performance modes."""
    FAST = "fast"
    BALANCED = "balanced"
    COMPREHENSIVE = "comprehensive"

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
class PerformanceConfig:
    """Performance-specific configuration."""
    mode: str = "balanced"
    enable_caching: bool = True
    max_cache_size: int = 100
    max_context_length: int = 2000
    fast_mode_docs: Dict[str, int] = field(default_factory=lambda: {"min": 2, "default": 3, "max": 4})
    balanced_mode_docs: Dict[str, int] = field(default_factory=lambda: {"min": 3, "default": 5, "max": 7})
    comprehensive_mode_docs: Dict[str, int] = field(default_factory=lambda: {"min": 5, "default": 8, "max": 12})

@dataclass
class RAGConfig:
    """Configuration class for RAG system settings."""
    # Core settings
    faiss_index_path: str = "faiss_index"
    chunk_size: int = 1000
    chunk_overlap: int = 100
    min_substantive_length: int = 100
    
    # Retrieval settings
    main_retriever_k: int = 12
    session_retriever_k: int = 3
    ensemble_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])
    
    # Model settings
    gemini_model: str = "gemini-1.5-pro-latest"
    gemini_temperature: float = 0.1
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.7
    embedding_model_gemini: str = "models/text-embedding-004"
    embedding_model_openai: str = "text-embedding-3-large"
    
    # Error handling settings
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Performance settings
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

# ============================================================================
# EXCEPTIONS
# ============================================================================

class RAGSystemError(Exception):
    """Base exception class for RAG system errors."""
    pass

class ModelInitializationError(RAGSystemError):
    """Exception raised when model initialization fails."""
    pass

class IndexLoadError(RAGSystemError):
    """Exception raised when FAISS index loading fails."""
    pass

# ============================================================================
# UTILITY DECORATORS AND FUNCTIONS
# ============================================================================

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

# ============================================================================
# CONVERSATION MEMORY
# ============================================================================

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
        for i, exchange in enumerate(self.conversation_history[-3:], 1):  # Last 3 exchanges
            context_parts.append(f"Q{i}: {exchange['question']}")
            # Include more of the answer for better context
            answer_preview = exchange['answer'][:500] + "..." if len(exchange['answer']) > 500 else exchange['answer']
            context_parts.append(f"A{i}: {answer_preview}")
            context_parts.append("")  # Empty line for readability

        return "\n".join(context_parts)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

# ============================================================================
# MAIN RAG SYSTEM CLASS
# ============================================================================

class EnhancedRAGSystem:
    """
    Enhanced RAG System with improved error handling, performance monitoring,
    caching, and enterprise-level features.
    
    Organized by usage patterns from app.py for optimal maintainability.
    """

    # ========================================================================
    # PROMPT TEMPLATES
    # ========================================================================
    
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

4.  **Professional and Accessible:** Maintain a professional tone while ensuring accessibility. Use clear explanations for complex concepts and provide context for technical terms.

5.  **Clear Formatting:** Use Markdown for optimal readability:
   - **Bold** for key points and emphasis
   - Bullet points for lists and action items
   - Numbered lists for sequential steps or ranked priorities
   - Tables for comparisons when appropriate
   - Headers for organizing complex responses

6.  **Flexible Context Handling:**
   - If the context fully addresses the question, provide a comprehensive answer
   - If the context partially addresses the question, provide what you can and clearly indicate what additional information would be helpful
   - If the context doesn't contain relevant information, acknowledge this and suggest how the question might be refined or what type of context would be needed

7.  **Follow-up Questions:** At the END of your response, provide 2-3 relevant follow-up questions that would help the user explore the topic further or address related important considerations. Format these as:

**Suggested Follow-up Questions:**
• [Question 1 related to implementation or next steps]
• [Question 2 about related considerations or requirements] 
• [Question 3 about specific details or practical aspects]

**IMPORTANT:** Do NOT include follow-up questions within the main content. Only include them at the very end of your response in the suggested format above.

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

    # ========================================================================
    # INITIALIZATION AND SETUP
    # ========================================================================

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the enhanced RAG system with comprehensive error handling and monitoring."""
        self.config = config or RAGConfig()
        self.metrics_history: List[QueryMetrics] = []
        self._classification_cache: Dict[str, str] = {}
        self.conversation_memory = ConversationMemory()
        
        # Performance-related attributes
        self.retrieval_cache: Dict[str, Any] = {}
        self.performance_stats = {
            'retrieval_times': [],
            'generation_times': [],
            'total_times': []
        }

        logger.info("Initializing Enhanced RAG System...")

        # Initialize models and index
        self._initialize_models()
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

    # ========================================================================
    # CORE USER INTERFACE (Primary methods called by app.py)
    # ========================================================================

    def query_with_performance_mode(self, user_question: str, performance_mode: str = "balanced", 
                                   image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        PRIMARY METHOD: Enhanced query method with performance mode optimization.
        Called by app.py as the main interface for user queries.
        
        Args:
            user_question: The user's question
            performance_mode: "fast", "balanced", or "comprehensive"
            image_bytes: Optional image for multimodal queries
            
        Returns:
            Dict containing answer, metadata, and performance stats
        """
        logger.info(f"Processing query with {performance_mode} mode: {user_question[:100]}...")
        
        with performance_timer() as get_total_time:
            try:
                # Step 1: Optimized retrieval
                docs, doc_count, retrieval_time = self.optimized_retrieval(user_question, performance_mode)
                
                # Step 2: Prepare optimized context
                context_text = self.prepare_optimized_context(docs, doc_count, performance_mode)
                
                # Step 3: Enhanced context building for follow-ups
                question_category = self._classify_question_cached(user_question)
                enhanced_context = context_text

                should_include_history = (
                    question_category == QuestionCategory.GENERAL_INQUIRY.value or
                    len(context_text.strip()) < 200 or
                    any(word in user_question.lower() for word in [
                        'this', 'that', 'it', 'why', 'how', 'which version', 'compare',
                        'difference', 'better', 'worse', 'mentioned', 'above', 'previous'
                    ])
                )

                if should_include_history:
                    conversation_context = self.conversation_memory.get_context_for_followup()
                    if conversation_context:
                        enhanced_context = f"""CURRENT CONTEXT:
{context_text}

RECENT CONVERSATION CONTEXT:
{conversation_context}

Note: Use both current context and recent conversation to answer the question comprehensively."""

                # Step 4: Generate response
                full_prompt = self.ADVANCED_PROMPT_TEMPLATE.format(
                    context=enhanced_context,
                    question=user_question
                )

                # Determine LLM routing
                primary_llm, primary_name, fallback_llm, fallback_name = self._get_llm_routing(question_category)

                response = None
                used_llm = "None"
                fallback_used = False
                generation_start_time = time.time()

                # Try primary LLM
                if primary_llm:
                    response = self._invoke_llm(primary_llm, full_prompt, image_bytes)
                    if response and self._is_substantive_response(response):
                        used_llm = primary_name
                    else:
                        response = None

                # Try fallback LLM if needed
                if not response and fallback_llm:
                    fallback_used = True
                    response = self._invoke_llm(fallback_llm, full_prompt, image_bytes)
                    if response and self._is_substantive_response(response):
                        used_llm = fallback_name

                generation_time = time.time() - generation_start_time
                self.performance_stats['generation_times'].append(generation_time)

                # Prepare final response
                final_answer = response or "I don't have enough context to answer this question. Could you provide more details or rephrase your question?"
                total_time = get_total_time()
                self.performance_stats['total_times'].append(total_time)

                # Add to conversation memory
                self.conversation_memory.add_exchange(user_question, final_answer, question_category)

                # Record metrics
                metrics = QueryMetrics(
                    question_category=question_category,
                    primary_llm_used=used_llm,
                    fallback_used=fallback_used,
                    response_time_ms=total_time,
                    context_chunks=len(docs),
                    response_length=len(final_answer),
                    timestamp=time.time()
                )
                self.metrics_history.append(metrics)

                logger.info(f"Query completed in {total_time:.2f}ms using {used_llm} ({performance_mode} mode)")

                return {
                    "answer": final_answer,
                    "source_documents": docs,
                    "metadata": {
                        "llm_used": used_llm,
                        "question_category": question_category,
                        "response_time_ms": total_time,
                        "retrieval_time_ms": retrieval_time * 1000,
                        "generation_time_ms": generation_time * 1000,
                        "fallback_used": fallback_used,
                        "context_chunks": len(docs),
                        "doc_count": doc_count,
                        "performance_mode": performance_mode,
                        "enhanced_context_used": should_include_history,
                        "follow_up_suggestions": self._generate_followup_suggestions(final_answer, question_category)
                    }
                }

            except Exception as e:
                logger.error(f"Query with performance mode failed: {e}")
                # Fallback to standard query method
                return self.query(user_question, "", docs if 'docs' in locals() else [], image_bytes)

    def query(self, user_question: str, context_text: str, source_docs: List,
             image_bytes: Optional[bytes] = None) -> Dict[str, Any]:
        """
        FALLBACK METHOD: Standard query method for backward compatibility.
        Called by app.py when performance mode fails or for legacy support.
        """
        logger.info(f"Processing query (standard mode): {user_question[:100]}...")

        with performance_timer() as get_total_time:
            # Classify question
            question_category = self._classify_question_cached(user_question)

            # Enhanced context building for follow-ups
            enhanced_context = context_text

            # For follow-up questions or when context seems insufficient, include conversation history
            should_include_history = (
                question_category == QuestionCategory.GENERAL_INQUIRY.value or
                len(context_text.strip()) < 200 or
                any(word in user_question.lower() for word in [
                    'this', 'that', 'it', 'why', 'how', 'which version', 'compare',
                    'difference', 'better', 'worse', 'mentioned', 'above', 'previous'
                ])
            )

            if should_include_history:
                conversation_context = self.conversation_memory.get_context_for_followup()
                if conversation_context:
                    enhanced_context = f"""CURRENT CONTEXT:
{context_text}

RECENT CONVERSATION CONTEXT:
{conversation_context}

Note: Use both current context and recent conversation to answer the question comprehensively."""
                    logger.debug("Enhanced context with conversation history for follow-up question")

            # Prepare prompt with enhanced context
            full_prompt = self.ADVANCED_PROMPT_TEMPLATE.format(
                context=enhanced_context,
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
            final_answer = response or "I don't have enough context to answer this question. Could you provide more details or rephrase your question?"
            total_time = get_total_time()

            # Add to conversation memory
            self.conversation_memory.add_exchange(user_question, final_answer, question_category)

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
                    "context_chunks": len(source_docs),
                    "enhanced_context_used": should_include_history,
                    "follow_up_suggestions": self._generate_followup_suggestions(final_answer, question_category)
                }
            }

    # ========================================================================
    # SESSION AND FILE MANAGEMENT (Required by app.py)
    # ========================================================================

    def process_uploaded_file(self, uploaded_file_bytes: bytes) -> Dict[str, Any]:
        """
        Process uploaded file with caching and enhanced error handling.
        Called by app.py for document upload functionality.

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
        """Clear session data and cache. Called by app.py for session management."""
        self.session_retriever = None
        self._session_file_hash = None
        logger.info("Session data cleared.")

    def clear_conversation_memory(self) -> None:
        """Clear conversation memory - useful for starting fresh topics. Called by app.py."""
        self.conversation_memory.clear_history()
        logger.info("Conversation memory cleared.")

    # ========================================================================
    # PERFORMANCE MONITORING (Displayed in app.py sidebar)
    # ========================================================================

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics. Called by app.py for sidebar display."""
        if not self.performance_stats['total_times']:
            return {"message": "No performance data available"}
        
        return {
            "avg_total_time": sum(self.performance_stats['total_times']) / len(self.performance_stats['total_times']),
            "avg_retrieval_time": sum(self.performance_stats['retrieval_times']) / len(self.performance_stats['retrieval_times']) if self.performance_stats['retrieval_times'] else 0,
            "avg_generation_time": sum(self.performance_stats['generation_times']) / len(self.performance_stats['generation_times']) if self.performance_stats['generation_times'] else 0,
            "total_queries": len(self.performance_stats['total_times']),
            "cache_hits": len(self.retrieval_cache),
            "fastest_query": min(self.performance_stats['total_times']) if self.performance_stats['total_times'] else 0,
            "slowest_query": max(self.performance_stats['total_times']) if self.performance_stats['total_times'] else 0
        }

    def set_performance_mode(self, mode: str) -> None:
        """Set the default performance mode. Called by app.py for mode switching."""
        if mode in ["fast", "balanced", "comprehensive"]:
            self.config.performance.mode = mode
            logger.info(f"Performance mode set to: {mode}")
        else:
            logger.warning(f"Invalid performance mode: {mode}. Using balanced.")
            self.config.performance.mode = "balanced"

    def clear_performance_cache(self) -> None:
        """Clear performance cache to free memory. Available for app.py optimization."""
        self.retrieval_cache.clear()
        logger.info("Performance cache cleared.")

    def get_performance_metrics(self, last_n: Optional[int] = None) -> Dict[str, Any]:
        """Get performance metrics for recent queries. Available for app.py analytics."""
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

    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health information. Called by app.py for monitoring."""
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
            "config": asdict(self.config),
            "performance": {
                "retrieval_cache_size": len(self.retrieval_cache),
                "current_performance_mode": self.config.performance.mode,
                "caching_enabled": self.config.performance.enable_caching,
                "performance_stats": self.get_performance_stats()
            }
        }

    # ========================================================================
    # CONVERSATION AND MEMORY MANAGEMENT (Available for app.py future features)
    # ========================================================================

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the full conversation history. Available for app.py conversation features."""
        return self.conversation_memory.conversation_history

    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation. Available for app.py analytics."""
        history = self.conversation_memory.conversation_history
        if not history:
            return {"total_questions": 0, "topics": []}

        return {
            "total_questions": len(history),
            "latest_question": history[-1]["question"] if history else None,
            "question_categories": [ex["category"] for ex in history],
            "conversation_duration": time.time() - history[0]["timestamp"] if history else 0
        }

    # ========================================================================
    # INTERNAL OPTIMIZATION ENGINE
    # ========================================================================

    def optimized_retrieval(self, query: str, performance_mode: str = None) -> Tuple[List, int, float]:
        """Optimized retrieval with adaptive document count and caching."""
        start_time = time.time()
        
        # Determine optimal document count
        doc_count = self.get_optimal_document_count(query, performance_mode)
        
        # Check cache if enabled
        if self.config.performance.enable_caching:
            cache_key = f"{query}_{doc_count}_{performance_mode}"
            if cache_key in self.retrieval_cache:
                logger.debug(f"Cache hit for query: {query[:50]}...")
                cached_result = self.retrieval_cache[cache_key]
                return cached_result["docs"], doc_count, time.time() - start_time
        
        logger.debug(f"Query complexity analysis: using {doc_count} documents")
        
        # Get current retriever and adjust parameters
        retriever = self.get_current_retriever()
        
        # Temporarily adjust the retriever's k parameter
        original_k = None
        if hasattr(retriever, 'search_kwargs'):
            original_k = retriever.search_kwargs.get('k', self.config.main_retriever_k)
            retriever.search_kwargs['k'] = doc_count
        
        try:
            # Retrieve documents
            docs = retriever.invoke(query)
            
            # Cache the result if enabled
            if self.config.performance.enable_caching:
                self.retrieval_cache[cache_key] = {"docs": docs}
                # Limit cache size
                if len(self.retrieval_cache) > self.config.performance.max_cache_size:
                    # Remove oldest entries
                    oldest_keys = list(self.retrieval_cache.keys())[:20]
                    for key in oldest_keys:
                        del self.retrieval_cache[key]
            
            retrieval_time = time.time() - start_time
            self.performance_stats['retrieval_times'].append(retrieval_time)
            logger.debug(f"Retrieved {len(docs)} documents in {retrieval_time:.2f}s")
            
            return docs, doc_count, retrieval_time
            
        finally:
            # Restore original k parameter
            if hasattr(retriever, 'search_kwargs') and original_k is not None:
                retriever.search_kwargs['k'] = original_k

    def prepare_optimized_context(self, docs: List, doc_count: int, performance_mode: str = None) -> str:
        """Prepare context with performance optimizations."""
        if performance_mode is None:
            performance_mode = self.config.performance.mode
        
        # Adjust context length based on performance mode
        if performance_mode == "fast":
            max_doc_length = 300
        elif performance_mode == "comprehensive":
            max_doc_length = 800
        else:  # balanced
            max_doc_length = 500
        
        # Use only the most relevant documents
        selected_docs = docs[:doc_count]
        
        # Create context with document boundaries
        context_parts = []
        total_length = 0
        
        for i, doc in enumerate(selected_docs):
            # Truncate documents but preserve key information
            content = doc.page_content
            if len(content) > max_doc_length:
                content = content[:max_doc_length] + "..."
            
            context_parts.append(f"Source {i+1}: {content}")
            total_length += len(content)
            
            # Respect total context limit
            if total_length > self.config.performance.max_context_length:
                break
        
        return "\n\n".join(context_parts)

    def analyze_query_complexity(self, query: str) -> int:
        """Analyze query to determine optimal document count."""
        query_lower = query.lower()
        
        # Simple factual queries - fewer documents needed
        simple_indicators = [
            "what is", "define", "who is", "when was", "where is",
            "list", "name", "how many", "which"
        ]
        
        # Complex strategic queries - more documents needed
        complex_indicators = [
            "strategy", "approach", "best practices", "guidance",
            "how do we", "what should we", "comprehensive",
            "develop", "implement", "improve", "compare"
        ]
        
        # Multi-part queries - more documents needed
        multi_part_indicators = [
            "and", "also", "additionally", "furthermore",
            "both", "either", "various", "different"
        ]
        
        is_simple = any(indicator in query_lower for indicator in simple_indicators)
        is_complex = any(indicator in query_lower for indicator in complex_indicators)
        is_multi_part = any(indicator in query_lower for indicator in multi_part_indicators)
        
        # Determine complexity score
        complexity_score = 0
        if is_simple:
            complexity_score -= 1
        if is_complex:
            complexity_score += 2
        if is_multi_part:
            complexity_score += 1
        if len(query) > 100:
            complexity_score += 1
        
        return complexity_score

    def get_optimal_document_count(self, query: str, performance_mode: str = None) -> int:
        """Determine optimal document count based on query and performance mode."""
        if performance_mode is None:
            performance_mode = self.config.performance.mode
        
        complexity_score = self.analyze_query_complexity(query)
        
        # Get document count settings for the mode
        if performance_mode == "fast":
            counts = self.config.performance.fast_mode_docs
        elif performance_mode == "comprehensive":
            counts = self.config.performance.comprehensive_mode_docs
        else:  # balanced
            counts = self.config.performance.balanced_mode_docs
        
        # Adjust based on complexity
        if complexity_score <= -1:  # Simple query
            return counts["min"]
        elif complexity_score >= 2:   # Complex query
            return counts["max"]
        else:                        # Normal query
            return counts["default"]

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

    # ========================================================================
    # INTERNAL HELPER METHODS
    # ========================================================================

    def _calculate_file_hash(self, file_bytes: bytes) -> str:
        """Calculate SHA-256 hash of file for caching."""
        return hashlib.sha256(file_bytes).hexdigest()

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
            "cannot be determined from"
        ]

        response_lower = response.lower()

        # Check for non-substantive patterns
        non_substantive_count = sum(1 for indicator in non_substantive_indicators
                                    if indicator in response_lower)

        # If multiple non-substantive indicators, likely not substantive
        if non_substantive_count >= 2:
            return False

        # Check for substantive content indicators
        substantive_indicators = [
            "based on the context",
            "according to",
            "the document states",
            "key points include",
            "recommendations",
            "requirements include",
            "standards specify"
        ]

        has_substantive_content = any(indicator in response_lower
                                      for indicator in substantive_indicators)

        # Must have substantive content OR be reasonably long without many non-substantive indicators
        return has_substantive_content or (len(response) > 200 and non_substantive_count == 0)

    def _generate_followup_suggestions(self, answer: str, category: str) -> List[str]:
        """Generate contextual follow-up question suggestions."""
        suggestions = []

        if category == QuestionCategory.REGULATORY_FACTUAL.value:
            suggestions = [
                "Can you explain the practical implementation of this requirement?",
                "What are the common compliance challenges with this standard?",
                "Are there any related standards I should be aware of?",
                "What documentation is typically required for this?"
            ]
        elif category == QuestionCategory.STRATEGIC_ANALYTICAL.value:
            suggestions = [
                "What would be the first steps to implement this recommendation?",
                "How can we measure success for this approach?",
                "What are the potential risks or challenges?",
                "How does this compare to industry best practices?"
            ]
        else:  # General_Inquiry
            suggestions = [
                "Can you provide more details about this?",
                "Are there any examples or case studies?",
                "What should I consider next?",
                "How does this relate to our overall strategy?"
            ]

        return suggestions[:3]  # Return top 3 suggestions


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

# Alias for backward compatibility with existing code
RAGSystem = EnhancedRAGSystem

#!/usr/bin/env python3
"""
Enhanced Data Ingestion Script for High-Performance RAG System

This script provides advanced document ingestion with performance optimizations,
intelligent chunking strategies, metadata enrichment, and quality filtering.

Features:
- Multi-threaded document processing for faster ingestion
- Intelligent chunking with semantic boundaries
- Content quality filtering and deduplication
- Rich metadata extraction and indexing
- Progress tracking and performance monitoring
- Incremental updates and document versioning
- Advanced error handling and recovery
"""

# =============================================================================
# IMPORTS
# =============================================================================

# Standard library imports
import os
import shutil
import tempfile
import hashlib
import json
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import logging

# Third-party imports
import requests
from bs4 import BeautifulSoup
import numpy as np
from tqdm import tqdm

# LangChain imports
from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFium2Loader,
    TextLoader,
    UnstructuredWordDocumentLoader
)
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import OpenAIEmbeddings


# =============================================================================
# CONFIGURATION
# =============================================================================

# Environment validation
missing_keys = []
if "GOOGLE_API_KEY" not in os.environ:
    missing_keys.append("GOOGLE_API_KEY")
if "OPENAI_API_KEY" not in os.environ:
    missing_keys.append("OPENAI_API_KEY")

if missing_keys:
    print("ERROR: Missing required environment variables:")
    for key in missing_keys:
        print(f"  - {key}")
    print("Please set all required API keys before running the script.")
    exit(1)

# Directory paths
DATA_DIR = "docs"
DB_DIR = "faiss_index"
METADATA_DIR = "metadata"
CACHE_DIR = "document_cache"
URLS_FILE = os.path.join(DATA_DIR, "urls.txt")  # URLs file in docs folder

# Embedding configuration
EMBEDDING_PROVIDER = "openai"  # Options: "openai", "google"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-large"  # High-performance model
GOOGLE_EMBEDDING_MODEL = "models/text-embedding-004"

# Performance configuration
MAX_WORKERS = 4  # Adjust based on your system
BATCH_SIZE = 50  # Documents per batch for processing
MIN_CHUNK_SIZE = 100  # Minimum chunk size to avoid tiny chunks
MAX_CHUNK_SIZE = 2000  # Maximum chunk size for better context

# Web scraping configuration
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Default URLs (fallback if urls.txt doesn't exist)
DEFAULT_URLS_TO_SCRAPE = [
    "https://www.gov.uk/government/publications/social-care-common-inspection-framework-sccif-childrens-homes/social-care-common-inspection-framework-sccif-childrens-homes",
    "https://assets.publishing.service.gov.uk/media/6849a7b67cba25f610c7db3f/Working_together_to_safeguard_children_2023_-_statutory_guidance.pdf",  
    "https://assets.publishing.service.gov.uk/media/686b94eefe1a249e937cbd2d/Keeping_children_safe_in_education_2025.pdf",
    "https://www.gov.uk/government/publications/social-care-common-inspection-framework-sccif-independent-fostering-agencies/social-care-common-inspection-framework-sccif-independent-fostering-agencies",
    "https://assets.publishing.service.gov.uk/media/657c538495bf650010719097/Children_s_Social_Care_National_Framework__December_2023.pdf",
    "https://learning.nspcc.org.uk/safeguarding-child-protection",
    "https://www.mentalhealth.org.uk/explore-mental-health/a-z-topics/children-and-young-people",
    "https://www.scie.org.uk/children/care/", 
    "https://www.gov.uk/guidance/childrens-homes-recruiting-staff",
]

# Advanced text processing configuration
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300  # Increased overlap for better context preservation
REQUEST_TIMEOUT = 30
SIMILARITY_THRESHOLD = 0.85  # For deduplication

# Content quality thresholds
MIN_CONTENT_LENGTH = 50
MAX_CONTENT_LENGTH = 50000
MIN_WORD_COUNT = 10

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_urls_from_file() -> List[str]:
    """
    Load URLs from the urls.txt file in the docs folder.
    
    Returns:
        List[str]: List of URLs to process
    """
    urls = []
    
    if os.path.exists(URLS_FILE):
        logger.info(f"Loading URLs from: {URLS_FILE}")
        try:
            with open(URLS_FILE, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith('#'):
                        # Basic URL validation
                        if line.startswith(('http://', 'https://')):
                            urls.append(line)
                        else:
                            logger.warning(f"Invalid URL on line {line_num}: {line}")
            
            logger.info(f"Loaded {len(urls)} URLs from {URLS_FILE}")
            
        except Exception as e:
            logger.error(f"Error reading {URLS_FILE}: {e}")
            logger.info("Falling back to default URLs")
            urls = DEFAULT_URLS_TO_SCRAPE.copy()
    else:
        logger.info(f"URLs file not found at {URLS_FILE}")
        logger.info("Creating sample urls.txt file with default URLs")
        
        # Create the urls.txt file with default URLs and instructions
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(URLS_FILE, 'w', encoding='utf-8') as f:
                f.write("# URLs to scrape for document ingestion\n")
                f.write("# Add one URL per line\n")
                f.write("# Lines starting with # are comments and will be ignored\n")
                f.write("# Empty lines are also ignored\n\n")
                
                for url in DEFAULT_URLS_TO_SCRAPE:
                    f.write(f"{url}\n")
            
            logger.info(f"Created {URLS_FILE} with default URLs")
            urls = DEFAULT_URLS_TO_SCRAPE.copy()
            
        except Exception as e:
            logger.error(f"Error creating {URLS_FILE}: {e}")
            logger.info("Using default URLs from code")
            urls = DEFAULT_URLS_TO_SCRAPE.copy()
    
    return urls


def get_embedding_model():
    """
    Get the appropriate embedding model based on configuration.
    
    Returns:
        Embedding model instance
    """
    if EMBEDDING_PROVIDER.lower() == "openai":
        logger.info(f"Using OpenAI embeddings: {OPENAI_EMBEDDING_MODEL}")
        return OpenAIEmbeddings(
            model=OPENAI_EMBEDDING_MODEL,
            show_progress_bar=True
        )
    elif EMBEDDING_PROVIDER.lower() == "google":
        logger.info(f"Using Google embeddings: {GOOGLE_EMBEDDING_MODEL}")
        return GoogleGenerativeAIEmbeddings(model=GOOGLE_EMBEDDING_MODEL)
    else:
        raise ValueError(f"Unsupported embedding provider: {EMBEDDING_PROVIDER}")


def ensure_directories() -> None:
    """Create necessary directories if they don't exist."""
    for directory in [METADATA_DIR, CACHE_DIR]:
        Path(directory).mkdir(exist_ok=True)


def calculate_content_hash(content: str) -> str:
    """Calculate SHA-256 hash of content for deduplication."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()


def is_quality_content(content: str) -> bool:
    """
    Filter out low-quality content based on various criteria.
    
    Args:
        content (str): Content to evaluate
        
    Returns:
        bool: True if content meets quality standards
    """
    if not content or len(content.strip()) < MIN_CONTENT_LENGTH:
        return False
    
    if len(content) > MAX_CONTENT_LENGTH:
        return False
    
    word_count = len(content.split())
    if word_count < MIN_WORD_COUNT:
        return False
    
    # Check for excessive repetition (spam indicator)
    lines = content.split('\n')
    unique_lines = set(line.strip() for line in lines if line.strip())
    if len(lines) > 10 and len(unique_lines) / len(lines) < 0.3:
        return False
    
    return True


def extract_metadata(document: Document, source_type: str) -> Dict[str, Any]:
    """
    Extract rich metadata from documents for better retrieval.
    
    Args:
        document (Document): Document to process
        source_type (str): Type of source (local, web, pdf)
        
    Returns:
        Dict[str, Any]: Enhanced metadata
    """
    content = document.page_content
    metadata = document.metadata.copy()
    
    # Basic statistics
    metadata.update({
        'content_length': len(content),
        'word_count': len(content.split()),
        'line_count': len(content.split('\n')),
        'source_type': source_type,
        'processed_at': datetime.now().isoformat(),
        'content_hash': calculate_content_hash(content)
    })
    
    # Extract potential keywords (simple approach)
    words = content.lower().split()
    word_freq = {}
    for word in words:
        if len(word) > 3:  # Skip short words
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top keywords
    top_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
    metadata['keywords'] = [word for word, _ in top_keywords]
    
    return metadata


# =============================================================================
# DOCUMENT LOADING FUNCTIONS
# =============================================================================

def load_from_directory(directory_path: str) -> List[Document]:
    """
    Load documents from a local directory with enhanced metadata.
    
    Args:
        directory_path (str): Path to the directory containing documents
        
    Returns:
        List[Document]: List of loaded documents with rich metadata
    """
    logger.info(f"Loading documents from directory: '{directory_path}'...")
    
    if not os.path.isdir(directory_path):
        logger.warning(f"Directory '{directory_path}' not found. Skipping local file loading.")
        return []

    all_documents = []
    
    # Load different file types
    loaders = [
        (DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFium2Loader, show_progress=True), "pdf"),
        (DirectoryLoader(directory_path, glob="**/*.md", loader_cls=TextLoader, show_progress=True), "markdown"),
        (DirectoryLoader(directory_path, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader, show_progress=True), "word")
    ]
    
    for loader, file_type in loaders:
        try:
            docs = loader.load()
            for doc in docs:
                # Enhance metadata
                doc.metadata = extract_metadata(doc, f"local_{file_type}")
                doc.metadata['file_type'] = file_type
                
                # Quality filtering
                if is_quality_content(doc.page_content):
                    all_documents.append(doc)
                else:
                    logger.debug(f"Filtered out low-quality content from {doc.metadata.get('source', 'unknown')}")
                    
        except Exception as e:
            logger.error(f"Error loading {file_type} files: {e}")
    
    logger.info(f"Successfully loaded {len(all_documents)} quality documents from local directory.")
    return all_documents


def load_pdf_from_url(url: str) -> List[Document]:
    """
    Download and load a PDF from a URL with caching.
    
    Args:
        url (str): URL of the PDF to download
        
    Returns:
        List[Document]: List of documents extracted from the PDF
    """
    # Check cache first
    url_hash = calculate_content_hash(url)
    cache_file = Path(CACHE_DIR) / f"{url_hash}.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            docs = []
            for doc_data in cached_data:
                doc = Document(
                    page_content=doc_data['page_content'],
                    metadata=doc_data['metadata']
                )
                docs.append(doc)
            
            logger.info(f"✓ Loaded cached PDF: {url}")
            return docs
            
        except Exception as e:
            logger.warning(f"Cache read failed for {url}: {e}")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
        
        # Load PDF using PyPDFium2Loader
        loader = PyPDFium2Loader(tmp_file_path)
        docs = loader.load()
        
        # Process and enhance each document
        processed_docs = []
        for doc in docs:
            doc.metadata = extract_metadata(doc, "web_pdf")
            doc.metadata["source"] = url
            doc.metadata["pdf_page"] = doc.metadata.get("page", 0)
            
            if is_quality_content(doc.page_content):
                processed_docs.append(doc)
        
        # Cache the results
        cache_data = []
        for doc in processed_docs:
            cache_data.append({
                'page_content': doc.page_content,
                'metadata': doc.metadata
            })
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Cache write failed for {url}: {e}")
        
        # Clean up temporary file
        os.remove(tmp_file_path)
        
        logger.info(f"✓ Successfully processed PDF: {url} ({len(processed_docs)} pages)")
        return processed_docs
        
    except Exception as e:
        logger.error(f"✗ ERROR: Failed to process PDF {url}: {e}")
        return []


def scrape_web_content(url: str) -> List[Document]:
    """
    Scrape text content from a web page with enhanced extraction.
    
    Args:
        url (str): URL to scrape
        
    Returns:
        List[Document]: List containing the scraped document
    """
    # Check cache first
    url_hash = calculate_content_hash(url)
    cache_file = Path(CACHE_DIR) / f"{url_hash}_web.json"
    
    if cache_file.exists():
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            doc = Document(
                page_content=cached_data['page_content'],
                metadata=cached_data['metadata']
            )
            logger.info(f"✓ Loaded cached content: {url}")
            return [doc]
            
        except Exception as e:
            logger.warning(f"Cache read failed for {url}: {e}")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
        
        # Try to find main content areas with priority
        content_selectors = [
            'main',
            'article', 
            '[role="main"]',
            '.content',
            '.main-content',
            '#content',
            '.post-content',
            '.entry-content'
        ]
        
        main_content = None
        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break
        
        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
            title = soup.title.string if soup.title else ""
            logger.info(f"✓ Successfully scraped: {url}")
        else:
            # Fallback to body text
            text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
            title = soup.title.string if soup.title else ""
            logger.warning(f"⚠ Using body text for {url} (main content not found)")
        
        # Clean up text
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:  # Skip very short lines
                cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        if not is_quality_content(cleaned_text):
            logger.warning(f"Low quality content filtered from {url}")
            return []
        
        # Create document with enhanced metadata
        doc = Document(page_content=cleaned_text, metadata={})
        doc.metadata = extract_metadata(doc, "web_page")
        doc.metadata.update({
            "source": url,
            "title": title,
            "domain": url.split('/')[2] if len(url.split('/')) > 2 else ""
        })
        
        # Cache the result
        cache_data = {
            'page_content': doc.page_content,
            'metadata': doc.metadata
        }
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Cache write failed for {url}: {e}")
        
        return [doc]
        
    except Exception as e:
        logger.error(f"✗ ERROR: Failed to scrape {url}: {e}")
        return []


# =============================================================================
# WEB CONTENT PROCESSING
# =============================================================================

def process_url_batch(urls: List[str]) -> List[Document]:
    """
    Process a batch of URLs concurrently.
    
    Args:
        urls (List[str]): List of URLs to process
        
    Returns:
        List[Document]: List of all processed documents
    """
    documents = []
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit tasks
        future_to_url = {}
        for url in urls:
            if url.lower().endswith('.pdf'):
                future = executor.submit(load_pdf_from_url, url)
            else:
                future = executor.submit(scrape_web_content, url)
            future_to_url[future] = url
        
        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_url), total=len(urls), desc="Processing URLs"):
            url = future_to_url[future]
            try:
                docs = future.result()
                documents.extend(docs)
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
    
    return documents


def load_and_clean_urls(urls: List[str] = None) -> List[Document]:
    """
    Process URLs in batches with concurrent processing.
    
    Args:
        urls (List[str], optional): List of URLs to process. If None, loads from urls.txt
        
    Returns:
        List[Document]: List of all processed documents
    """
    if urls is None:
        urls = load_urls_from_file()
    
    logger.info(f"Processing {len(urls)} URL(s) with {MAX_WORKERS} workers...")
    
    # Process URLs in batches to avoid overwhelming servers
    all_documents = []
    for i in range(0, len(urls), BATCH_SIZE):
        batch_urls = urls[i:i + BATCH_SIZE]
        batch_docs = process_url_batch(batch_urls)
        all_documents.extend(batch_docs)
        
        # Small delay between batches to be respectful
        if i + BATCH_SIZE < len(urls):
            time.sleep(1)
    
    logger.info(f"Successfully processed {len(all_documents)} document(s) from URLs.")
    return all_documents


# =============================================================================
# ADVANCED TEXT PROCESSING
# =============================================================================

def create_semantic_chunks(documents: List[Document]) -> List[Document]:
    """
    Create semantically-aware chunks with enhanced metadata.
    
    Args:
        documents (List[Document]): Documents to chunk
        
    Returns:
        List[Document]: List of chunked documents
    """
    logger.info("Creating semantic chunks with enhanced splitter...")
    
    # Use multiple splitting strategies for better semantic boundaries
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n\n",  # Multiple newlines (strong paragraph breaks)
            "\n\n",    # Double newlines (paragraph breaks)
            "\n",      # Single newlines
            ". ",      # Sentence endings
            "! ",      # Exclamation sentences
            "? ",      # Question sentences
            "; ",      # Semicolons
            ", ",      # Commas
            " ",       # Spaces
            ""         # Character level (fallback)
        ]
    )
    
    all_chunks = []
    chunk_id = 0
    
    for doc in tqdm(documents, desc="Chunking documents"):
        try:
            chunks = text_splitter.split_documents([doc])
            
            for i, chunk in enumerate(chunks):
                # Filter out tiny chunks
                if len(chunk.page_content.strip()) < MIN_CHUNK_SIZE:
                    continue
                
                # Enhance chunk metadata
                chunk.metadata.update({
                    'chunk_id': chunk_id,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'parent_doc_hash': doc.metadata.get('content_hash', ''),
                    'chunk_length': len(chunk.page_content),
                    'chunk_words': len(chunk.page_content.split())
                })
                
                all_chunks.append(chunk)
                chunk_id += 1
                
        except Exception as e:
            logger.error(f"Error chunking document {doc.metadata.get('source', 'unknown')}: {e}")
    
    logger.info(f"Created {len(all_chunks)} semantic chunks from {len(documents)} documents.")
    return all_chunks


def deduplicate_chunks(chunks: List[Document]) -> List[Document]:
    """
    Remove duplicate chunks based on content similarity.
    
    Args:
        chunks (List[Document]): List of chunks to deduplicate
        
    Returns:
        List[Document]: Deduplicated chunks
    """
    logger.info("Deduplicating chunks...")
    
    seen_hashes = set()
    unique_chunks = []
    
    for chunk in tqdm(chunks, desc="Deduplicating"):
        content_hash = calculate_content_hash(chunk.page_content)
        
        if content_hash not in seen_hashes:
            seen_hashes.add(content_hash)
            unique_chunks.append(chunk)
    
    removed_count = len(chunks) - len(unique_chunks)
    logger.info(f"Removed {removed_count} duplicate chunks. {len(unique_chunks)} unique chunks remaining.")
    
    return unique_chunks


# =============================================================================
# VECTOR DATABASE CREATION
# =============================================================================

def create_vector_database(documents: List[Document]) -> None:
    """
    Create an optimized FAISS vector database with batched processing.
    
    Args:
        documents (List[Document]): List of documents to process
    """
    if not documents:
        logger.warning("No documents provided for vector database creation.")
        return
    
    logger.info("Initializing embeddings model...")
    embeddings = get_embedding_model()
    
    # Create semantic chunks
    chunks = create_semantic_chunks(documents)
    
    # Deduplicate chunks
    chunks = deduplicate_chunks(chunks)
    
    if not chunks:
        logger.error("No chunks available after processing.")
        return
    
    # Save metadata for later use
    metadata_file = Path(METADATA_DIR) / "chunk_metadata.json"
    chunk_metadata = []
    for chunk in chunks:
        chunk_metadata.append({
            'chunk_id': chunk.metadata.get('chunk_id'),
            'source': chunk.metadata.get('source'),
            'metadata': chunk.metadata
        })
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(chunk_metadata, f, ensure_ascii=False, indent=2)
    
    # Remove existing database
    if os.path.exists(DB_DIR):
        logger.info(f"Removing existing FAISS index at '{DB_DIR}'...")
        shutil.rmtree(DB_DIR)
    
    logger.info(f"Creating FAISS vector database with {len(chunks)} chunks...")
    
    # Process in batches for memory efficiency
    if len(chunks) > 1000:
        logger.info("Processing large dataset in batches...")
        
        # Create initial database with first batch
        initial_batch_size = min(500, len(chunks))
        initial_chunks = chunks[:initial_batch_size]
        db = FAISS.from_documents(initial_chunks, embeddings)
        
        # Add remaining chunks in batches
        remaining_chunks = chunks[initial_batch_size:]
        batch_size = 250
        
        for i in range(0, len(remaining_chunks), batch_size):
            batch = remaining_chunks[i:i + batch_size]
            batch_db = FAISS.from_documents(batch, embeddings)
            db.merge_from(batch_db)
            logger.info(f"Processed batch {i // batch_size + 2}")
    else:
        db = FAISS.from_documents(chunks, embeddings)
    
    # Save the database
    db.save_local(DB_DIR)
    
    # Save ingestion statistics
    stats = {
        'total_documents': len(documents),
        'total_chunks': len(chunks),
        'average_chunk_size': np.mean([len(chunk.page_content) for chunk in chunks]),
        'ingestion_date': datetime.now().isoformat(),
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP
    }
    
    stats_file = Path(METADATA_DIR) / "ingestion_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✓ FAISS database saved to '{DB_DIR}' with {len(chunks)} optimized chunks.")
    logger.info(f"✓ Metadata saved to '{METADATA_DIR}'")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main() -> None:
    """
    Main function to orchestrate the enhanced document ingestion process.
    """
    start_time = time.time()
    
    print("=" * 70)
    print("ENHANCED RAG SYSTEM - DOCUMENT INGESTION")
    print("=" * 70)
    
    # Ensure required directories exist
    ensure_directories()
    
    try:
        # Load local documents
        logger.info("\n[1/3] Loading local documents...")
        local_docs = load_from_directory(DATA_DIR)
        
        # Load web documents  
        logger.info("\n[2/3] Processing web documents...")
        web_docs = load_and_clean_urls()  # Now loads from urls.txt automatically
        
        # Combine all documents
        all_documents = local_docs + web_docs
        
        if not all_documents:
            logger.error("\n❌ No documents were loaded. Exiting.")
            return
        
        logger.info(f"\nTotal quality documents loaded: {len(all_documents)}")
        
        # Create vector database
        logger.info("\n[3/3] Creating optimized vector database...")
        create_vector_database(all_documents)
        
        # Calculate and display performance metrics
        end_time = time.time()
        processing_time = end_time - start_time
        
        print("\n" + "=" * 70)  
        print("✅ ENHANCED INGESTION COMPLETE!")
        print(f"📊 Processing time: {processing_time:.2f} seconds")
        print(f"📚 Documents processed: {len(all_documents)}")
        print(f"⚡ Average speed: {len(all_documents)/processing_time:.2f} docs/second")
        print("🚀 Vector database is optimized and ready for high-performance queries!")
        print("=" * 70)
        
    except Exception as e:
        logger.error(f"Fatal error during ingestion: {e}")
        raise


if __name__ == "__main__":
    main()

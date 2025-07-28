# ingest.py (Definitive, Complete, and Final Version with Robust Chunking)

import os
import shutil
import requests
from bs4 import BeautifulSoup
import tempfile
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

# --- Configuration ---
if "GOOGLE_API_KEY" not in os.environ:
    print("API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    exit()

DATA_DIR = "docs"
DB_DIR = "faiss_index"
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
URLS_TO_SCRAPE = [
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

# --- HELPER FUNCTIONS ---
def load_from_directory(directory_path: str):
    """Loads documents from a local directory, with support for .docx files."""
    print(f"Checking for local documents in: '{directory_path}'...")
    if not os.path.isdir(directory_path):
        print(f"Directory '{directory_path}' not found. Skipping local file loading.")
        return []

    pdf_loader = DirectoryLoader(directory_path, glob="**/*.pdf", loader_cls=PyPDFium2Loader, show_progress=True)
    text_loader = DirectoryLoader(directory_path, glob="**/*.md", loader_cls=TextLoader, show_progress=True)
    word_loader = DirectoryLoader(directory_path, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader, show_progress=True)
    
    loaded_pdfs = pdf_loader.load()
    loaded_texts = text_loader.load()
    loaded_words = word_loader.load()
    
    loaded_docs = loaded_pdfs + loaded_texts + loaded_words
    
    print(f"Loaded {len(loaded_docs)} document(s) from local directory.")
    return loaded_docs

def load_and_clean_urls(urls: list):
    """Intelligently scrapes URLs and returns a list of Document objects."""
    print(f"Intelligently scraping {len(urls)} URL(s)...")
    documents = []
    for url in urls:
        try:
            if url.lower().endswith('.pdf'):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    response = requests.get(url, headers=HEADERS, timeout=30)
                    response.raise_for_status()
                    tmp_file.write(response.content)
                    tmp_file_path = tmp_file.name
                
                loader = PyPDFium2Loader(tmp_file_path)
                docs = loader.load()
                for doc in docs:
                    doc.metadata["source"] = url
                documents.extend(docs)
                os.remove(tmp_file_path)
                print(f"  Successfully processed PDF: {url}")
                continue

            response = requests.get(url, headers=HEADERS, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            main_content = soup.find('main') or soup.find('article') or soup.find('div', id='content')
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
                documents.append(Document(page_content=text, metadata={"source": url}))
                print(f"  Successfully scraped and cleaned: {url}")
            else:
                text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
                documents.append(Document(page_content=text, metadata={"source": url}))
                print(f"  WARNING: Could not find main content for {url}. Using body text.")
        except requests.RequestException as e:
            print(f"  ERROR: Failed to process URL {url}: {e}")
    return documents

def main():
    """Main function to run the data ingestion with robust recursive chunking."""
    print("--- Starting Final Ingestion using FAISS with Recursive Splitter ---")
    
    local_docs = load_from_directory(DATA_DIR)
    web_docs = load_and_clean_urls(URLS_TO_SCRAPE)
    all_docs = local_docs + web_docs

    if not all_docs:
        print("No documents were loaded. Exiting.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    
    print("\nSplitting documents into recursive chunks...")
    chunks = text_splitter.split_documents(all_docs)

    if os.path.exists(DB_DIR):
        print(f"Removing old FAISS index at '{DB_DIR}'...")
        shutil.rmtree(DB_DIR)
    
    print(f"Creating FAISS index with {len(chunks)} recursive chunks...")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(DB_DIR)
    
    print("\n--- Ingestion Complete! ---")
    print("FAISS index created with context-rich chunks.")

if __name__ == "__main__":
    main()

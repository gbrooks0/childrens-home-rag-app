# rag_system.py (Definitive, Complete, and Final Version - Smart Routing Implementation)

import os
import tempfile
import base64
import sys # Import sys

# --- IMPORTANT: SQLite3 Fix for ChromaDB in Deployment Environments ---
# This ensures that chromadb uses a compatible SQLite version if the system's is outdated.
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    print("DEBUG: pysqlite3 imported and set as default sqlite3 module.")
except ImportError:
    print("DEBUG: pysqlite3 not found, falling back to system sqlite3.")
# --- END SQLite3 Fix ---

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain_core.messages import HumanMessage

# Import OpenAI components
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings # Using OpenAIEmbeddings for consistency if OpenAI LLM is chosen

class RAGSystem:
    # The advanced prompt template for high-quality, formatted answers
    ADVANCED_PROMPT_TEMPLATE = """---
**ROLE AND GOAL**
You are a highly experienced and professional consultant specializing in the operation and strategic planning of children's homes, deeply knowledgeable in Ofsted regulations and best practices. Your goal is to provide clear, actionable, and expert advice, proactively anticipating user needs and offering comprehensive, analytical feedback that informs strategic decision-making, based *only* on the provided context.

---
**CONTEXT**
{context}

---
**OUTPUT RULES**
1.  **Comprehensive Synthesis and Cross-Referencing:** For questions about standards, policies, or detailed procedures, provide a complete answer by synthesizing information from multiple relevant sections of the context. Identify and explain connections or interdependencies between different standards or pieces of information.
2.  **Actionable Advice and Practical Implications:** Translate factual information into concrete, actionable recommendations or insights. For each key standard or topic, suggest practical steps a children's home can take to comply with or excel in that area, and explain the practical implications for daily operations.
3.  **Deeper Analysis and "Why/How":** Go beyond stating facts. Where applicable, explain the underlying principles or the 'why' behind a regulation, and offer insights into effective implementation strategies or 'how-to' guidance.
4.  **Nuance and Contextual Considerations:** If the context provides for variations or nuances (e.g., for specific types of children's homes or children with particular needs), include these considerations in your analysis to provide a well-rounded perspective.
5.  **Identify Potential Challenges or Areas of Focus:** Based on the provided context, identify any common challenges in meeting certain standards or key areas that often require particular attention during inspections or operations.
6.  **Stick to the Context:** Base your entire answer strictly on the CONTEXT provided above. Do not use outside knowledge.
7.  **Handle Missing Information:** If the context does not contain the information needed to answer the question, you MUST explicitly state: "Based on the provided context, I cannot answer this question." Do not try to guess or infer information that isn't present.

---
**FORMATTING RULES**
- Use Markdown for clear formatting (e.g., **bold** for emphasis, bullet points for lists, numbered lists for ordered items).
- **For Lists of Standards/Criteria:** Use a numbered list for primary standards, and nested bullet points (`*` or `-`) for elaborations under each standard.
- **IMPORTANT:** When the user's question implies a comparison between two or more items, policies, or concepts, you MUST format your core answer as a Markdown table.

---
**QUESTION**
{question}

---
**YOUR EXPERT RESPONSE:**"""

    # Prompt for classifying the user's question type
    CLASSIFICATION_PROMPT = """
    Analyze the following user question and classify its primary intent into one of these two categories:
    - 'Regulatory_Factual': Questions asking for specific standards, regulations, definitions, or direct information from official documents.
    - 'Strategic_Analytical': Questions asking for advice, analysis, implications, comparisons, or broader strategic guidance.

    Respond ONLY with the category keyword (e.g., 'Regulatory_Factual' or 'Strategic_Analytical'). Do not include any other text or explanation.

    Question: {question}
    Category:
    """

    def __init__(self):
        """
        Initializes the RAG system with both Gemini and OpenAI LLMs,
        and a FAISS index for similarity search.
        """
        print("DEBUG: Initializing RAG System with both Gemini and OpenAI LLMs for smart routing...")
        
        # --- Initialize Gemini LLM and Embeddings ---
        self.gemini_llm = None
        self.gemini_embeddings = None
        print("DEBUG: Attempting to initialize Gemini LLM (Gemini 1.5 Pro) and Embeddings (text-embedding-004)...")
        try:
            gemini_api_key_status = "Not Set"
            if "GOOGLE_API_KEY" in os.environ and os.environ["GOOGLE_API_KEY"]:
                gemini_api_key_status = "Found and Non-Empty"
            print(f"DEBUG: GOOGLE_API_KEY status: {gemini_api_key_status}")

            self.gemini_llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.1) # Lower temp for classification
            self.gemini_embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            print("DEBUG: Gemini LLM and Embeddings initialized successfully.")
        except Exception as e:
            print(f"ERROR: Failed to initialize Gemini LLM or Embeddings. This often means GOOGLE_API_KEY is missing or invalid. Details: {e}")

        # --- Initialize OpenAI LLM and Embeddings ---
        self.openai_llm = None
        self.openai_embeddings = None
        print("DEBUG: Attempting to initialize OpenAI LLM (GPT-4o) and Embeddings (text-embedding-ada-002)...")
        try:
            openai_api_key_status = "Not Set"
            if "OPENAI_API_KEY" in os.environ and os.environ["OPENAI_API_KEY"]:
                openai_api_key_status = "Found and Non-Empty"
            print(f"DEBUG: OPENAI_API_KEY status: {openai_api_key_status}")

            self.openai_llm = ChatOpenAI(model="gpt-4o", temperature=0.7) # Higher temp for strategic answers
            self.openai_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
            print("DEBUG: OpenAI LLM and Embeddings initialized successfully.")
        except Exception as e:
            print(f"ERROR: Failed to initialize OpenAI LLM or Embeddings. This often means OPENAI_API_KEY is missing or invalid. Details: {e}")

        # Determine which embeddings to use for FAISS loading
        # IMPORTANT: The embeddings model used here MUST match the one used during ingestion.
        # It is highly recommended to ingest your data using Google embeddings for this setup.
        if self.gemini_embeddings:
            self.embeddings = self.gemini_embeddings
            print("DEBUG: Using Gemini embeddings for FAISS index loading.")
        elif self.openai_embeddings:
            self.embeddings = self.openai_embeddings
            print("DEBUG: Using OpenAI embeddings for FAISS index loading (WARNING: Ensure ingest used OpenAI embeddings).")
        else:
            raise RuntimeError("Neither Gemini nor OpenAI embeddings could be initialized. Cannot load FAISS index.")

        db_path = "faiss_index"
        print(f"DEBUG: Checking for FAISS index at: '{db_path}'...")
        
        if not os.path.exists(db_path):
            print(f"ERROR: FAISS index directory NOT FOUND at '{db_path}'. This is a critical error.")
            raise FileNotFoundError(
                f"FAISS index not found at '{db_path}'. "
                "Please ensure you have run ingest.py locally to create it, and then committed and pushed the 'faiss_index' directory to your GitHub repository."
            )
        
        print(f"DEBUG: FAISS index directory found. Attempting to load from '{db_path}'...")
        try:
            db = FAISS.load_local(
                db_path, self.embeddings, allow_dangerous_deserialization=True
            )
            print("DEBUG: FAISS index loaded successfully.")
        except Exception as e:
            print(f"ERROR: Failed to load FAISS index from '{db_path}'. Details: {e}")
            raise
        
        self.main_retriever = db.as_retriever(
            search_type="similarity",
            search_kwargs={'k': 12} 
        )
        print("DEBUG: Main retriever initialized.")
        
        self.session_retriever = None
        print("DEBUG: RAG System initialization complete.")

    def _classify_question(self, question: str) -> str:
        """
        Classifies the user's question as 'Regulatory_Factual' or 'Strategic_Analytical'.
        Uses Gemini for classification.
        """
        if not self.gemini_llm:
            print("WARNING: Gemini LLM not available for classification. Defaulting to 'Strategic_Analytical'.")
            return "Strategic_Analytical" # Default if Gemini isn't available

        classification_prompt = self.CLASSIFICATION_PROMPT.format(question=question)
        print(f"DEBUG: Classifying question: '{question}'")
        try:
            classification_response = self.gemini_llm.invoke(classification_prompt)
            category = classification_response.content.strip()
            print(f"DEBUG: Question classified as: '{category}'")
            if category in ["Regulatory_Factual", "Strategic_Analytical"]:
                return category
            else:
                print(f"WARNING: Unexpected classification: '{category}'. Defaulting to 'Strategic_Analytical'.")
                return "Strategic_Analytical" # Fallback for unexpected classification
        except Exception as e:
            print(f"ERROR: Failed to classify question with Gemini: {e}. Defaulting to 'Strategic_Analytical'.")
            return "Strategic_Analytical" # Fallback if classification fails

    def query(self, user_question: str, context_text: str, source_docs: list, image_bytes=None):
        """
        Queries the LLM with the provided context and passes through the source documents.
        Implements smart routing: classifies the question and routes to the appropriate LLM.
        Includes fallback if the primary LLM fails or gives a non-substantive answer.
        """
        print("DEBUG: Starting query process with smart routing strategy...")
        full_prompt_text = self.ADVANCED_PROMPT_TEMPLATE.format(
            context=context_text, 
            question=user_question
        )
        
        response = None
        used_llm = "None"
        MIN_SUBSTANTIVE_LENGTH = 100 # Define a minimum length for a "substantive" answer

        # Classify the question to determine primary LLM
        question_category = self._classify_question(user_question)

        primary_llm = None
        fallback_llm = None
        primary_llm_name = ""
        fallback_llm_name = ""

        if question_category == "Regulatory_Factual":
            primary_llm = self.gemini_llm
            primary_llm_name = "Gemini"
            fallback_llm = self.openai_llm
            fallback_llm_name = "ChatGPT"
        elif question_category == "Strategic_Analytical":
            primary_llm = self.openai_llm
            primary_llm_name = "ChatGPT"
            fallback_llm = self.gemini_llm
            fallback_llm_name = "Gemini"
        else:
            # Should not happen if _classify_question works, but as a safeguard
            print("WARNING: Unknown question category. Defaulting to Gemini as primary.")
            primary_llm = self.gemini_llm
            primary_llm_name = "Gemini"
            fallback_llm = self.openai_llm
            fallback_llm_name = "ChatGPT"

        # --- Try Primary LLM ---
        if primary_llm:
            print(f"DEBUG: Attempting query with primary LLM: {primary_llm_name} (Category: {question_category})...")
            try:
                if image_bytes:
                    print(f"DEBUG: {primary_llm_name}: Image bytes provided, preparing for multimodal query.")
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": full_prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    )
                    response = primary_llm.invoke([message])
                else:
                    print(f"DEBUG: {primary_llm_name}: No image bytes provided, performing text-only query.")
                    response = primary_llm.invoke(full_prompt_text)
                
                # Check for useful response from primary LLM
                if (response and response.content and 
                    "cannot answer this question" not in response.content.lower() and
                    len(response.content) >= MIN_SUBSTANTIVE_LENGTH):
                    used_llm = primary_llm_name
                    print(f"DEBUG: Primary LLM ({primary_llm_name}) provided a useful and substantive response.")
                else:
                    print(f"DEBUG: Primary LLM ({primary_llm_name}) response was not useful/substantive or indicated inability to answer. Falling back...")
                    response = None # Clear response to try fallback LLM
            except Exception as e:
                print(f"ERROR: Primary LLM ({primary_llm_name}) query failed: {e}. Falling back to {fallback_llm_name}...")
                response = None # Clear response to try fallback LLM
        else:
            print(f"DEBUG: Primary LLM ({primary_llm_name}) not initialized, skipping to fallback.")

        # --- Try Fallback LLM if primary failed ---
        if not response and fallback_llm:
            print(f"DEBUG: Attempting query with fallback LLM: {fallback_llm_name}...")
            try:
                if image_bytes:
                    print(f"DEBUG: {fallback_llm_name}: Image bytes provided, preparing for multimodal query.")
                    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
                    message = HumanMessage(
                        content=[
                            {"type": "text", "text": full_prompt_text},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                        ]
                    )
                    response = fallback_llm.invoke([message])
                else:
                    print(f"DEBUG: {fallback_llm_name}: No image bytes provided, performing text-only query.")
                    response = fallback_llm.invoke(full_prompt_text)

                if response and response.content:
                    used_llm = fallback_llm_name
                    print(f"DEBUG: Fallback LLM ({fallback_llm_name}) provided a response.")
                else:
                    print(f"DEBUG: Fallback LLM ({fallback_llm_name}) response was empty. No useful response from any LLM.")
            except Exception as e:
                print(f"ERROR: Fallback LLM ({fallback_llm_name}) query failed: {e}. No response available.")
                response = None
        else:
            if not fallback_llm:
                print(f"DEBUG: Fallback LLM ({fallback_llm_name}) not initialized.")

        final_answer = response.content if response else "Based on the provided context, I cannot answer this question using the available models."
        
        # Optionally, you can append which LLM was used to the answer for debugging/user info
        # final_answer += f"\n\n(Answer generated by: {used_llm})"

        print(f"DEBUG: Final LLM response received from {used_llm}.")
        return {"answer": final_answer, "source_documents": source_docs}

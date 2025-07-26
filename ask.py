# ask.py (Guaranteed Fix Version - Re-ranker Removed)

import os
import datetime
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
# --- CHANGE 1: REMOVED CohereRerank and ContextualCompressionRetriever ---
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Import the RAGSystem class from rag_system.py
from rag_system import RAGSystem

# --- Configuration ---
# --- CHANGE 2: REMOVED Cohere API key check ---
if "GOOGLE_API_KEY" not in os.environ:
    print("Google API Key not found. Please set the GOOGLE_API_KEY environment variable.")
    exit()

PERSIST_DIRECTORY = "db"
REPORTS_DIRECTORY = "reports"
os.makedirs(REPORTS_DIRECTORY, exist_ok=True)

# The prompt template updated for more detailed responses
BLENDED_PROMPT_TEMPLATE = """
You are a professional, creative, and highly knowledgeable consultant assisting in the strategic planning for a new children's home.
Your tone should be clear, structured, and supportive. You are encouraged to be expansive in your answers.
Use the specific context provided below as your primary source of truth to inform your answer.
Then, leverage your extensive general knowledge to elaborate deeply, provide creative and practical examples, brainstorm innovative ideas, and anticipate potential challenges, all while staying consistent with the provided context.
**Instead of bullet points, please provide detailed explanations and structured paragraphs.** Ensure your responses are comprehensive and offer actionable insights.
Format your answer clearly with headings and subheadings where appropriate to enhance readability.

Context:
{context}

Question:
{question}
"""

# Initialize the RAG system
rag_system = RAGSystem()

def get_answer(rag_system_instance: RAGSystem, question: str):
    """
    Retrieves answer and source documents from the RAG system.
    """
    # Create the prompt with the question (context will be filled by the chain)
    # The ConversationalRetrievalChain will handle getting context and combining it with history
    result = rag_system_instance.query(question)
    return result

def main():
    print("Welcome to Lumen Way Homes AI Assistant!")
    print("Type 'quit' to exit.")
    print("Type 'clear' to clear conversation history.")
    print("Type 'upload' to upload a document for temporary session-specific retrieval.")

    full_report_content = []

    while True:
        query = input("\nEnter your question: ")
        if query.lower() == 'quit':
            break
        elif query.lower() == 'clear':
            rag_system.clear_memory()
            rag_system.update_retriever(session_retriever=None) # Reset retriever if cleared
            print("Conversation history cleared and session documents removed.")
            continue
        elif query.lower() == 'upload':
            file_path = input("Enter the path to the PDF file you want to upload: ")
            try:
                # Read file content as base64
                with open(file_path, 'rb') as f:
                    file_content = base64.b64encode(f.read()).decode('utf-8')
                
                # Load documents to a session-specific retriever
                session_retriever = rag_system.load_documents_to_session_retriever(file_content)
                rag_system.update_retriever(session_retriever=session_retriever)
                print(f"Document '{os.path.basename(file_path)}' uploaded and integrated for this session.")
            except FileNotFoundError:
                print(f"Error: File not found at '{file_path}'")
            except Exception as e:
                print(f"Error uploading document: {e}")
            continue

        try:
            # Get answer and sources
            response = get_answer(rag_system, query)
            answer = response.get("answer", "No answer found.")
            source_documents = response.get("source_documents", [])

            print("\n" + "="*50)
            print("RESPONSE:")
            print(answer)
            print("\nSOURCES CONSULTED:")
            
            unique_sources = set()
            for source in source_documents:
                source_name = source.metadata.get('source', 'Unknown Source')
                if source_name.startswith('http'):
                    unique_sources.add(f"- URL: {source_name}")
                else:
                    page_num = source.metadata.get('page', -1)
                    page_display = f", Page: {page_num + 1}" if page_num != -1 else ""
                    source_text = f"- File: {os.path.basename(source_name)}{page_display}"
                unique_sources.add(source_text)
            
            sources_text_list = sorted(list(unique_sources))
            for source_text in sources_text_list:
                print(source_text)

            print("="*50)

            full_report_content.append(f"## QUESTION\n{query}\n\n")
            full_report_content.append(f"## RESPONSE\n{answer}\n\n")
            full_report_content.append(f"### SOURCES CONSULTED\n" + "\n".join(sources_text_list) + "\n\n")
            full_report_content.append("---\n\n")
        
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.")

    # --- Saving the session report (unchanged) ---
    if len(full_report_content) > 1:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        filename = f"report_{timestamp}.md"
        filepath = os.path.join(REPORTS_DIRECTORY, filename)
        print(f"\nExiting... Saving conversation to: {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(full_report_content)
        print(f"Session report saved to: {filepath}")
    else:
        print("\nExiting without saving a report (no substantial conversation).")

if __name__ == "__main__":
    main()

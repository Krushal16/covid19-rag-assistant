# Databricks notebook source
# DBTITLE 1,Install dependencies
# MAGIC %pip install numpy langchain langchain-community langchain-core mlflow pandas faiss-cpu sentence-transformers databricks-langchain
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,📦 Import Libraries & Setup MLflow
# ============================================================================
# IMPORTS & MLFLOW CONFIGURATION
# ============================================================================
# Core libraries for data handling and timing
import time
import json
import pandas as pd

# FAISS: Vector similarity search for semantic retrieval
import faiss

# MLflow: Experiment tracking and model management
import mlflow

# Sentence Transformers: Generate embeddings for semantic search
from sentence_transformers import SentenceTransformer

# LangChain: Framework for building LLM applications
from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain import ChatDatabricks

# Configure MLflow to track experiments
mlflow.langchain.autolog()  # Auto-log LangChain interactions
mlflow.set_tracking_uri("databricks")  # Use Databricks tracking server
mlflow.set_experiment("/Shared/covid_rag_experiments")  # Shared experiment location

print("✅ Libraries imported and MLflow configured")

# COMMAND ----------

# DBTITLE 1,Load FAISS index and chunk metadata
# ============================================================================
# LOAD VECTOR STORE & DOCUMENT CHUNKS
# ============================================================================
# Load the embedding model (must match the model used during indexing)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Load pre-built FAISS index containing 384-dimensional embeddings
index = faiss.read_index("/Volumes/main/covid_rag/covid_docs/faiss_index.bin")

# Load chunk metadata (source file, page number, chunk text) from Unity Catalog
chunk_df = spark.table("main.covid_rag.covid_chunk_embeddings").toPandas()

print("FAISS index loaded.")
print(f"Chunk metadata shape: {chunk_df.shape}")
print(f"  • {len(chunk_df)} document chunks indexed")

# COMMAND ----------

# DBTITLE 1,Initialize LLM
# ============================================================================
# INITIALIZE LARGE LANGUAGE MODEL
# ============================================================================
# Configure Databricks Foundation Model endpoint
llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",  # Meta Llama 3.3 70B Instruct
    temperature=0.1,  # Low temperature for focused, deterministic responses
    max_tokens=400    # Limit response length to keep answers concise
)

print("✅ LLM initialized: databricks-meta-llama-3-3-70b-instruct")

# COMMAND ----------

# DBTITLE 1,🔍 Vector Search Function
# ============================================================================
# RETRIEVE RELEVANT CHUNKS USING VECTOR SIMILARITY
# ============================================================================
def retrieve_chunks(query, top_k=4):
    """
    Search FAISS index for most semantically similar document chunks.
    
    Args:
        query (str): User's question/search query
        top_k (int): Number of top results to return (default: 4)
    
    Returns:
        pandas.DataFrame: Top-k chunks with metadata and similarity distances
                          Columns: source_file, page_num, chunk_text, distance
    """
    # Encode query into 384-dimensional vector
    q_emb = embed_model.encode([query]).astype("float32")
    
    # Search FAISS index for k-nearest neighbors
    distances, indices = index.search(q_emb, top_k)
    
    # Retrieve metadata for matched chunks
    results = chunk_df.iloc[indices[0]][["source_file", "page_num", "chunk_text"]].copy()
    results["distance"] = distances[0]  # Lower distance = more similar
    
    return results

# COMMAND ----------

# DBTITLE 1,Cell 6
# ============================================================================
# EVALUATE RETRIEVED CONTEXT QUALITY
# ============================================================================
def internal_context_good_enough(results, query, min_rows=2, min_chars=250, max_avg_distance=1.2):
    """
    Determine if retrieved chunks are relevant enough to answer the query.
    
    Quality criteria:
      1. Minimum number of chunks (min_rows)
      2. Sufficient content length (min_chars total)
      3. Good semantic similarity (max_avg_distance threshold)
      4. Keyword overlap between query and context
    
    Args:
        results (DataFrame): Retrieved chunks with distance scores
        query (str): Original user question
        min_rows (int): Minimum required chunks (default: 2)
        min_chars (int): Minimum total characters (default: 250)
        max_avg_distance (float): Maximum average distance threshold (default: 1.2)
    
    Returns:
        bool: True if context quality is sufficient, False otherwise
    """
    # Check 1: Minimum number of results
    if results is None or len(results) < min_rows:
        return False

    # Check 2: Total content length
    total_chars = results["chunk_text"].astype(str).str.len().sum()
    if total_chars < min_chars:
        return False

    # Check 3: Semantic similarity (lower distance = more similar)
    if "distance" in results.columns:
        avg_distance = float(results["distance"].mean())
        if avg_distance > max_avg_distance:
            return False

    # Check 4: Keyword overlap between query and retrieved chunks
    query_words = set(str(query).lower().split())
    context_words = set(" ".join(results["chunk_text"].astype(str).str.lower().tolist()).split())
    overlap = len(query_words.intersection(context_words))
    if overlap == 0:
        return False

    return True

# COMMAND ----------

# DBTITLE 1,🛠️ Retriever Helper Functions
# ============================================================================
# HELPER FUNCTIONS FOR DOCUMENT RETRIEVAL
# ============================================================================
def covid_retriever_tool(question: str, top_k=4):
    """
    Wrapper function for COVID document retrieval.
    
    Args:
        question (str): User's question
        top_k (int): Number of chunks to retrieve
    
    Returns:
        DataFrame: Retrieved chunks with metadata
    """
    return retrieve_chunks(question, top_k=top_k)


def source_lookup_tool(results):
    """
    Extract unique source citations from retrieved chunks.
    
    Args:
        results (DataFrame): Retrieved chunks with metadata
    
    Returns:
        list: List of {source_file, page_num} dictionaries
    """
    if results is None or len(results) == 0:
        return []
    return results[["source_file", "page_num"]].drop_duplicates().to_dict("records")

# COMMAND ----------

# DBTITLE 1,Cell 8
# ============================================================================
# EXTERNAL KNOWLEDGE: WIKIPEDIA API
# ============================================================================
import requests
from urllib.parse import quote
import re

def fetch_wikipedia_summary(query):
    """
    Fetch Wikipedia summary for general knowledge questions.
    
    Features:
      - Cleans query (removes "what is", "who is", trailing punctuation)
      - Handles 404 errors by retrying without articles ("the", "a", "an")
      - Returns structured result with text and source URL
    
    Args:
        query (str): Search query
    
    Returns:
        dict: {"text": summary_text, "sources": [{title, url, type}]}
              Empty text if no results found
    """
    # Clean query: Remove question prefixes (case-insensitive)
    search_term = re.sub(r'^(what is|who is|what are|who are|tell me about)\s+', '', 
                         query.strip(), flags=re.IGNORECASE)
    # Remove trailing punctuation
    search_term = search_term.rstrip('?!.')
    
    if not search_term:
        search_term = query.strip()

    # Wikipedia REST API endpoint
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(search_term)}"
    headers = {"User-Agent": "covid-rag-project/1.0"}  # Required by Wikipedia
    
    try:
        r = requests.get(url, headers=headers, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract", "").strip()
            page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
            title = data.get("title", search_term)
            
            if extract:
                return {
                    "text": extract,
                    "sources": [{
                        "title": title,
                        "url": page_url,
                        "type": "wikipedia"
                    }]
                }
                
        elif r.status_code == 404:
            # Retry without leading articles ("the", "a", "an")
            if search_term.lower().startswith(('the ', 'a ', 'an ')):
                search_term = re.sub(r'^(the|a|an)\s+', '', search_term, flags=re.IGNORECASE)
                return fetch_wikipedia_summary(search_term)  # Recursive retry
                
    except Exception:
        pass  # Silently fail and return empty result

    return {"text": "", "sources": []}

# COMMAND ----------

# DBTITLE 1,🎥 YouTube Search Link
# ============================================================================
# EXTERNAL KNOWLEDGE: YOUTUBE SEARCH LINKS
# ============================================================================
from urllib.parse import quote_plus

def build_youtube_search_source(query):
    """
    Generate YouTube search URL as an additional resource.
    
    Args:
        query (str): Search query
    
    Returns:
        dict: YouTube search link metadata
    """
    return {
        "title": f"YouTube search: {query}",
        "url": f"https://www.youtube.com/results?search_query={quote_plus(query)}",
        "type": "youtube"
    }

# COMMAND ----------

# DBTITLE 1,🌐 External Knowledge Function
# ============================================================================
# ANSWER FROM EXTERNAL SOURCES (WIKIPEDIA + YOUTUBE)
# ============================================================================
def answer_from_external(question, llm, chat_history=""):
    """
    Generate answer using external knowledge sources.
    
    Process:
      1. Fetch Wikipedia summary
      2. Add YouTube search link as extra resource
      3. Call LLM with external context + chat history
    
    Note: LLM is always called, even without Wikipedia results,
          so it can use chat history to answer conversational questions.
    
    Args:
        question (str): User's question
        llm: ChatDatabricks LLM instance
        chat_history (str): Formatted chat history
    
    Returns:
        tuple: (answer_text, sources_list)
    """
    # Fetch Wikipedia information
    wiki_result = fetch_wikipedia_summary(question)

    external_chunks = []
    external_sources = []

    # Add Wikipedia content if available
    if wiki_result["text"]:
        external_chunks.append(f"Wikipedia:\n{wiki_result['text']}")
        external_sources.extend(wiki_result["sources"])

    # Add YouTube search link as extra resource
    external_sources.append(build_youtube_search_source(question))

    # Prepare context (may be empty if no Wikipedia results)
    context = "\n\n".join(external_chunks) if external_chunks else "No external information found."

    # Call LLM with external context + chat history
    # (LLM can answer using chat history even without external context)
    prompt_value = external_prompt_template.invoke({
        "chat_history": chat_history,
        "context": context,
        "question": question
    })

    response = llm.invoke(prompt_value)
    answer = response.content if hasattr(response, "content") else str(response)

    return answer, external_sources

# COMMAND ----------

# DBTITLE 1,📝 Context Builder
# ============================================================================
# FORMAT RETRIEVED CHUNKS FOR LLM CONTEXT
# ============================================================================
def build_context(results):
    """
    Format retrieved chunks into readable context with source citations.
    
    Args:
        results (DataFrame): Retrieved chunks with metadata
    
    Returns:
        str: Formatted context string with sources
             Format: "Source: file | Page: N\nchunk_text\n\n..."
    """
    context = []
    for _, row in results.iterrows():
        context.append(
            f"Source: {row['source_file']} | Page: {row['page_num']}\n{row['chunk_text']}"
        )
    return "\n\n".join(context)

# COMMAND ----------

# DBTITLE 1,Cell 12
# ============================================================================
# PROMPT TEMPLATE FOR INTERNAL COVID DOCUMENTS
# ============================================================================
# Template for answering from COVID-19 PDF documents
internal_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant for COVID-19 documents.

Answer the user's question in a natural, conversational, and descriptive way.
Return only the answer itself.
Do not use numbered points, labels, headings, source notes, confidence notes, or extra formatting.
If the context does not contain enough relevant information, say that briefly and plainly.
Do not make up information.

If there is chat history, use it to understand context and references (like "my last question", "that", etc.).
"""
    ),
    (
        "human",
        """Chat history:
{chat_history}

Context:
{context}

Question:
{question}

Answer:"""
    )
])

# COMMAND ----------

# DBTITLE 1,🌐 External Prompt Template
# ============================================================================
# PROMPT TEMPLATE FOR EXTERNAL SOURCES (WIKIPEDIA)
# ============================================================================
# Template for answering from external web information
external_prompt_template = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a COVID-19 assistant specializing in COVID-19 information, but you can answer general questions too.

Answer in plain natural language.
Return only the answer text.
Do not include labels, source notes, numbered sections, or confidence notes.

If there is chat history, check if you can answer the question using ONLY the chat history 
(for example, questions like "what was my last question?", "tell me more about that", etc.).
If the question can be answered from chat history alone, do so - don't mention lack of external information.

If external information is needed and provided, use it to answer.
If external information is needed but not provided, say briefly that you couldn't find reliable information.

Do not make up information.
"""
    ),
    (
        "human",
        """Chat history:
{chat_history}

External information:
{context}

Question:
{question}

Answer:"""
    )
])

# COMMAND ----------

# DBTITLE 1,Chat history formatter
# ============================================================================
# FORMAT CHAT HISTORY FOR CONVERSATIONAL CONTEXT
# ============================================================================
def get_chat_history_text(messages, max_turns=6):
    """
    Convert message list to formatted chat history text.
    
    Args:
        messages (list): List of {role, content} message dicts
        max_turns (int): Maximum number of exchanges to include (default: 6)
    
    Returns:
        str: Formatted chat history ("User: ...\nAssistant: ...")
    """
    if not messages:
        return ""
    
    # Keep only recent exchanges to avoid context overflow
    recent_messages = messages[-(max_turns * 2):] if len(messages) > max_turns * 2 else messages
    
    history_lines = []
    for msg in recent_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            history_lines.append(f"User: {content}")
        elif role == "assistant":
            history_lines.append(f"Assistant: {content}")
    
    return "\n".join(history_lines)

# COMMAND ----------

# DBTITLE 1,Greeting Detection
# ============================================================================
# GREETING DETECTION & RESPONSE
# ============================================================================
def is_greeting(question):
    """
    Check if the question is a greeting or casual conversation.
    
    Args:
        question (str): User's input
    
    Returns:
        bool: True if greeting detected, False otherwise
    """
    question_lower = question.strip().lower()
    
    # Common greetings
    greetings = [
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon", 
        "good evening", "howdy", "what's up", "whats up", "sup"
    ]
    
    # Check for exact match or starts with greeting
    for greeting in greetings:
        if question_lower == greeting or question_lower.startswith(greeting + " "):
            return True
    
    # Check for "how are you" variations
    if any(phrase in question_lower for phrase in 
           ["how are you", "how r u", "how do you do", "how's it going", "hows it going"]):
        return True
    
    return False


def get_greeting_response():
    """
    Return a friendly greeting from the COVID-19 assistant.
    
    Returns:
        str: Greeting message with assistant capabilities
    """
    return ("Hello! I'm your COVID-19 assistant. I can help you with information about COVID-19, "
            "including symptoms, transmission, prevention, vaccines, and treatment. "
            "You can also ask me general questions, and I'll do my best to help. What would you like to know?")

# COMMAND ----------

# DBTITLE 1,Cell 16
# ============================================================================
# MAIN QUERY FUNCTION: ORCHESTRATES THE ENTIRE RAG PIPELINE
# ============================================================================
import time
from requests.exceptions import RequestException

def answer_query(question, top_k=4, messages=None, max_retries=2):
    """
    Answer a query using RAG (Retrieval-Augmented Generation) with routing logic.
    
    Routing Logic:
      1. GREETING: Fast response for greetings (no LLM/retrieval)
      2. INTERNAL: Retrieves from COVID PDFs when context quality is good
      3. EXTERNAL: Uses Wikipedia when internal context insufficient
      4. ERROR: Returns error details with retry logic
    
    Args:
        question (str): User's question
        top_k (int): Number of chunks to retrieve (default: 4)
        messages (list): Chat history for conversational context
        max_retries (int): Number of retry attempts for LLM/API calls (default: 2)
    
    Returns:
        dict: {
            "answer": str,                  # Generated answer
            "route_used": str,              # "greeting", "internal", "external", or "error"
            "source_type": str,             # Source type identifier
            "sources": list,                # List of source citations
            "retrieved_chunks": list        # Retrieved document chunks (if internal)
        }
    """
    if messages is None:
        messages = []
    
    # ===== ROUTE 1: GREETING DETECTION =====
    # Fast path for greetings - no LLM/retrieval needed
    if is_greeting(question):
        greeting_answer = get_greeting_response()
        return {
            "answer": greeting_answer,
            "route_used": "greeting",
            "source_type": "greeting",
            "sources": [],
            "retrieved_chunks": []
        }
    
    # Format chat history for conversational context
    chat_history = get_chat_history_text(messages, max_turns=6)
    
    # ===== RETRIEVAL PHASE =====
    # Retrieve relevant chunks from COVID documents
    try:
        internal_results = retrieve_chunks(question, top_k=top_k)
    except Exception as e:
        return {
            "answer": f"Error retrieving internal documents: {str(e)}",
            "route_used": "error",
            "source_type": "error",
            "sources": [],
            "retrieved_chunks": []
        }

    # ===== ROUTE 2: INTERNAL (COVID DOCUMENTS) =====
    # Use internal documents if context quality is sufficient
    if internal_context_good_enough(internal_results, question):
        route_used = "internal"
        source_type = "internal"
        context = build_context(internal_results)
        sources = source_lookup_tool(internal_results)

        # Retry logic for LLM calls (handles transient connection issues)
        for attempt in range(max_retries + 1):
            try:
                prompt_value = internal_prompt_template.invoke({
                    "chat_history": chat_history,
                    "context": context,
                    "question": question
                })

                response = llm.invoke(prompt_value)
                answer = response.content if hasattr(response, "content") else str(response)
                break  # Success - exit retry loop
                
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1)  # Wait 1 second before retry
                    continue
                else:
                    # Final attempt failed - return error
                    return {
                        "answer": f"LLM connection error after {max_retries + 1} attempts. Please try again. Error: {str(e)}",
                        "route_used": "error",
                        "source_type": "internal",
                        "sources": sources,
                        "retrieved_chunks": internal_results.to_dict(orient="records")
                    }

    # ===== ROUTE 3: EXTERNAL (WIKIPEDIA) =====
    # Use external sources when internal context is insufficient
    else:
        route_used = "external"
        source_type = "external"
        
        # Retry logic for external API calls
        for attempt in range(max_retries + 1):
            try:
                answer, sources = answer_from_external(question, llm, chat_history)
                break  # Success - exit retry loop
                
            except RequestException as e:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                else:
                    return {
                        "answer": f"External API connection error after {max_retries + 1} attempts. Please check your internet connection and try again. Error: {str(e)}",
                        "route_used": "error",
                        "source_type": "external",
                        "sources": [],
                        "retrieved_chunks": []
                    }
            except Exception as e:
                if attempt < max_retries:
                    time.sleep(1)
                    continue
                else:
                    return {
                        "answer": f"Error getting external answer after {max_retries + 1} attempts: {str(e)}",
                        "route_used": "error",
                        "source_type": "external",
                        "sources": [],
                        "retrieved_chunks": []
                    }

    # Return final structured response
    return {
        "answer": answer,
        "route_used": route_used,
        "source_type": source_type,
        "sources": sources,
        "retrieved_chunks": internal_results.to_dict(orient="records") if internal_results is not None else []
    }

# COMMAND ----------

# DBTITLE 1,How to Use the Chatbot
# MAGIC %md
# MAGIC # 🤖 COVID-19 RAG Chatbot
# MAGIC
# MAGIC ## ✅ Using the Chatbot
# MAGIC
# MAGIC Run the cell below to initialize the chat interface, then use these commands in **any new cell**:
# MAGIC
# MAGIC ```python
# MAGIC # Ask questions
# MAGIC chat("What is COVID-19?")
# MAGIC chat("What are the symptoms?")  # Maintains conversation history
# MAGIC chat("hello")                   # Try a greeting!
# MAGIC
# MAGIC # Manage history
# MAGIC show_history()                  # View conversation
# MAGIC clear_history()                 # Start fresh
# MAGIC ```
# MAGIC
# MAGIC ## 🎯 Features
# MAGIC
# MAGIC * ✅ **Greetings** - Says hello and introduces itself
# MAGIC * ✅ **COVID Questions** - Retrieves from your PDF documents
# MAGIC * ✅ **General Questions** - Uses Wikipedia for non-COVID topics
# MAGIC * ✅ **Conversation History** - Understands context ("What was my last question?")
# MAGIC * ✅ **Error Handling** - Automatic retries for connection issues
# MAGIC
# MAGIC ## 📝 Routes
# MAGIC
# MAGIC * **greeting** - Friendly introduction
# MAGIC * **internal** - COVID documents (with sources)
# MAGIC * **external** - Wikipedia + YouTube
# MAGIC * **error** - Connection/retrieval failures
# MAGIC
# MAGIC ---

# COMMAND ----------

# DBTITLE 1,Cell 23
# ============================================================================
# SIMPLE CHAT INTERFACE FOR DATABRICKS NOTEBOOK
# ============================================================================
# Provides interactive chat functions with history management
# (Gradio has connection issues on serverless clusters, so using simple functions)

chat_messages = []

def chat(question):
    """
    Send a message and get a response with conversation history.
    
    The chat maintains history across multiple calls, enabling:
    - Contextual follow-ups ("what about that?", "tell me more")
    - Reference to previous questions ("what was my last question?")
    - Multi-turn conversations
    
    Args:
        question (str): User's question or message
    
    Returns:
        dict: Full result from answer_query including:
            - answer: Generated response
            - route_used: "greeting", "internal", "external", or "error"
            - sources: List of source citations
            - retrieved_chunks: Document chunks used (if internal)
    """
    global chat_messages
    
    # Call main RAG function with conversation history
    result = answer_query(question, top_k=4, messages=chat_messages)
    
    # Append to history for future context
    chat_messages.append({"role": "user", "content": question})
    chat_messages.append({"role": "assistant", "content": result["answer"]})
    
    # Display formatted result
    print(f"\n🤖 Route: {result['route_used']}")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources: {json.dumps(result['sources'], indent=2)}")
    
    return result


def clear_history():
    """
    Clear conversation history.
    
    Use this to start a fresh conversation without previous context.
    Useful when switching topics or testing different scenarios.
    """
    global chat_messages
    chat_messages = []
    print("✅ Chat history cleared!")


def show_history():
    """
    Display current conversation history.
    
    Shows all user questions and assistant responses in the current session.
    Useful for debugging or reviewing what the chatbot remembers.
    """
    global chat_messages
    if not chat_messages:
        print("No chat history yet.")
    else:
        print(f"\n=== Chat History ({len(chat_messages) // 2} exchanges) ===")
        for msg in chat_messages:
            role = msg['role'].upper()
            # Truncate long messages for readability
            content = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
            print(f"  {role}: {content}")

# Display usage instructions
print("\n✅ COVID RAG Chatbot Ready!")
print("="*60)
print("\n💬 Usage:")
print("  chat('Your question')       - Ask a question")
print("  show_history()              - View chat history")
print("  clear_history()             - Clear chat history")
print("\n💡 Examples:")
print("  chat('What is COVID-19?')")
print("  chat('What are the symptoms?')")
print("  chat('Who is Albert Einstein?')")
print("="*60)

# COMMAND ----------

# DBTITLE 1,Demo: Chat Interface
# Test chat history integration
print("Testing chat with conversational references...\n")

chat("What is COVID-19?")
print("\n" + "="*60 + "\n")

chat("What are the symptoms?")
print("\n" + "="*60 + "\n")

chat("What was my last question?")
print("\n" + "="*60 + "\n")

print("\nChat history:")
show_history()

# COMMAND ----------

chat("hello!! how r u today?")

# COMMAND ----------

chat("what is coronavirus")
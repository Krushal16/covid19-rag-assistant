# Databricks notebook source
# DBTITLE 1,Install required packages
# Install dependencies for chunking, embedding, and indexing
# - langchain-text-splitters: For splitting long documents into smaller chunks
# - mlflow: For experiment tracking
# - sentence-transformers: For generating embeddings from text
# - faiss-cpu: For building similarity search index
%pip install langchain-text-splitters mlflow pandas sentence-transformers faiss-cpu

# Restart Python to load newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Chunk Text from Pages
# ============================================================================
# CHUNK TEXT INTO SMALLER PIECES
# ============================================================================
import mlflow
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set MLflow experiment for tracking
mlflow.set_experiment("/Shared/covid_rag_experiments")

# Load the extracted PDF pages from Unity Catalog
pdf_df = spark.table("main.covid_rag.covid_pages").toPandas()

# Define chunking parameters
# chunk_size: Maximum characters per chunk (800 chars ≈ 150-200 words)
# chunk_overlap: Overlap between chunks to maintain context continuity
chunk_size = 800
chunk_overlap = 150

# Initialize the text splitter
# RecursiveCharacterTextSplitter tries to split on natural boundaries (paragraphs, sentences)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# Split each page into chunks
chunk_rows = []
for _, row in pdf_df.iterrows():
    # Split the page text into chunks
    chunks = splitter.split_text(row["text"])
    
    # Create a record for each chunk with metadata
    for i, chunk in enumerate(chunks):
        chunk_rows.append({
            "source_file": row["source_file"],  # Original PDF filename
            "page_num": row["page_num"],        # Page number in PDF
            "chunk_id": f"{row['source_file']}_p{row['page_num']}_c{i}",  # Unique chunk identifier
            "chunk_text": chunk,                # The actual text content
            "source_path": row["source_path"]  # Full path to source PDF
        })

# Create DataFrame from chunks and save to Unity Catalog
chunk_df = pd.DataFrame(chunk_rows)
spark.createDataFrame(chunk_df).write.mode("overwrite").saveAsTable("main.covid_rag.covid_chunks")

# COMMAND ----------

# DBTITLE 1,Generate Embeddings
# ============================================================================
# GENERATE EMBEDDINGS FOR EACH CHUNK
# ============================================================================
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the chunks from Unity Catalog
chunk_df = spark.table("main.covid_rag.covid_chunks").toPandas()

# Initialize the embedding model
# all-MiniLM-L6-v2: Fast, efficient model that creates 384-dimensional vectors
# Good balance between speed and semantic quality for retrieval tasks
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate embeddings for all chunks
# Embeddings are dense vector representations that capture semantic meaning
# Similar chunks will have similar embeddings (measured by cosine similarity or L2 distance)
embeddings = embed_model.encode(
    chunk_df["chunk_text"].tolist(),
    show_progress_bar=True  # Display progress during encoding
)

# Add embeddings as a new column (convert numpy arrays to lists for DataFrame compatibility)
chunk_df["embedding"] = [vec.tolist() for vec in embeddings]

# COMMAND ----------

# DBTITLE 1,Save Embeddings to Table
# ============================================================================
# SAVE EMBEDDINGS TO UNITY CATALOG
# ============================================================================
# Save the chunks with their embeddings to Unity Catalog
# This table contains: chunk text, metadata, and embedding vectors
# Will be used for retrieval in the RAG system
spark.createDataFrame(chunk_df).write.mode("overwrite").saveAsTable("main.covid_rag.covid_chunk_embeddings")

# COMMAND ----------

# DBTITLE 1,Build and save FAISS index
# ============================================================================
# BUILD FAISS INDEX FOR FAST SIMILARITY SEARCH
# ============================================================================
import faiss
import numpy as np
import pickle

# Load the embeddings from Unity Catalog
chunk_df = spark.table("main.covid_rag.covid_chunk_embeddings").toPandas()

# Convert embeddings to numpy array (FAISS requires float32)
X = np.array(chunk_df["embedding"].tolist(), dtype="float32")

# Create FAISS index
# IndexFlatL2: Exact L2 (Euclidean) distance search - no approximation
# X.shape[1] is the embedding dimension (384 for all-MiniLM-L6-v2)
index = faiss.IndexFlatL2(X.shape[1])

# Add all embeddings to the index
index.add(X)

# Save the FAISS index to Unity Catalog Volume
# This file will be loaded by the RAG application for fast retrieval
faiss.write_index(index, "/Volumes/main/covid_rag/covid_docs/faiss_index.bin")

# Save the chunk metadata (text, source info) as pickle file
# Used to retrieve actual chunk text and sources after FAISS returns indices
chunk_df.to_pickle("/Volumes/main/covid_rag/covid_docs/chunk_metadata.pkl")
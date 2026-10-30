# Databricks notebook source
# DBTITLE 1,Install required packages
# Install dependencies for PDF processing
# - pypdf: For reading and extracting text from PDF files
# - pandas: For data manipulation and DataFrame creation
%pip install pypdf pandas

# Restart Python to load newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Ingest PDFs from Volume
# ============================================================================
# PDF INGESTION FROM UNITY CATALOG VOLUME
# ============================================================================
import os
import pandas as pd
from pypdf import PdfReader

# Define the Volume path containing COVID-19 PDF documents
volume_path = "/Volumes/main/covid_rag/covid_docs/"

# List all PDF files in the Volume
files = [f.path for f in dbutils.fs.ls(volume_path) if f.path.endswith(".pdf")]

# Extract text from each page of each PDF
rows = []
for file_path in files:
    # Convert DBFS path to local path for PyPDF
    local_path = file_path.replace("dbfs:", "")
    reader = PdfReader(local_path)
    
    # Iterate through all pages in the PDF
    for page_num, page in enumerate(reader.pages, start=1):
        # Extract text from the page
        text = page.extract_text() or ""
        
        # Store page metadata and text
        rows.append({
            "source_file": os.path.basename(file_path),  # Just the filename
            "page_num": page_num,
            "text": text.strip(),
            "source_path": file_path  # Full path for reference
        })

# Create DataFrame from extracted pages
df_pages = pd.DataFrame(rows)

# Filter out pages with very little text (likely empty or image-only pages)
# Keep only pages with more than 50 characters
df_pages = df_pages[df_pages["text"].str.len() > 50]

# Display sample of extracted pages
display(df_pages.head())

# COMMAND ----------

# DBTITLE 1,Save to Unity Catalog Table
# ============================================================================
# SAVE TO UNITY CATALOG TABLE
# ============================================================================
# Convert pandas DataFrame to Spark DataFrame and save to Unity Catalog
# This makes the data available for downstream chunking and embedding steps
spark.createDataFrame(df_pages).write.mode("overwrite").saveAsTable("main.covid_rag.covid_pages")
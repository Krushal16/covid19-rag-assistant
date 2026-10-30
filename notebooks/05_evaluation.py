# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Install required packages
# Install dependencies for evaluation and experiment tracking
# - mlflow: For experiment tracking, logging metrics, and artifacts
# - pandas: For data manipulation and results analysis
%pip install mlflow pandas

# Restart Python to load newly installed packages
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load RAG Chain
# MAGIC %run ./04_rag_chain

# COMMAND ----------

# DBTITLE 1,Setup MLflow Tracking
# ============================================================================
# MLFLOW SETUP FOR EXPERIMENT TRACKING
# ============================================================================
import time
import json
import mlflow
import pandas as pd

# Configure MLflow to use Databricks tracking server
mlflow.set_tracking_uri("databricks")

# Set the experiment where evaluation runs will be logged
# All metrics, parameters, and artifacts will be stored here
mlflow.set_experiment("/Shared/covid_rag_experiments")

# COMMAND ----------

# DBTITLE 1,Verification
# ============================================================================
# VERIFY RAG FUNCTIONS ARE LOADED
# ============================================================================
# Ensure answer_query() function is available from the RAG chain notebook
# If not, Cell 2 (%run ./04_rag_chain) needs to be executed first
print("Notebook 5 started.")
print("Make sure answer_query(question, top_k=4) is available before running evaluation.")

# COMMAND ----------

# DBTITLE 1,Define Evaluation Questions
# ============================================================================
# EVALUATION DATASET
# ============================================================================
# Define test questions covering different scenarios and question types
# 
# Fields:
#   - question_id: Unique identifier for each question
#   - question: The actual question to ask the RAG system
#   - expected_route: Which route should handle this question ("internal" or "external")
#   - question_type: Category of question (fact, definition, guidance, comparison, summary, fallback)
#
# Question categories:
#   - internal: Questions answerable from COVID-19 documents (Q1-Q9)
#   - external: Questions requiring external knowledge/fallback (Q10-Q12)

eval_questions = [
    {
        "question_id": 1,
        "question": "What are common symptoms of COVID-19?",
        "expected_route": "internal",
        "question_type": "fact"
    },
    {
        "question_id": 2,
        "question": "How does COVID-19 spread?",
        "expected_route": "internal",
        "question_type": "fact"
    },
    {
        "question_id": 3,
        "question": "What is long COVID?",
        "expected_route": "internal",
        "question_type": "definition"
    },
    {
        "question_id": 4,
        "question": "What prevention methods are recommended?",
        "expected_route": "internal",
        "question_type": "guidance"
    },
    {
        "question_id": 5,
        "question": "What is the role of vaccination in COVID-19 prevention?",
        "expected_route": "internal",
        "question_type": "guidance"
    },
    {
        "question_id": 6,
        "question": "How is severe COVID-19 different from mild infection?",
        "expected_route": "internal",
        "question_type": "comparison"
    },
    {
        "question_id": 7,
        "question": "What treatments are mentioned in the uploaded documents?",
        "expected_route": "internal",
        "question_type": "fact"
    },
    {
        "question_id": 8,
        "question": "Summarize the main public health guidance from the documents.",
        "expected_route": "internal",
        "question_type": "summary"
    },
    {
        "question_id": 9,
        "question": "What are the long-term effects discussed in the documents?",
        "expected_route": "internal",
        "question_type": "fact"
    },
    {
        "question_id": 10,
        "question": "What are the latest global COVID-19 travel restrictions?",
        "expected_route": "external",
        "question_type": "fallback"
    },
    {
        "question_id": 11,
        "question": "What are the newest 2026 global COVID-19 case counts?",
        "expected_route": "external",
        "question_type": "fallback"
    },
    {
        "question_id": 12,
        "question": "Does the database mention airline policy changes related to COVID-19?",
        "expected_route": "external",
        "question_type": "fallback"
    }
]

# Convert to DataFrame for easier manipulation and display
eval_df = pd.DataFrame(eval_questions)
display(eval_df)

# COMMAND ----------

# DBTITLE 1,Run Evaluation
# ============================================================================
# RUN EVALUATION AND LOG METRICS
# ============================================================================
results = []

# Start MLflow run to track this evaluation experiment
with mlflow.start_run(run_name="covid_rag_evaluation"):
    # Log experiment metadata
    mlflow.log_param("evaluation_set_size", len(eval_df))
    mlflow.log_param("project_domain", "COVID-19 RAG")
    mlflow.log_param("evaluation_type", "manual_rubric_plus_latency")

    # Iterate through each evaluation question
    for _, row in eval_df.iterrows():
        question_id = row["question_id"]
        question = row["question"]
        expected_route = row["expected_route"]
        question_type = row["question_type"]

        # Measure end-to-end latency (retrieval + generation)
        start_time = time.time()
        result = answer_query(question, top_k=4)
        total_latency = time.time() - start_time

        # Extract result components
        answer_text = result.get("answer", "")
        route_used = result.get("route_used", "")  # "internal" or "external"
        sources = result.get("sources", [])

        # Store detailed results for each question
        results.append({
            "question_id": question_id,
            "question": question,
            "question_type": question_type,
            "expected_route": expected_route,
            "route_used": route_used,
            "route_match": route_used == expected_route,  # Did routing work correctly?
            "answer": answer_text,
            "sources": json.dumps(sources),  # Serialize sources for storage
            "num_sources": len(sources) if isinstance(sources, list) else 0,
            "total_latency_sec": round(total_latency, 4)
        })

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)

    # Calculate aggregate metrics
    route_accuracy = results_df["route_match"].mean()  # % of questions routed correctly
    avg_latency = results_df["total_latency_sec"].mean()  # Average response time
    internal_rate = (results_df["route_used"] == "internal").mean()  # % using internal docs
    external_rate = (results_df["route_used"] == "external").mean()  # % using external fallback

    # Log metrics to MLflow for tracking and comparison
    mlflow.log_metric("route_accuracy", float(route_accuracy))
    mlflow.log_metric("avg_total_latency_sec", float(avg_latency))
    mlflow.log_metric("internal_route_rate", float(internal_rate))
    mlflow.log_metric("external_route_rate", float(external_rate))

    # Log results table to MLflow for inspection
    mlflow.log_table(results_df, artifact_file="evaluation_results.json")
    
    # Save results to Unity Catalog Volume for persistence and sharing
    import os
    volume_path = "/Volumes/main/covid_rag/covid_docs/evaluation_results.csv"
    results_df.to_csv(volume_path, index=False)
    print(f"Results saved to: {volume_path}")

# Display results for immediate inspection
display(results_df)

# COMMAND ----------

# DBTITLE 1,Manual Evaluation Template
# ============================================================================
# CREATE MANUAL EVALUATION TEMPLATE
# ============================================================================
# Add columns for human evaluation of answer quality
# These scores should be filled in manually after reviewing each answer
#
# Scoring rubric (1-5 scale):
#   - relevance_score: Does the answer address the question?
#   - grounded_score: Is the answer supported by the retrieved context?
#   - retrieval_score: Were the right documents/chunks retrieved?
#   - overall_score: Overall answer quality
#   - notes: Free-form comments about the answer

manual_eval_df = results_df.copy()
manual_eval_df["relevance_score"] = None
manual_eval_df["grounded_score"] = None
manual_eval_df["retrieval_score"] = None
manual_eval_df["overall_score"] = None
manual_eval_df["notes"] = ""

display(manual_eval_df)

# COMMAND ----------

# DBTITLE 1,Log Manual Scores (Optional)
# ============================================================================
# LOG MANUAL SCORES TO MLFLOW (OPTIONAL)
# ============================================================================
# After manually filling in scores in the previous cell:
# 1. Assign scores to each row in manual_eval_df
# 2. Uncomment the code block below
# 3. Run this cell to log average scores to MLflow
#
# This creates a nested run under the main evaluation run with quality metrics

scored_df = manual_eval_df.copy()

print("Fill manual scores first, then uncomment the logging block below.")

# scored_df["relevance_score"] = scored_df["relevance_score"].astype(float)
# scored_df["grounded_score"] = scored_df["grounded_score"].astype(float)
# scored_df["retrieval_score"] = scored_df["retrieval_score"].astype(float)
# scored_df["overall_score"] = scored_df["overall_score"].astype(float)

# if scored_df["overall_score"].notna().sum() > 0:
#     with mlflow.start_run(run_name="covid_rag_manual_scores", nested=True):
#         # Log average quality scores across all questions
#         mlflow.log_metric("avg_relevance_score", float(scored_df["relevance_score"].mean()))
#         mlflow.log_metric("avg_grounded_score", float(scored_df["grounded_score"].mean()))
#         mlflow.log_metric("avg_retrieval_score", float(scored_df["retrieval_score"].mean()))
#         mlflow.log_metric("avg_overall_score", float(scored_df["overall_score"].mean()))
#
#         # Save scored results to Volume
#         scored_path = "/dbfs/FileStore/covid_rag/manual_scored_evaluation_results.csv"
#         scored_df.to_csv(scored_path, index=False)
#         mlflow.log_artifact(scored_path)

# COMMAND ----------

# DBTITLE 1,Evaluation Summary
# ============================================================================
# DISPLAY EVALUATION SUMMARY
# ============================================================================
# Calculate and display key metrics for quick assessment:
#   - Total Questions: Number of questions evaluated
#   - Route Accuracy: % of questions correctly routed (internal vs external)
#   - Average Latency: Mean response time in seconds
#   - Internal Answers: Count of questions answered from COVID docs
#   - External Fallbacks: Count of questions using external knowledge

summary = {
    "Total Questions": len(results_df),
    "Route Accuracy": round(results_df["route_match"].mean(), 3),
    "Average Latency (sec)": round(results_df["total_latency_sec"].mean(), 3),
    "Internal Answers": int((results_df["route_used"] == "internal").sum()),
    "External Fallbacks": int((results_df["route_used"] == "external").sum())
}

summary_df = pd.DataFrame([summary])
display(summary_df)
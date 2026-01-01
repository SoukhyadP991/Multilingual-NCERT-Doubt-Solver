import time
import logging
import pandas as pd
from typing import List, Dict
from src.pipeline import RAGPipeline

# Configure logging to show only errors to keep output clean
logging.getLogger().setLevel(logging.ERROR)

def run_benchmark(questions: List[str]):
    print("Initializing System for Benchmarking...")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    results = []

    print("\n" + "="*80)
    print(f"{'BENCHMARKING STARTED':^80}")
    print("="*80 + "\n")

    for i, query in enumerate(questions, 1):
        print(f"Running Query {i}/{len(questions)}: '{query}'")
        
        # Metric: Retrieval Latency
        t0 = time.time()
        docs = pipeline.retriever.retrieve(query)
        t_retrieval = time.time() - t0
        
        doc_count = len(docs)
        
        # Metric: Generation Latency
        t1 = time.time()
        # Note: We pass the docs manually to measure just generation time
        answer = pipeline.generator.generate_answer(query, docs)
        t_generation = time.time() - t1
        
        # Metric: Total Latency
        total_latency = t_retrieval + t_generation
        
        # Metric: Output Size & Speed
        # We approximate tokens as words * 1.3 for simple estimation or just use word count
        word_count = len(answer.split())
        words_per_sec = word_count / t_generation if t_generation > 0 else 0
        
        print(f"  -> Retrieval: {t_retrieval:.2f}s ({doc_count} docs)")
        print(f"  -> Generation: {t_generation:.2f}s ({word_count} words)")
        print(f"  -> Speed: {words_per_sec:.2f} words/sec")
        print("-" * 40)

        results.append({
            "Query": query,
            "Retrieval Time (s)": round(t_retrieval, 2),
            "Generation Time (s)": round(t_generation, 2),
            "Total Latency (s)": round(total_latency, 2),
            "Docs Retrieved": doc_count,
            "Response Words": word_count,
            "Words/Sec": round(words_per_sec, 2)
        })

    # Summary
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print(f"{'BENCHMARK RESULTS':^80}")
    print("="*80)
    print(df.to_string(index=False))
    print("\n" + "="*80)
    
    # Averages
    print(f"\nAverage Retrieval Time: {df['Retrieval Time (s)'].mean():.2f} s")
    print(f"Average Generation Time: {df['Generation Time (s)'].mean():.2f} s")
    print(f"Average Words/Sec:      {df['Words/Sec'].mean():.2f}")
    print("="*80)

if __name__ == "__main__":
    test_queries = [
        "What is photosynthesis?",
        "How do plants get water?",
        "What is the function of stomata?",
        "Define chlorophyll.",
        "What are the products of photosynthesis?"
    ]
    
    run_benchmark(test_queries)

import time
import logging
import pandas as pd
from typing import List
from src.pipeline import RAGPipeline

# Configure logging
logging.getLogger().setLevel(logging.ERROR)

def run_benchmark_50(questions: List[str]):
    print("Initializing System for Extensive Benchmarking (50 Questions)...")
    try:
        pipeline = RAGPipeline()
    except Exception as e:
        print(f"Failed to initialize pipeline: {e}")
        return

    results = []
    total_questions = len(questions)

    print("\n" + "="*80)
    print(f"{'BENCHMARK_50 STARTED':^80}")
    print(f"{'Estimated Runtime: ~30-40 Minutes':^80}")
    print("="*80 + "\n")

    start_global = time.time()

    for i, query in enumerate(questions, 1):
        print(f"[{i}/{total_questions}] Query: '{query}'")
        
        # Metric: Retrieval
        t0 = time.time()
        try:
            docs = pipeline.retriever.retrieve(query)
            t_retrieval = time.time() - t0
            doc_count = len(docs)
        except Exception as e:
            print(f"  Error in retrieval: {e}")
            t_retrieval = 0
            doc_count = 0
            docs = []

        # Metric: Generation
        t1 = time.time()
        try:
            # Pass retrieved docs to generator
            if docs:
                answer = pipeline.generator.generate_answer(query, docs)
            else:
                answer = "No context found."
            t_generation = time.time() - t1
        except Exception as e:
            print(f"  Error in generation: {e}")
            answer = "Generation Failed"
            t_generation = 0
        
        # Metrics
        word_count = len(answer.split())
        words_per_sec = word_count / t_generation if t_generation > 0 else 0
        
        print(f"  -> Ret: {t_retrieval:.2f}s | Gen: {t_generation:.2f}s | Spd: {words_per_sec:.2f} w/s")
        print("-" * 40)

        results.append({
            "Query": query,
            "Subject": "General", # Could classify if needed, but keeping simple
            "Retrieval Time (s)": round(t_retrieval, 2),
            "Generation Time (s)": round(t_generation, 2),
            "Total Latency (s)": round(t_retrieval + t_generation, 2),
            "Docs Retrieved": doc_count,
            "Response Words": word_count,
            "Words/Sec": round(words_per_sec, 2),
            "Answer Preview": answer[:50] + "..." if len(answer) > 50 else answer
        })

    total_time = time.time() - start_global
    
    # Save results
    df = pd.DataFrame(results)
    csv_filename = "benchmark_50_results.csv"
    df.to_csv(csv_filename, index=False)
    
    print("\n" + "="*80)
    print(f"{'BENCHMARK COMPLETE':^80}")
    print(f"Total Runtime: {total_time/60:.2f} minutes")
    print(f"Results saved to: {csv_filename}")
    print("="*80)
    
    # Summary Stats
    print(f"\nAverage Retrieval Time: {df['Retrieval Time (s)'].mean():.2f} s")
    print(f"Average Generation Time: {df['Generation Time (s)'].mean():.2f} s")
    print(f"Average Words/Sec:      {df['Words/Sec'].mean():.2f}")
    print(f"Total Successful Queries: {len(df[df['Response Words'] > 5])}/{total_questions}")

if __name__ == "__main__":
    # 50 diverse questions from NCERT subjects (Science, Social Science, Math/Generative)
    questions_50 = [
        # --- Science (Biology) ---
        "What is photosynthesis?",
        "Explain the function of stomata.",
        "What are the components of blood?",
        "Difference between arteries and veins.",
        "How is sex determined in human beings?",
        "Draw a labeled diagram of a neuron.",
        "What is the role of saliva in digestion?",
        "Explain the process of nutrition in Amoeba.",
        "What are trophic levels?",
        "Why should we conserve forests and wildlife?",
        
        # --- Science (Physics) ---
        "State Newton's first law of motion.",
        "What is the law of conservation of momentum?",
        "Define power and its unit.",
        "What is the scattering of light?",
        "Why do stars twinkle?",
        "State Ohm's Law.",
        "What is a solenoid?",
        "Fleming's Left-Hand Rule definition.",
        "What are the advantages of AC over DC?",
        "Explain the working of an electric motor.",
        
        # --- Science (Chemistry) ---
        "Balance the chemical equation: H2 + O2 -> H2O",
        "What is a displacement reaction?",
        "Why do ionic compounds have high melting points?",
        "Difference between roasting and calcination.",
        "What are amphoteric oxides?",
        "Define homologous series.",
        "Why is carbon tetravalent?",
        "Modern Periodic Law definition.",
        "Properties of ethanol.",
        "What is the pH scale?",

        # --- Social Science (History) ---
        "What was the French Revolution?",
        "Who was Napoleon Bonaparte?",
        "Explain the idea of Satyagraha.",
        "Why did the Non-Cooperation movement start?",
        "Who was Giuseppe Mazzini?",
        "What is the outcome of the Treaty of Vienna 1815?",
        "Explain the concept of Liberalism.",
        "What was the Jallianwala Bagh massacre?",
        "Significance of the Civil Disobedience Movement.",
        "Who were the Jacobins?",
        
        # --- Social Science (Geography/Civics) ---
        "What is resource planning?",
        "Classify resources on the basis of origin.",
        "What is federalism?",
        "Features of democracy.",
        "What is power sharing?",
        "What is the role of political parties?",
        "Different sectors of the Indian economy.",
        "What is globalization?",
        "Functions of the Reserve Bank of India.",
        "What is consumer protection?"
    ]
    
    run_benchmark_50(questions_50)

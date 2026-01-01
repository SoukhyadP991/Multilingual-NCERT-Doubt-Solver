import logging
import os
import re
from llama_cpp import Llama
import config

logging.basicConfig(level=logging.INFO)

class LocalLLMGenerator:
    def __init__(self):
        self.llm = self._load_model()

    def _load_model(self):
        """Loads the quantized GGUF model."""
        if not os.path.exists(config.MODEL_PATH):
            logging.error(f"Model file not found at {config.MODEL_PATH}. Please download it.")
            return None
        
        try:
            logging.info(f"Loading model from {config.MODEL_PATH}...")
            return Llama(
                model_path=config.MODEL_PATH,
                n_ctx=config.CONTEXT_WINDOW,
                n_threads=os.cpu_count() - 2, # Reserve some threads
                verbose=False
            )
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            return None

    def generate_answer(self, query: str, context_docs: list) -> str:
        """
        Generates an answer given the query and retrieved context.
        """
        if not self.llm:
            return "Error: Language Model is not loaded."

        # Format Context
        # We keep the source in context so the model knows THEM, but we tell it not to print them.
        context_text = "\n\n".join([
            f"[Source: {d.metadata.get('source', 'Unknown').replace('.pdf', '')}, Page: {d.metadata.get('page', 'Unknown')}]\n{d.page_content}" 
            for d in context_docs
        ])
        
        # Simplified Prompt - Remove conflicting instructions
        prompt = f"""[INST] You are an intelligent NCERT Doubt Solver. 
Goal: Answer the student's question clearly and accurately based ONLY on the provided context.

**Formatting Rules (Match this style):**
1. **Bold Keywords**: Bold terms like **chlorophyll**, **stomata**, **glucose**, **sunlight**, etc.
2. **Structure**: 
   - Start with a clear definition.
   - Limit paragraphs to 2-3 sentences max.
   - Use spaces between paragraphs.
3. **Equations**: 
   - If applicable, write "The word equation is:" followed by the equation on a new line.
   - Example: 
     Carbon dioxide + Water → Glucose + Oxygen
     *(in the presence of sunlight and chlorophyll)*
4. **No Citations**: Do NOT include source names or page numbers in the text.

**Context:**
{context_text}

**Question:** {query}
[/INST]"""

        output = self.llm(
            prompt,
            max_tokens=config.MAX_NEW_TOKENS,
            temperature=config.TEMPERATURE,
            stop=["</s>", "[/INST]"],
            echo=False
        )
        
        raw_answer = output['choices'][0]['text'].strip()

        # Post-Processing: Super Aggressive Removal
        # 1. Remove standard (Source: ...)
        clean_answer = re.sub(r'\s*[\(\[]\s*Source:.*?[\)\]]', '', raw_answer, flags=re.IGNORECASE)
        # 2. Remove (Filename.pdf, Page: X) patterns
        clean_answer = re.sub(r'\s*[\(\[]\s*[a-zA-Z0-9_]+\.pdf.*?[\)\]]', '', clean_answer, flags=re.IGNORECASE)
        # 3. Remove (ClassX_..., Page: X) patterns
        clean_answer = re.sub(r'\s*[\(\[]\s*Class\d+.*?[\)\]]', '', clean_answer, flags=re.IGNORECASE)
        # 4. Remove standalone (Page: X)
        clean_answer = re.sub(r'\s*[\(\[]\s*Page:.*?[\)\]]', '', clean_answer, flags=re.IGNORECASE)
        
        # Consolidate Sources for the Footer
        unique_sources = {}
        for doc in context_docs:
            source_name = doc.metadata.get('source', 'Unknown').replace('.pdf', '')
            page = doc.metadata.get('page', 'Unknown')
            if source_name not in unique_sources:
                unique_sources[source_name] = set()
            unique_sources[source_name].add(str(page))
            
        source_footer = "\n\n**Source:**\n"
        for name, pages in unique_sources.items():
            sorted_pages = sorted(list(pages), key=lambda x: int(x) if x.isdigit() else x)
            source_footer += f"- {name} (Page: {', '.join(sorted_pages)})\n"
            
        return clean_answer + source_footer

if __name__ == "__main__":
    # Test stub (requires model file)
    generator = LocalLLMGenerator()
    if generator.llm:
        print(generator.generate_answer("Test Question", []))

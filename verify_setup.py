import os
import sys
import importlib

def check_import(module_name):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} is installed.")
        return True
    except ImportError:
        print(f"❌ {module_name} is NOT installed.")
        return False

def check_path(path, description):
    if os.path.exists(path):
        print(f"✅ {description} found: {path}")
        return True
    else:
        print(f"❌ {description} NOT found: {path}")
        return False

def verify_setup():
    print("=== Verifying Multilingual NCERT Doubt Solver Setup ===\n")

    # 1. Check Dependencies
    print("--- Checking Dependencies ---")
    dependencies = [
        "streamlit", "langchain", "faiss", "sentence_transformers", 
        "pytesseract", "pdf2image", "llama_cpp", "langdetect"
    ]
    all_deps_ok = all([check_import(dep) for dep in dependencies])
    
    # 2. Check Directories
    print("\n--- Checking Directories ---")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "raw")
    models_dir = os.path.join(base_dir, "models")
    
    dirs_ok = all([
        check_path(data_dir, "Data Directory"),
        check_path(models_dir, "Models Directory")
    ])

    # 3. Check Models (Warning only)
    print("\n--- Checking Model Files ---")
    model_files = os.listdir(models_dir) if os.path.exists(models_dir) else []
    gguf_models = [f for f in model_files if f.endswith(".gguf")]
    
    if gguf_models:
        print(f"✅ GGUF Model found: {gguf_models[0]}")
    else:
        print("⚠️ No GGUF model found in `models/`. Please download one (e.g., Mistral-7B).")

    # 4. Check Config
    print("\n--- Checking Config ---")
    try:
        import config
        print(f"✅ Config loaded. Model path set to: {config.MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error loading config: {e}")

    print("\n=== Verification Complete ===")
    if not all_deps_ok:
        print("ACTION REQUIRED: Run `pip install -r requirements.txt`")
    if not gguf_models:
        print("ACTION REQUIRED: Download a .gguf model into `models/`")

if __name__ == "__main__":
    verify_setup()

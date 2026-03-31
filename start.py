"""
System Verification and Startup - Jatayu AFDRN
Checks all dependencies and starts both servers
"""
import sys
import subprocess
import importlib
import time
import webbrowser
from pathlib import Path

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name.replace("-", "_")
    
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_missing_packages():
    """Install missing required packages"""
    required = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("httpx", "httpx"),
        ("websockets", "websockets"),
        ("python-multipart", "multipart"),
        ("llama-cpp-python", "llama_cpp"),
        ("sse-starlette", "sse_starlette"),
    ]
    
    missing = []
    print("🔍 Checking dependencies...")
    for pip_name, import_name in required:
        if check_package(pip_name, import_name):
            print(f"  ✓ {pip_name}")
        else:
            print(f"  ✗ {pip_name} - MISSING")
            missing.append(pip_name)
    
    if missing:
        print(f"\n📦 Installing {len(missing)} missing packages...")
        for package in missing:
            print(f"  Installing {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package, "--quiet"])
        print("  ✓ All packages installed")
    else:
        print("  ✓ All dependencies installed")
    
    return len(missing) == 0

def check_model():
    """Check if model is downloaded"""
    model_path = Path("models/LFM2.5-1.2B-Instruct-Q4_K_M.gguf")
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Model found ({size_mb:.0f} MB)")
        return True
    else:
        print(f"  ✗ Model not found at {model_path}")
        return False

def check_database():
    """Check if database exists"""
    db_path = Path("jatayu.db")
    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        print(f"  ✓ Database found ({size_kb:.1f} KB)")
    else:
        print("  ℹ Database will be created on first run")
    return True

def start_server(name, command, port):
    """Start a server in a new window"""
    print(f"\n🚀 Starting {name}...")
    if sys.platform == "win32":
        # Windows: Start in new console window
        process = subprocess.Popen(
            command,
            creationflags=subprocess.CREATE_NEW_CONSOLE,
            cwd=Path.cwd()
        )
    else:
        # Unix: Start in background
        process = subprocess.Popen(command, cwd=Path.cwd())
    
    print(f"  ✓ {name} started (PID: {process.pid})")
    print(f"  📍 Running on port {port}")
    return process

def main():
    print("=" * 70)
    print("🦅 JATAYU - Autonomous Fraud Detection & Response Network")
    print("=" * 70)
    print()
    
    # Step 1: Verify dependencies
    print("[1/5] Verifying dependencies...")
    install_missing_packages()
    print()
    
    # Step 2: Check model
    print("[2/5] Checking AI model...")
    model_exists = check_model()
    if not model_exists:
        print("  ⚠️  Model not found. Run: python run_llama_server.py")
        print("  (It will download automatically on first run)")
    print()
    
    # Step 3: Check database
    print("[3/5] Checking database...")
    check_database()
    print()
    
    # Step 4: Start Llama Server
    print("[4/5] Starting AI Model Server...")
    llama_cmd = [sys.executable, "run_llama_server.py"]
    llama_process = start_server("Llama Server", llama_cmd, 8080)
    print("  ⏳ Waiting 15 seconds for model initialization...")
    time.sleep(15)
    
    # Step 5: Start Main Application
    print("[5/5] Starting Main Application...")
    main_cmd = [sys.executable, "-m", "uvicorn", "main:app", "--reload", 
                "--host", "0.0.0.0", "--port", "8000"]
    main_process = start_server("Main Application", main_cmd, 8000)
    print("  ⏳ Waiting 5 seconds for startup...")
    time.sleep(5)
    
    # Success!
    print()
    print("=" * 70)
    print("✅ JATAYU IS RUNNING!")
    print("=" * 70)
    print()
    print("🌐 Access URLs:")
    print("  📊 Dashboard:    http://localhost:8000")
    print("  📚 API Docs:     http://localhost:8000/docs")
    print("  🤖 Llama Server: http://127.0.0.1:8080")
    print()
    print("📝 Process Information:")
    print(f"  Llama Server PID: {llama_process.pid}")
    print(f"  Main App PID:     {main_process.pid}")
    print()
    print("🧪 Test the System:")
    print("  1. Upload CSV: Use web UI or POST to /upload")
    print("  2. Sample file: Qwen_csv_20260317_1cadcof0r.csv")
    print("  3. View results: Dashboard shows real-time processing")
    print()
    print("🛑 To Stop:")
    print("  - Close the console windows, or")
    print("  - Press Ctrl+C in each terminal")
    print()
    print("=" * 70)
    
    # Open browser
    print("\n🌐 Opening dashboard in browser...")
    time.sleep(2)
    try:
        webbrowser.open("http://localhost:8000")
    except:
        pass
    
    print("\n✨ System is ready! This window can stay open for monitoring.")
    print("   Both servers are running in separate console windows.")
    print("\nPress Enter to exit this launcher (servers will keep running)...")
    try:
        input()
    except KeyboardInterrupt:
        pass
    
    print("\n👋 Launcher closed. Servers are still running in background.")
    print("   Close their console windows to stop them.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nPress Enter to exit...")
        input()
        sys.exit(1)

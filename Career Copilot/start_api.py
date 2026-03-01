"""
Startup script for Career Readiness API
Loads environment variables from .env and starts the FastAPI server
"""

import os
import sys
from pathlib import Path

# Load environment variables from .env file
from dotenv import load_dotenv

env_file = Path(__file__).parent / ".env"
if env_file.exists():
    load_dotenv(env_file)
    print(f"✓ Loaded environment from {env_file}")
else:
    print(f"⚠️  .env file not found at {env_file}")
    print("   Create .env file with your GEMINI_API_KEY")
    sys.exit(1)

# Verify API key is set
if not os.getenv("GEMINI_API_KEY") and not os.getenv("OPENAI_API_KEY"):
    print("❌ Error: No API key found!")
    print("   Set GEMINI_API_KEY or OPENAI_API_KEY in .env file")
    sys.exit(1)

# Start the API server
import uvicorn
from api_server import app

if __name__ == "__main__":
    print("\n🚀 Starting Career Readiness API Server...")
    print("   API will be available at: http://localhost:8000")
    print("   Docs at: http://localhost:8000/docs")
    print("\n   Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )

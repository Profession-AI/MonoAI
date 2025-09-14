#!/usr/bin/env python3
"""
MonoAI Command Line Interface

This module provides a command-line interface for MonoAI.

Available Commands:
    serve        Start the API server with configurable options

Examples:
    # Start server with default settings
    monoai serve
    
    # Start server on custom host and port
    monoai serve --host 127.0.0.1 --port 8080
    
    # Start server with auto-reload enabled
    monoai serve --reload
    
    # Start server with multiple workers
    monoai serve --workers 4
    
Configuration:
    The server uses the configuration from main.py, which should include:
    - Model configuration
    - Agent definitions
    - Rate limiter settings (optional)
    - User validator (optional)
"""
import typer
from typing import Optional
import sys
import os

# Add project root directory to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

cli_app = typer.Typer(
    help="MonoAI CLI - Command line interface for MonoAI API server",
    no_args_is_help=True,
    add_completion=False
)

@cli_app.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0", 
        "--host", "-h", 
        help="Host address to bind the server to"
    ),
    port: int = typer.Option(
        8000, 
        "--port", "-p", 
        help="Port number to bind the server to"
    ),
    reload: bool = typer.Option(
        False, 
        "--reload", "-r", 
        help="Enable auto-reload when code changes are detected"
    ),
    workers: Optional[int] = typer.Option(
        None, 
        "--workers", "-w", 
        help="Number of worker processes to use (None for single process)"
    ),
    log_level: str = typer.Option(
        "info", 
        "--log-level", "-l", 
        help="Log level (debug, info, warning, error, critical)"
    ),
):
    """
    Start the MonoAI API server.
    
    This command starts the FastAPI server with all configured endpoints
    for models, agents, authentication, and rate limiting.
    
    The server will be available at http://{host}:{port} and includes:
    - Interactive API documentation at /docs
    - Alternative API documentation at /redoc
    - Health check endpoint at /health
    
    Examples:
        # Start with default settings
        monoai serve
        
        # Start on custom host and port
        monoai serve --host 127.0.0.1 --port 8080
        
        # Start with auto-reload for development
        monoai serve --reload
        
        # Start with multiple workers for production
        monoai serve --workers 4 --log-level warning
    """
    print(f"🚀 Starting MonoAI server on {host}:{port}")
    print(f"📊 Log level: {log_level}")
    if reload:
        print("🔄 Auto-reload enabled")
    if workers:
        print(f"👥 Workers: {workers}")
    else:
        print("👤 Single process mode")
    
    try:
        from main import app as main_app
        main_app.serve(host=host, port=port, reload=reload, workers=workers, log_level=log_level)
    except ImportError as e:
        print(f"❌ Error importing main app: {e}")
        print("💡 Make sure main.py exists in the project root")
        raise typer.Exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        raise typer.Exit(1)

@cli_app.command()
def hello():
    """
    Test command to verify CLI functionality.
    
    This is a simple test command that prints a greeting message
    to verify that the CLI is working correctly.
    """
    print("👋 Hello from MonoAI CLI!")
    print("✅ CLI is working correctly")

@cli_app.command()
def version():
    """
    Show MonoAI version information.
    
    Displays the current version of MonoAI and related information.
    """
    try:
        import monoai
        print(f"MonoAI version: {getattr(monoai, '__version__', 'unknown')}")
    except ImportError:
        print("MonoAI version: unknown (not installed)")
    
    print(f"Python version: {sys.version}")
    print(f"CLI location: {__file__}")

@cli_app.command()
def info():
    """
    Show detailed information about the MonoAI installation.
    
    Displays comprehensive information about the MonoAI installation,
    including available models, agents, and configuration.
    """
    print("📋 MonoAI Information")
    print("=" * 50)
    
    # Version info
    try:
        import monoai
        print(f"Version: {getattr(monoai, '__version__', 'unknown')}")
    except ImportError:
        print("Version: unknown (not installed)")
    
    # Python info
    print(f"Python: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # Project info
    print(f"Project root: {project_root}")
    print(f"CLI file: {__file__}")
    
    # Check main.py
    main_py_path = os.path.join(project_root, "main.py")
    if os.path.exists(main_py_path):
        print(f"✅ main.py found: {main_py_path}")
    else:
        print(f"❌ main.py not found: {main_py_path}")
    
    # Check for configuration files
    config_files = ["requirements.txt", "pyproject.toml", "setup.py"]
    print("\n📁 Configuration files:")
    for config_file in config_files:
        config_path = os.path.join(project_root, config_file)
        if os.path.exists(config_path):
            print(f"  ✅ {config_file}")
        else:
            print(f"  ❌ {config_file}")
    
    print("\n🔗 Available commands:")
    print("  serve    - Start the API server")
    print("  hello    - Test CLI functionality")
    print("  version  - Show version information")
    print("  info     - Show detailed information")
    
if __name__ == "__main__":
    cli_app()

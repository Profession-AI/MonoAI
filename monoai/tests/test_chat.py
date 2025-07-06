import pytest
import shutil
import os
import time
from pathlib import Path
from monoai.chat import Chat

TEST_MODEL = "gpt-4o-mini"
TEST_SYSTEM_PROMPT = "You are a calculator, return only the number without any any other text"

"""
@pytest.fixture(autouse=True)
def cleanup_histories():
    yield
    # Delete JSON history folder
    json_history_path = Path("histories")
    if json_history_path.exists():
        shutil.rmtree(json_history_path)
    
    # Delete SQLite history folder
    sqlite_history_path = Path("sqlite_histories")
    if sqlite_history_path.exists():
        shutil.rmtree(sqlite_history_path)
"""
        
@pytest.fixture
def json_chat():
    """Create a new chat with JSON history."""
    return Chat(provider="openai", model=TEST_MODEL, system_prompt=TEST_SYSTEM_PROMPT)

@pytest.fixture
def sqlite_chat():
    """Create a new chat with SQLite history."""
    return Chat(provider="openai", model=TEST_MODEL, system_prompt=TEST_SYSTEM_PROMPT, history_type="sqlite")

def test_chat_initialization():
    """Test that chat is properly initialized with required attributes."""
    chat = Chat(provider="openai", model=TEST_MODEL, system_prompt=TEST_SYSTEM_PROMPT)
    assert chat.chat_id is not None
    assert chat._history is not None

def test_json_history_basic_operations(json_chat):
    """Test basic operations with JSON history."""
    # Test initial calculation
    response = json_chat.ask("2+2")
    assert response == "4"
    
    # Test continuation
    response = json_chat.ask("+2")
    assert response == "6"

def test_json_history_persistence(json_chat):
    """Test that JSON history persists between chat instances."""
    # First chat instance
    json_chat.ask("2+2")
    json_chat.ask("+2")
    
    # New chat instance with same history
    new_chat = Chat(provider="openai", model=TEST_MODEL, history_type="json", chat_id=json_chat.chat_id)
    response = new_chat.ask("+2")
    assert response == "8"

def test_sqlite_history_basic_operations(sqlite_chat):
    """Test basic operations with SQLite history."""
    # Test initial calculation
    response = sqlite_chat.ask("2+2")
    assert response == "4"
    
    # Test continuation
    response = sqlite_chat.ask("+2")
    assert response == "6"

def test_sqlite_history_persistence(sqlite_chat):
    """Test that SQLite history persists between chat instances."""
    # First chat instance
    sqlite_chat.ask("2+2")
    sqlite_chat.ask("+2")


def test_summarizer():
    """Test that the summarizer works."""
    new_chat = Chat(provider="openai", 
                    model=TEST_MODEL, 
                    history_type="json", 
                    history_summarizer_provider="openai", 
                    history_summarizer_model="gpt-4o-mini", 
                    history_summarizer_max_tokens=100)
    
    new_chat.ask("Mi chiamo Giuseppe")
    response = new_chat.ask("Come mi chiamo? Ritorna solo il mio nome e niente altro.")
    assert response.lower() == "giuseppe"

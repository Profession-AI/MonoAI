from monoai.chat.base_history import BaseHistory
import os
import json
import uuid
import sqlite3
from monoai.models import Model
from pymongo import MongoClient
import datetime

class BaseHistory:

    def __init__(self, 
                 history_path: str, 
                 last_n: int=None): 
        self._history_path = history_path
        self._last_n = last_n
        
    def generate_chat_id(self):
        return str(uuid.uuid4())

    def load(self):
        pass

    def store(self, chat_id: str, messages: list):
        # Aggiungi un timestamp ISO 8601 a ciascun messaggio
        now = datetime.datetime.utcnow().isoformat() + 'Z'
        for msg in messages:
            if 'timestamp' not in msg:
                msg['timestamp'] = now
        return messages

    def clear(self):
        pass

class JSONHistory(BaseHistory):
    
    def __init__(self, 
                 history_path: str="histories/", 
                 last_n: int=None): 
        self._history_path = history_path
        self._last_n = last_n
        if not os.path.exists(self._history_path):
            os.makedirs(self._history_path)

    def load(self, chat_id: str):
        with open(self._history_path+chat_id+".json", "r") as f:
            self.messages = json.load(f)
        if self._last_n is not None and len(self.messages) > (self._last_n+1)*2:
            self.messages = [self.messages[0]]+self.messages[-self._last_n*2:]
        return self.messages
    
    def new(self, system_prompt: str):
        chat_id = self.generate_chat_id()
        self.store(chat_id, [{"role": "system", "content": system_prompt}])
        return chat_id

    def store(self, chat_id: str, messages: list):
        messages = super().store(chat_id, messages)
        # Load existing messages
        try:
            with open(self._history_path+chat_id+".json", "r") as f:
                existing_messages = json.load(f)
        except FileNotFoundError:
            existing_messages = []
        
        # Add the new messages (già con timestamp)
        new_messages = existing_messages + messages
        
        with open(self._history_path+chat_id+".json", "w") as f:
            json.dump(new_messages, f, indent=4)

class SQLiteHistory(BaseHistory):
    
    def __init__(self, db_path: str="histories/chat.db", last_n: int=None):
        self._db_path = db_path
        self._last_n = last_n
        self._init_db()
    
    def _init_db(self):
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    chat_id TEXT,
                    order_index INTEGER,
                    role TEXT,
                    content TEXT,
                    PRIMARY KEY (chat_id, order_index)
                )
            """)
    
    def load(self, chat_id: str):
        with sqlite3.connect(self._db_path) as conn:
            if self._last_n is not None:
                # Get system message
                cursor = conn.execute(
                    "SELECT role, content FROM messages WHERE chat_id = ? AND order_index = 0",
                    (chat_id,)
                )
                system_message = cursor.fetchone()
                
                # Get last N messages
                cursor = conn.execute(
                    """
                    SELECT role, content 
                    FROM messages 
                    WHERE chat_id = ? 
                    ORDER BY order_index DESC 
                    LIMIT ?
                    """,
                    (chat_id, self._last_n * 2)
                )
                last_messages = [{"role": role, "content": content} for role, content in cursor]
                last_messages.reverse()  # Reverse to get correct order
                
                # Combine system message with last N messages
                self.messages = [{"role": system_message[0], "content": system_message[1]}] + last_messages
            else:
                cursor = conn.execute(
                    "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY order_index",
                    (chat_id,)
                )
                self.messages = [{"role": role, "content": content} for role, content in cursor]
        return self.messages
    
    def new(self, system_prompt: str):
        chat_id = self.generate_chat_id()
        self.store(chat_id, [{"role": "system", "content": system_prompt, "order_index": 0}])
        return chat_id

    def store(self, chat_id: str, messages: list):
        messages = super().store(chat_id, messages)
        with sqlite3.connect(self._db_path) as conn:
            # Get the last order_index
            cursor = conn.execute(
                "SELECT MAX(order_index) FROM messages WHERE chat_id = ?",
                (chat_id,)
            )
            last_index = cursor.fetchone()[0]
            
            # If no messages exist yet, start from -1
            if last_index is None:
                last_index = -1
            
            # Insert the new messages con timestamp
            for i, message in enumerate(messages, start=last_index + 1):
                conn.execute(
                    "INSERT INTO messages (chat_id, order_index, role, content) VALUES (?, ?, ?, ?)",
                    (chat_id, i, message["role"], message["content"])
                )
                conn.commit()
        

class MongoDBHistory(BaseHistory):
    def __init__(self, db_path, db_name: str = "chat", collection_name: str = "histories", last_n: int = None):
        self._uri = db_path
        self._db_name = db_name
        self._collection_name = collection_name
        self._last_n = last_n
        self._client = MongoClient(self._uri)
        self._db = self._client[self._db_name]
        self._collection = self._db[self._collection_name]

    def load(self, chat_id: str):
        doc = self._collection.find_one({"chat_id": chat_id})
        if not doc:
            self.messages = []
            return self.messages
        messages = doc.get("messages", [])
        if self._last_n is not None and len(messages) > (self._last_n + 1) * 2:
            messages = [messages[0]] + messages[-self._last_n * 2:]
        self.messages = messages
        return self.messages

    def new(self, system_prompt: str):
        chat_id = self.generate_chat_id()
        messages = [{"role": "system", "content": system_prompt}]
        self.store(chat_id, messages)
        return chat_id

    def store(self, chat_id: str, messages: list):
        messages = super().store(chat_id, messages)
        # Get existing messages
        doc = self._collection.find_one({"chat_id": chat_id})
        existing_messages = doc.get("messages", []) if doc else []
        
        # Add the new messages (già con timestamp)
        new_messages = existing_messages + messages
        
        self._collection.update_one(
            {"chat_id": chat_id},
            {"$set": {"messages": new_messages}},
            upsert=True
        )


class HistorySummarizer():

    def __init__(self, model: Model, max_tokens: int=None):
        self._model = model
        self._max_tokens = max_tokens

    def summarize(self, messages: list):
        response = self._model.ask("Summarize the following conversation: "+json.dumps(messages))
        response = response["response"]
        return response


from coicoi.chat import Chat
from coicoi.chat.history import JSONHistory, SQLiteHistory
import asyncio

history = SQLiteHistory(last_n=1)
chat = Chat(provider="openai", model="gpt-4o-mini", history_type=history, system_prompt="Sei un esperto di storia, rispondi sempre in italiano")
print(chat.ask("Ciao!"))
print(chat.ask("Mi chiamo Giuseppe"))
print(chat.ask("Come mi chiamo?"))
print(chat.ask("Quanti anni hai?"))
print(chat.ask("Cosa fai?"))
print(chat.ask("Come sei nato?"))
"""
async def call():
    async for response in chat.ask_async("Spiegami qualcosa di lungo"):
        print(response, end="", flush=True)

asyncio.run(call())
"""
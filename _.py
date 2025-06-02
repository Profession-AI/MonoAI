from coicoi.chat import Chat


chat = Chat(provider="openai", 
            model="gpt-4o-mini", 
            history_type="sqlite",
            system_prompt="Sei un esperto di storia, rispondi sempre in italiano")

chat.ask("Mi chiamo Giuseppe")
response = chat.ask("Come mi chiamo?")
print(response)
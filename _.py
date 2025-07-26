from monoai.chat import Chat

chat = Chat(provider="openai", 
            model="gpt-4o-mini", 
            history_type="mongodb", 
            history_path="mongodb+srv://gfgullo:aYBSjqaQ4s3I7p0g@cluster0.hbajnzz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        )
print(chat.chat_id)
response = chat.ask("What's in this image?", file="todo.md")
print(response)

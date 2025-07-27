from monoai.chat import Chat

chat = Chat(provider="openai", 
            model="gpt-4o-mini", 
            history_type="mongodb", 
            history_path="mongodb+srv://gfgullo:aYBSjqaQ4s3I7p0g@cluster0.hbajnzz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        )
response = chat.ask_stream("Hi")
for r in response:
    print(r)

from monoai.chat import Chat
from monoai.chat.history import MongoDBHistory

mongo_history = MongoDBHistory(
    db_path="mongodb+srv://admin:GPnwZjTdu9K5IibW@cluster0.4j5co.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
    db_name="profai",
    collection_name="chats_content",
    last_n=10
)

chat = Chat(provider="mistral", 
            model="mistral-small-latest", 
            history_type=mongo_history,
            chat_id="c80475ec-347a-4c4f-bf82-716758d7d739"
        )

response = chat.ask("Spiegami in maniera più semplice")
print(response)
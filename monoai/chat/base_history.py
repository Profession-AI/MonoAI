import uuid

class BaseHistory:

    def generate_chat_id(self):
        return str(uuid.uuid4())

    def load(self):
        pass

    def save(self):
        pass

    def clear(self):
        pass
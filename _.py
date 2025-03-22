from pydantic_ai import Agent
from keys_manager import load_keys

load_keys()

agent = Agent("deepseek:deepseek-chat")
answer = agent.run_sync("What is the capital of France?")
print(answer)

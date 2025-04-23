from coicoi._agents import CollaborativeAgent
from coicoi.prompts.prompt_chain import PromptChain
from coicoi._tools import WebSearch


collaborative_agent = CollaborativeAgent(
    models=[
        {
            "provider": "openai",
            "model": "gpt-4o"
        },
        {
            "provider": "openai",
            "model": "gpt-4o-mini"
        }
    ],
    aggregator={
        "provider": "openai",
        "model": "gpt-4o-mini"
    },
    tools=[WebSearch()]
)

chain = PromptChain([
    "Who is the president of the United States in 2025?",
    "How old is him ?"
])

result = collaborative_agent.run(chain)
print(result)

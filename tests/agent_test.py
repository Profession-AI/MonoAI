from monoai.tools.websearch import WebSearch
from os import environ
from monoai.models import Model
from monoai._agents import Agent

agent = Agent(provider="openai", model="gpt-4o-mini", tools=[WebSearch().get_tool()])
result = agent.run("Who is the president of the United States in 2025?")
print(result)
from monoai.application import Application, RateLimiter, Limit
from monoai.models import Model
from monoai.agents import Agent

model = Model(provider="openai", model="gpt-4o-mini")

def weather(city: str):
    """
    Get the weather for a given city

    Args:
        city (string): The city to get the weather for

    Returns:
        The weather in the given city
    """

    return f"The weather in {city} is 24°C"

agent = Agent(model, paradigm="function_calling", name="weather", tools=[weather])
rate_limiter = RateLimiter(limits=[Limit(unit="token", value=1000, reset_unit="minute", reset_value=1)])
app = Application(name="weather_app", model=model, agents=[agent], rate_limiter=rate_limiter)

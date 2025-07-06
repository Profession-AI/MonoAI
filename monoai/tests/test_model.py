from monoai.models import Model
from monoai.prompts import Prompt, PromptChain

TEST_PROMPT = "This is a test"
TEST_PROMPT_FOLDER = "monoai/tests/prompts"
TEST_MODEL = "gpt-4o-mini"

def test_base_model():
    assert "response" in Model(provider="openai", model=TEST_MODEL).ask(TEST_PROMPT)

def test_base_model_with_prompt():
    prompt = Prompt(prompt_id=f"{TEST_PROMPT_FOLDER}/base")
    assert "response" in Model(provider="openai", model=TEST_MODEL).ask(prompt=prompt)

def test_model_with_system_prompt():
    system_prompt = "You are a helpful assistant"
    model = Model(provider="openai", model=TEST_MODEL, system_prompt=system_prompt)
    response = model.ask(TEST_PROMPT)
    assert "response" in response
    assert response["model"]["provider"] == "openai"
    assert response["model"]["name"] == TEST_MODEL

def test_model_with_token_counting():
    model = Model(provider="openai", model=TEST_MODEL, count_tokens=True)
    response = model.ask(TEST_PROMPT)
    assert "response" in response

def test_model_with_cost_counting():
    model = Model(provider="openai", model=TEST_MODEL, count_tokens=True, count_cost=True)
    response = model.ask(TEST_PROMPT)
    assert "response" in response

def test_model_with_prompt_chain():
    chain = PromptChain(prompts=[
        "What is 2+2?",
        "Based on the previous answer, what is 2+2+2?"
    ])
    model = Model(provider="openai", model=TEST_MODEL)
    response = model.ask(chain)
    assert "response" in response
    assert isinstance(response["response"], str)
    assert len(response["response"]) > 0

def test_model_with_prompt_variables():
    prompt = Prompt(
        prompt_id=f"{TEST_PROMPT_FOLDER}/with_variables",
        prompt_data={"country": "Italy"}
    )
    model = Model(provider="openai", model=TEST_MODEL)
    response = model.ask(prompt)
    assert "response" in response
    assert isinstance(response["response"], str)
    assert len(response["response"]) > 0
    assert "Rome" in response["response"]

def test_model_with_formatted_prompt():
    test_text = "The quick brown fox jumps over the lazy dog. This is a test text for summarization."
    prompt = Prompt(
        prompt_id=f"{TEST_PROMPT_FOLDER}/with_formatting",
        prompt_data={"text": test_text}
    )
    model = Model(provider="openai", model=TEST_MODEL)
    response = model.ask(prompt)
    assert "response" in response
    assert isinstance(response["response"], str)
    assert len(response["response"]) > 0
    assert "-" in response["response"]  # Check for bullet points


def test_model_with_type():
    prompt = Prompt(
        prompt="2+2=",
        response_type="int"
    )
    model = Model(provider="openai", model=TEST_MODEL)
    response = model.ask(prompt)
    assert "response" in response
    assert isinstance(response["response"], int)
    assert response["response"] == 4


def test_model_with_prompt_type():
    prompt = Prompt(prompt_id=f"{TEST_PROMPT_FOLDER}/with_type")
    model = Model(provider="openai", model=TEST_MODEL)
    response = model.ask(prompt)
    assert "response" in response
    assert isinstance(response["response"], int)
    assert response["response"] == 4


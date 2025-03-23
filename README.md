# COICOI
### Collective Intelligence via Collaborative Inference

**COICOI** is a framework designed to simplify interaction with modern AI models and harness their collective intelligence through multi-model concurrent inference and cooperation in solving complex tasks.
---

## Getting Started

### Load Your API Keys

Place your providers’ API keys in a file named `providers.keys` at the root of your project:

```
OPENAI = <YOUR OPENAI API KEY>
DEEPSEEK = <YOUR DEEPSEEK API KEY>
ANTHROPIC = <YOUR ANTHROPIC API KEY>
GEMINI = <YOUR GEMINI API KEY>
PERPLEXITY = <YOUR PERPLEXITY API KEY>
...
```

No further configuration is needed—COICOI automatically loads the correct key when a model is invoked.

---

## Usage Examples

### Using a Single Model

The `Model` class allows you to interact with a single AI model:

```python
from coicoi.models import Model

model = Model(
    provider_name="openai",
    model_name="gpt-4o-mini",
    count_tokens=True,
    count_cost=True
)

response = model.ask("What is the capital of Italy?")
```

---

### Using Multiple Models

The `MultiModels` class enables concurrent interaction with multiple AI models:

```python
from coicoi.models import MultiModels

multi_model = MultiModels(
    models=[
        {'provider': 'openai', 'model': 'gpt-4o-mini'},
        {'provider': 'deepseek', 'model': 'deepseek-chat'},
    ],
    count_tokens=True,
    count_cost=True
)

responses = multi_model.ask("What is the meaning of life?")
```

---

### Using Collective Intelligence

The `CollectiveModel` class allows you to query multiple models concurrently and merge their outputs into a unified, detailed response:

```python
from coicoi.models import CollectiveModel

collective_model = CollectiveModel(
    models=[
        {'provider': 'openai', 'model': 'gpt-4o-mini'},
        {'provider': 'deepseek', 'model': 'deepseek-chat'},
    ],
    aggregator={'provider': 'deepseek', 'model': 'deepseek-chat'},
    count_tokens=True,
    count_cost=True
)

response = collective_model.ask("What is the meaning of life?")
```

---

Let me know if you'd like to add usage tips, error handling, or a section for contributing!

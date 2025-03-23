# COICOI
### Collective Intelligence via Collaborative Inference

## Usage

### Load your API Keys
Put your providers API Keys in a file providers.keys in the root of your projects

```
OPENAI = <YOUR OPENAI API KEY>
DEEPSEEK = <YOUR DEEPSEEK API KEY>
ANTROPHIC = <YOUR ANTROPHIC API KEY>
GEMINI = <YOUR GEMINI API KEY>
...
 PERPLEXITY = <YOUR PERPLEXITY API KEY>
```

This is all, the correct key is automatically loaded when you invoke the model

### Individual Model
```python
from coicoi.models import Model

model = Model(
        provider_name="openai",
        model_name="gpt-4o-mini",
        count_tokens=True,
        count_cost=True
    )

result = model.ask("What is the capital of Italy?")
```


### Multiple Models
```python
from coicoi.models import MultiModels

multi_model = MultiModel(
  models=[
    {'provider': 'openai', 'model': 'gpt-4o-mini'},
    {'provider': 'deepseek', 'model': 'deepseek-chat'},
  ],
  count_tokens=True,
  count_cost=True
)

results = multi_model.ask("What is the meaning of life?")
```

### Collective Model
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

results = collective_model.ask("What is the meaning of life?")
```

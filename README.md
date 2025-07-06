# MonoAI

<img src="https://raw.githubusercontent.com/Profession-AI/MonoAI/refs/heads/main/docs/res/logo.png" alt="alt text" width="500"/>

### The complete framework to build AI-powered applications


**MonoAI** is a Python library that enables structured, standardized, and efficient interactions with multiple AI models, harnessing their collective intelligence to deliver richer, more accurate responses.

---

## 🚀 Quick Start

### Prompts
In MonoAI, a **prompt** is the request you send to a model.  
You can define prompts in three different ways:

---

#### 1. Plain Text
Directly write the prompt inside the `ask` method:
```python
from MonoAI.models import Model

model = Model()
response = model.ask("What is 2 + 2?")
```

---

#### 2. `Prompt` Class
Use the `Prompt` class to enhance your prompts with additional parameters such as response types, chaining, and iteration control:
```python
from MonoAI.models import Model
from MonoAI.prompts import Prompt

model = Model()
prompt = Prompt("What is 2 + 2?", response_type=int)
response = model.ask(prompt)
```

---

#### 3. `.prompt` Files
Store prompts in external `.prompt` files to separate your codebase from content logic.

**Example - `sum.prompt`:**
```
What is 2 + 2?
```

```python
from MonoAI.models import Model
from MonoAI.prompts import Prompt

model = Model()
prompt = Prompt(prompt_id="sum", response_type=int)
response = model.ask(prompt)
```

You can also define additional metadata using an XML-like syntax inside `.prompt` files:

**Example with metadata:**
```
# sum.prompt
<prompt response_type="int">
What is 2 + 2?
</prompt>
```

---

### Models
MonoAI provides multiple model interfaces to handle different use cases, from basic generation to multi-model aggregation:

---

#### `Model`
The standard interface for interacting with a single AI model.

---

#### `MultiModel`
Send a prompt to multiple models asynchronously and retrieve their individual outputs:
```python
from MonoAI.models import MultiModel

model = MultiModel(models=[
    {"provider": "openai", "model": "gpt-4o"},
    {"provider": "deepseek", "model": "chat"}
])
response = model.ask("What is the capital of Italy?")
```

---

#### `CollectiveModel`
Send a prompt to multiple models and aggregate their outputs using a separate aggregator model for a richer, consolidated answer:
```python
from MonoAI.models import CollectiveModel

model = CollectiveModel(
    models=[
        {"provider": "openai", "model": "gpt-4o"},
        {"provider": "deepseek", "model": "chat"}
    ],
    aggregator={"provider": "openai", "model": "gpt-4o"}
)
response = model.ask("What is the capital of Italy?")
```

---

#### `AutoModel`
Automatically select the best model for a given prompt based on configuration or inference:
```python
from MonoAI.models import AutoModel

model = AutoModel()
response = model.ask("What is the capital of Italy?")
```

---

### 🔑 API Key Management
MonoAI simplifies API key management through a `providers.keys` file in the root directory.  
Each line should follow this format:
```
PROVIDER_NAME=API_KEY
```

**Example `providers.keys`:**
```
OPENAI=sk-proj-ABCDE12345
DEEPSEEK=sk-proj-FGHIJ67890
```

MonoAI automatically loads these keys at runtime — no extra setup needed.

---

## ⚙️ Configuration

Configure MonoAI globally via a `monoai.yaml` file.  
Supported fields:

| Field             | Description                                          | Default             |
|-------------------|------------------------------------------------------|---------------------|
| `prompts_path`    | Directory where `.prompt` files are stored           | `""` (current folder) |
| `keysfile_path`   | Path to the API keys file                            | `"providers.keys"`  |
| `base_model`      | Default model to use when no model is specified      | None                |

**Example `monoai.yaml`:**
```yaml
prompts_path: prompts
keysfile_path: providers.keys
base_model:
  provider: openai
  model: gpt-4o-mini
```

---

## 📚 Documentation

Full documentation is available at:  
👉 [MonoAI Documentation](https://profession-ai.github.io/MonoAI/monoai.html)

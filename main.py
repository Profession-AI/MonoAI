from monoai.models import Model
    from monoai.prompts import Prompt
    # Initialize model
    
        # Remember to set your provider key in the providers.keys file
        provider = "" # set your provider (es. openai, anthropic, google, etc.)
        model = "" # set your model (es. gpt-4o-mini, claude-3, gemini-1.5-flash, etc.)
        model = Model(provider=provider, model=model)
        
    # Load and use the hello prompt
    prompt = Prompt(prompt_id="hello")
    response = model.ask(prompt)
    print(f"Response: {response['response']}")
    
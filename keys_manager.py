from os import environ

def load_keys():
    
    KEY_EXT = "_API_KEY"

    try:
        with open('providers.keys', 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                if '=' in line:
                    provider, key = line.split('=', 1)
                    provider, key = provider.strip().upper(), key.strip().strip('"')
                    
                    if not provider.endswith(KEY_EXT):
                        provider += KEY_EXT

                    environ[provider] = key
                else:
                    print(f"Warning: Invalid line format in providers.keys: {line}")
    except FileNotFoundError:
        raise FileNotFoundError("providers.keys file not found. Please create it with your API keys.")
    

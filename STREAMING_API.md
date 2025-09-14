# API di Streaming per MonoAI

Questo documento descrive come utilizzare gli endpoint di streaming per modelli e agenti in MonoAI.

## Endpoint Disponibili

### 1. Endpoint HTTP di Streaming

#### Modello con Streaming
```
POST /model/stream
```

**Body:**
```json
{
  "prompt": "Il tuo prompt qui",
  "user_id": "opzionale_user_id"
}
```

**Risposta:**
```json
{
  "response": "Risposta del modello",
  "usage": {
    "total_tokens": 150,
    "prompt_tokens": 20,
    "completion_tokens": 130
  }
}
```

#### Agente con Streaming
```
POST /agent/{agent_name}/stream
```

**Body:**
```json
{
  "prompt": "Il tuo prompt qui",
  "user_id": "opzionale_user_id"
}
```

**Risposta:**
```json
{
  "prompt": "Il tuo prompt qui",
  "response": "Risposta dell'agente",
  "iterations": [...],
  "usage": {
    "total_tokens": 200,
    "prompt_tokens": 30,
    "completion_tokens": 170
  }
}
```

### 2. Endpoint WebSocket

#### WebSocket Modello
```
WS /model/ws
```

**Messaggi inviati:**
```json
{
  "prompt": "Il tuo prompt qui",
  "user_id": "opzionale_user_id"
}
```

**Messaggi ricevuti:**
```json
// Chunk di streaming
{
  "type": "chunk",
  "content": "parte della risposta"
}

// Risposta completa
{
  "type": "complete",
  "result": {
    "response": "Risposta completa",
    "usage": {...}
  }
}

// Errore
{
  "error": "Messaggio di errore"
}
```

#### WebSocket Agente
```
WS /agent/{agent_name}/ws
```

**Messaggi inviati:**
```json
{
  "prompt": "Il tuo prompt qui",
  "user_id": "opzionale_user_id"
}
```

**Messaggi ricevuti:**
```json
// Chunk di streaming
{
  "type": "chunk",
  "content": "parte della risposta"
}

// Risposta completa
{
  "type": "complete",
  "result": {
    "prompt": "Il tuo prompt",
    "response": "Risposta completa",
    "iterations": [...],
    "usage": {...}
  }
}

// Errore
{
  "error": "Messaggio di errore"
}
```

## Esempi di Utilizzo

### Python con requests

```python
import requests

# Modello normale
response = requests.post("http://localhost:8000/model", json={
    "prompt": "Ciao mondo"
})
print(response.json())

# Modello streaming
response = requests.post("http://localhost:8000/model/stream", json={
    "prompt": "Ciao mondo streaming"
})
print(response.json())
```

### Python con WebSocket

```python
import asyncio
import websockets
import json

async def test_websocket():
    async with websockets.connect("ws://localhost:8000/model/ws") as websocket:
        # Invia prompt
        await websocket.send(json.dumps({
            "prompt": "Ciao WebSocket"
        }))
        
        # Ricevi risposte
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            
            if data["type"] == "chunk":
                print(f"Chunk: {data['content']}")
            elif data["type"] == "complete":
                print(f"Completo: {data['result']}")
                break
            elif "error" in data:
                print(f"Errore: {data['error']}")
                break

asyncio.run(test_websocket())
```

### JavaScript con fetch

```javascript
// Modello normale
fetch('http://localhost:8000/model', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        prompt: 'Ciao mondo'
    })
})
.then(response => response.json())
.then(data => console.log(data));

// Modello streaming
fetch('http://localhost:8000/model/stream', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        prompt: 'Ciao mondo streaming'
    })
})
.then(response => response.json())
.then(data => console.log(data));
```

### JavaScript con WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8000/model/ws');

ws.onopen = function() {
    // Invia prompt
    ws.send(JSON.stringify({
        prompt: 'Ciao WebSocket'
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    
    if (data.type === 'chunk') {
        console.log('Chunk:', data.content);
    } else if (data.type === 'complete') {
        console.log('Completo:', data.result);
    } else if (data.error) {
        console.error('Errore:', data.error);
    }
};
```

## Rate Limiting

Tutti gli endpoint supportano il rate limiting se configurato nell'applicazione:

- **Identificazione utente**: Prima cerca `user_id` nel body della richiesta, poi usa l'IP del client
- **Limiti**: Supporta sia limiti per richieste che per token
- **Statistiche**: Disponibili tramite `/rate-limit/stats` e `/rate-limit/stats/{user_id}`

## Gestione Errori

### Errori HTTP
- `400`: JSON non valido o prompt mancante
- `404`: Agente non trovato
- `429`: Rate limit superato

### Errori WebSocket
- Messaggi di errore inviati come JSON con campo `error`
- Connessione chiusa automaticamente in caso di errore grave

## Note Tecniche

1. **Streaming**: Gli endpoint `/stream` abilitano lo streaming interno ma restituiscono comunque la risposta completa
2. **WebSocket**: Gli endpoint WebSocket forniscono streaming in tempo reale con chunk separati
3. **Rate Limiting**: Funziona sia per endpoint HTTP che WebSocket
4. **Usage Tracking**: Tutti gli endpoint restituiscono informazioni sull'uso dei token
5. **CORS**: Configurato per accettare richieste da qualsiasi origine

## User Validation

L'applicazione supporta la validazione degli utenti tramite il parametro `user_validator`:

```python
def my_user_validator(user_id: str):
    # Restituisce True se valido, False se non valido, 
    # o una stringa se valido ma normalizzato
    if user_id.startswith("user_"):
        return True
    elif user_id.startswith("admin_"):
        return f"user_{user_id[6:]}"  # Normalizza
    else:
        return False

app = Application(
    name="My App",
    model=model,
    user_validator=my_user_validator
)
```

### Endpoint di Validazione

```
POST /validate-user
```

**Body:**
```json
{
  "user_id": "user_123"
}
```

**Risposta:**
```json
{
  "valid": true,
  "user_id": "user_123",
  "normalized": null
}
```

## Configurazione

Per utilizzare gli endpoint di streaming, assicurati che:

1. L'applicazione abbia un modello configurato per `/model` e `/model/stream`
2. L'applicazione abbia agenti configurati per `/agent/{name}` e `/agent/{name}/stream`
3. I modelli e agenti supportino il metodo `enable_streaming()`
4. Il rate limiter sia configurato se necessario
5. Il user_validator sia configurato se necessario per la validazione utenti

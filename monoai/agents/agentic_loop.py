import json
import inspect
from typing import Any, Dict, List, Optional
from ..prompts import Prompt


class _FunctionCallingMixin:
    """Mixin per gestire chiamate di funzioni con tool calls OpenAI.
    
    Questo mixin fornisce metodi per gestire le chiamate di tool in formato
    OpenAI function calling, convertendo le risposte in messaggi standardizzati.
    """
    
    def _call_tool(self, tool_call: Any) -> Dict[str, str]:
        """Esegue una chiamata di tool e restituisce la risposta formattata.
        
        Parameters
        ----------
        tool_call : Any
            Oggetto tool_call contenente informazioni sulla funzione da chiamare.
            Deve avere attributi `function.name`, `function.arguments` e `id`.
        
        Returns
        -------
        Dict[str, str]
            Dizionario con la risposta del tool formattata per i messaggi:
            - tool_call_id: ID della chiamata tool
            - role: Ruolo del messaggio (sempre "tool")
            - name: Nome della funzione chiamata
            - content: Risposta del tool come stringa
        """
        function_name = tool_call.function.name
        function_to_call = self._tools[function_name]
        function_args = json.loads(tool_call.function.arguments)                
        function_response = str(function_to_call(**function_args))
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }


class _ReactMixin:
    """Mixin per gestire chiamate di tool in formato React/JSON.
    
    Questo mixin fornisce metodi per gestire le chiamate di tool in formato
    JSON strutturato, tipico degli approcci React-style per gli agenti AI.
    """
    
    def _encode_tool(self, func: Any) -> str:
        """Codifica una funzione in formato stringa descrittivo.
        
        Parameters
        ----------
        func : Any
            Funzione da codificare. Deve avere attributi `__name__` e `__doc__`.
        
        Returns
        -------
        str
            Stringa descrittiva della funzione nel formato:
            "nome_funzione(signature): documentazione"
        """
        sig = inspect.signature(func)
        doc = inspect.getdoc(func)
        encoded = func.__name__ + str(sig) + ": " + doc
        encoded = encoded.replace("\n", " ")
        return encoded
    
    def _call_tool(self, tool_call: Dict[str, Any]) -> Any:
        """Esegue una chiamata di tool in formato React.
        
        Parameters
        ----------
        tool_call : Dict[str, Any]
            Dizionario con le informazioni del tool:
            - name: Nome del tool da chiamare
            - arguments: Argomenti del tool come dizionario
        
        Returns
        -------
        Any
            Risultato dell'esecuzione del tool
        """
        tool = self._tools[tool_call["name"]]
        kwargs = list(tool_call["arguments"].values())
        return tool(*kwargs)


class _AgenticLoop:
    """Classe base per tutti i loop agentici.
    
    Questa classe fornisce la funzionalità di base per tutti gli agenti AI,
    inclusa la gestione degli strumenti, la creazione di messaggi e l'esecuzione
    del modello. È progettata per essere estesa da classi specifiche che
    implementano diversi approcci agentici.
    
    Attributes
    ----------
    _model : Any
        Modello AI da utilizzare per l'esecuzione
    _agentic_prompt : str
        Prompt personalizzato per l'agente (opzionale)
    _debug : bool
        Flag per abilitare la stampa di debug
    _max_iter : Optional[int]
        Numero massimo di iterazioni consentite
    _tools : Dict[str, Any]
        Dizionario degli strumenti disponibili, mappati per nome
    """
    
    def __init__(self, model: Any, agentic_prompt: str=None, debug: bool=False, max_iter: Optional[int]=None) -> None:
        """Inizializza l'agente con il modello e gli strumenti.
        
        Parameters
        ----------
        model : Any
            Modello AI da utilizzare per l'esecuzione
        agentic_prompt : str
            Prompt personalizzato per l'agente (None per usare quello di default)
        debug : bool
            Flag per abilitare la stampa di debug
        max_iter : Optional[int]
            Numero massimo di iterazioni consentite (None per illimitato)
        """
        self._model = model
        self._agentic_prompt = agentic_prompt
        self._debug = debug
        self._max_iter = max_iter
        self._tools = {}


    def register_tools(self, tools: List[Any]) -> None:
        for tool in tools:
            self._tools[tool.__name__] = tool

    def _get_tools(self) -> str:
        """Genera la stringa descrittiva degli strumenti disponibili.
        
        Returns
        -------
        str
            Stringa formattata con la descrizione di tutti gli strumenti disponibili,
            uno per riga con prefisso " - "
        """
        if not self._tools:
            return ""
        
        tools = []
        for tool_name, tool_func in self._tools.items():
            tools.append(f" - {self._encode_tool(tool_func)}")
        return "\n".join(tools)

    def _get_base_messages(self, agent_type: str, query: str) -> List[Dict[str, Any]]:
        """Genera i messaggi base per l'agente specifico.
        
        Parameters
        ----------
        agent_type : str
            Tipo di agente per determinare il prompt da utilizzare
        query : str
            Query dell'utente da includere nel prompt
        
        Returns
        -------
        List[Dict[str, Any]]
            Lista dei messaggi base per l'agente, inclusi il prompt e la query
        """
        tools = self._get_tools()
        prompt_id = (f"monoai/agents/prompts/{agent_type}.prompt" 
                    if self._agentic_prompt is None else self._agentic_prompt)
        
        prompt = Prompt(
            prompt_id=prompt_id,
            prompt_data={"query": query, "available_tools": tools}
        )
        
        return [prompt.as_dict()]

    def _debug_print(self, content: str) -> None:
        """Stampa debug se abilitato.
        
        Parameters
        ----------
        content : str
            Contenuto da stampare in modalità debug
        """
        if self._debug:
            print(content)
            print("-------")

    def _execute_model_step(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Esegue un passo del modello e restituisce la risposta.
        
        Parameters
        ----------
        messages : List[Dict[str, Any]]
            Lista dei messaggi da inviare al modello
        
        Returns
        -------
        Dict[str, Any]
            Risposta del modello nel formato standard OpenAI
        """
        
        resp = self._model._execute(messages)
        return resp["choices"][0]["message"]

    def _create_base_response(self, query: str) -> Dict[str, Any]:
        """Crea la struttura base della risposta.
        
        Parameters
        ----------
        query : str
            Query originale dell'utente
        
        Returns
        -------
        Dict[str, Any]
            Dizionario con la struttura base della risposta:
            - prompt: Query originale
            - iterations: Lista vuota per le iterazioni
        """
        return {"prompt": query, "iterations": []}

    def _handle_final_answer(self, iteration: Dict[str, Any], response: Dict[str, Any]) -> bool:
        """Gestisce una risposta finale, restituisce True se è la fine.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Iterazione corrente contenente potenzialmente una risposta finale
        response : Dict[str, Any]
            Dizionario della risposta da aggiornare
        
        Returns
        -------
        bool
            True se è stata trovata una risposta finale, False altrimenti
        
        Notes
        -----
        Questo metodo modifica direttamente l'oggetto response passato come parametro.
        """
        if "final_answer" in iteration:
            response["iterations"].append(iteration)
            response["response"] = iteration["final_answer"]
            return True
        return False

    def _handle_tool_action(self, iteration: Dict[str, Any], response: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
        """Gestisce un'azione di tool.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Iterazione corrente contenente l'azione del tool
        response : Dict[str, Any]
            Dizionario della risposta da aggiornare
        messages : List[Dict[str, Any]]
            Lista dei messaggi da aggiornare con l'osservazione
        
        Notes
        -----
        Questo metodo modifica direttamente gli oggetti response e messages passati come parametri.
        """
        if "action" in iteration and iteration["action"].get("name"):
            tool_call = iteration["action"]
            tool_result = self._call_tool(tool_call)
            iteration["observation"] = tool_result
            response["iterations"].append(iteration)
            
            msg = json.dumps({"observation": tool_result})
            messages.append({"type": "user", "content": msg})

    def _handle_default(self, iteration: Dict[str, Any], response: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
        """Gestisce il caso default per le iterazioni non gestite.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Iterazione corrente da gestire
        response : Dict[str, Any]
            Dizionario della risposta da aggiornare
        messages : List[Dict[str, Any]]
            Lista dei messaggi da aggiornare
        
        Notes
        -----
        Questo metodo modifica direttamente gli oggetti response e messages passati come parametri.
        """
        response["iterations"].append(iteration)
        messages.append({"type": "user", "content": json.dumps(iteration)})

    def start(self, query: str) -> Dict[str, Any]:
        """Metodo astratto per avviare il loop agentico.
        
        Parameters
        ----------
        query : str
            Query dell'utente da processare
        
        Returns
        -------
        Dict[str, Any]
            Risposta dell'agente con le iterazioni e il risultato finale
        
        Raises
        ------
        NotImplementedError
            Questo metodo deve essere implementato dalle classi figlie
        """
        raise NotImplementedError


class FunctionCallingAgenticLoop(_AgenticLoop, _FunctionCallingMixin):
    """Agente che utilizza function calling OpenAI.
    
    Questo agente implementa un loop che utilizza il sistema di function calling
    nativo di OpenAI, permettendo al modello di chiamare direttamente le funzioni
    disponibili senza parsing manuale delle risposte.
    
    Attributes
    ----------
    _model : Any
        Modello OpenAI con supporto per function calling
    _tools : Dict[str, Any]
        Strumenti disponibili per l'agente
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Avvia il loop agentico utilizzando function calling.
        
        Parameters
        ----------
        query : str
            Query dell'utente da processare
        
        Returns
        -------
        Dict[str, Any]
            Risposta dell'agente con:
            - prompt: Query originale
            - iterations: Lista delle chiamate di tool eseguite
            - response: Risposta finale del modello
        """
        self._model._add_tools(list(self._tools.values()))
        messages = [{"type": "user", "content": query}]
        response = self._create_base_response(query)
        
        while True:
            resp = self._execute_model_step(messages)
            messages.append(resp)
            content = resp["content"]

            self._debug_print(content)

            if content is not None:
                response["response"] = content
                break
            
            if resp.get("tool_calls"):
                for tool_call in resp["tool_calls"]:
                    tool_result = self._call_tool(tool_call)
                    response["iterations"].append({
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                        "result": tool_result["content"]
                    })
                    messages.append(tool_result)
        
        return response


class _BaseReactLoop(_AgenticLoop, _ReactMixin):
    """Classe base per tutti gli agenti React-style.
    
    Questa classe implementa il loop standard per gli agenti che utilizzano
    un approccio React-style, dove il modello produce risposte JSON strutturate
    che vengono parse e gestite iterativamente.
    
    Attributes
    ----------
    _max_iter : Optional[int]
        Numero massimo di iterazioni consentite
    """
    
    def _run_react_loop(self, query: str, agent_type: str, 
                        custom_handlers: Optional[Dict[str, callable]] = None) -> Dict[str, Any]:
        """Esegue il loop React standard.
        
        Parameters
        ----------
        query : str
            Query dell'utente da processare
        agent_type : str
            Tipo di agente per determinare il prompt da utilizzare
        custom_handlers : Optional[Dict[str, callable]], optional
            Dizionario di handler personalizzati per gestire tipi specifici
            di iterazioni. Le chiavi sono i campi nell'iterazione, i valori
            sono funzioni che gestiscono quelle iterazioni.
        
        Returns
        -------
        Dict[str, Any]
            Risposta dell'agente con:
            - prompt: Query originale
            - iterations: Lista delle iterazioni processate
            - response: Risposta finale (se presente)
        
        Notes
        -----
        Questo metodo gestisce automaticamente:
        - Risposte finali (final_answer)
        - Azioni di tool (action)
        - Casi personalizzati tramite custom_handlers
        - Casi default per iterazioni non gestite
        - Gestione degli errori JSON
        """
        messages = self._get_base_messages(agent_type, query)
        current_iter = 0
        response = self._create_base_response(query)
        
        # Handler personalizzati per casi speciali
        custom_handlers = custom_handlers or {}

        while True:
            if self._max_iter is not None and current_iter >= self._max_iter:
                break
            
            resp = self._execute_model_step(messages)
            messages.append(resp)
            content = resp["content"]

            self._debug_print(content)

            if content is not None:
                try:
                    iteration = json.loads(content)
                    
                    # Gestione risposta finale
                    if self._handle_final_answer(iteration, response):
                        break
                    
                    # Gestione azioni di tool
                    if "action" in iteration:
                        self._handle_tool_action(iteration, response, messages)
                        continue
                    
                    # Gestione casi personalizzati
                    handled = False
                    for key, handler in custom_handlers.items():
                        if key in iteration and iteration[key] is not None:
                            handler(iteration, response, messages)
                            handled = True
                            break
                    
                    if not handled:
                        self._handle_default(iteration, response, messages)
                        
                except json.JSONDecodeError:
                    # Se non è JSON valido, aggiungi come messaggio utente
                    messages.append({"type": "user", "content": content})

            current_iter += 1

        return response


class ReactAgenticLoop(_BaseReactLoop):
    """Agente React standard.
    
    Questo agente implementa il pattern React standard, dove il modello
    produce risposte JSON che vengono parse e gestite iterativamente.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Avvia il loop agentico React.
        
        Parameters
        ----------
        query : str
            Query dell'utente da processare
        
        Returns
        -------
        Dict[str, Any]
            Risposta dell'agente processata tramite il loop React
        """
        return self._run_react_loop(query, "react")


class ReactWithFCAgenticLoop(_AgenticLoop, _FunctionCallingMixin):
    """Agente che combina React e Function Calling.
    
    Questo agente combina l'approccio React con il function calling nativo
    di OpenAI, permettendo una gestione ibrida delle chiamate di tool.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Avvia il loop agentico ibrido React + Function Calling.
        
        Parameters
        ----------
        query : str
            Query dell'utente da processare
        
        Returns
        -------
        Dict[str, Any]
            Risposta dell'agente (da implementare)
        
        Notes
        -----
        TODO: Implementare combinazione di React e Function Calling
        """
        # TODO: Implementare combinazione di React e Function Calling
        pass


class ProgrammaticAgenticLoop(_BaseReactLoop):
    """Agente programmatico.
    
    Questo agente implementa un approccio programmatico dove il modello
    produce codice o istruzioni strutturate che vengono eseguite.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Avvia il loop agentico programmatico.
        
        Parameters
        ----------
        query : str
            Query dell'utente da processare
        
        Returns
        -------
        Dict[str, Any]
            Risposta dell'agente processata tramite il loop programmatico
        """
        return self._run_react_loop(query, "programmatic")


class PlanAndExecuteAgenticLoop(_BaseReactLoop):
    """Agente plan-and-execute.
    
    Questo agente implementa il pattern plan-and-execute, dove il modello
    prima pianifica le azioni e poi le esegue sequenzialmente.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Avvia il loop agentico plan-and-execute.
        
        Parameters
        ----------
        query : str
            Query dell'utente da processare
        
        Returns
        -------
        Dict[str, Any]
            Risposta dell'agente processata tramite il loop plan-and-execute
        """
        return self._run_react_loop(query, "plan_and_execute")


class ReflexionAgenticLoop(_BaseReactLoop):
    """Agente con riflessione.
    
    Questo agente implementa il pattern reflexion, dove il modello
    riflette sulle proprie azioni e decisioni per migliorare le performance.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Avvia il loop agentico con riflessione.
        
        Parameters
        ----------
        query : str
            Query dell'utente da processare
        
        Returns
        -------
        Dict[str, Any]
            Risposta dell'agente processata tramite il loop reflexion
        """
        return self._run_react_loop(query, "reflexion")


class SelfAskAgenticLoop(_BaseReactLoop):
    """Agente self-ask.
    
    Questo agente implementa il pattern self-ask, dove il modello
    si pone domande a se stesso per guidare il processo di ragionamento.
    """
    
    def start(self, query: str) -> Dict[str, Any]:
        """Avvia il loop agentico self-ask.
        
        Parameters
        ----------
        query : str
            Query dell'utente da processare
        
        Returns
        -------
        Dict[str, Any]
            Risposta dell'agente processata tramite il loop self-ask
        """
        return self._run_react_loop(query, "self_ask")


class SelfAskWithSearchLoop(_BaseReactLoop):
    """Agente self-ask con ricerca web.
    
    Questo agente estende il pattern self-ask con la capacità di
    eseguire ricerche web per ottenere informazioni aggiuntive.
    
    Attributes
    ----------
    _handle_search_query : callable
        Metodo per gestire le query di ricerca web
    """
    
    def _handle_search_query(self, iteration: Dict[str, Any], response: Dict[str, Any], messages: List[Dict[str, Any]]) -> None:
        """Gestisce le query di ricerca web.
        
        Parameters
        ----------
        iteration : Dict[str, Any]
            Iterazione corrente contenente la query di ricerca
        response : Dict[str, Any]
            Dizionario della risposta da aggiornare
        messages : List[Dict[str, Any]]
            Lista dei messaggi da aggiornare con i risultati della ricerca
        
        Notes
        -----
        Questo metodo modifica direttamente gli oggetti response e messages passati come parametri.
        Utilizza il motore di ricerca Tavily per eseguire le ricerche web.
        """
        from ..tools.websearch import search_web
        
        query = iteration["search_query"]
        result = search_web(query, engine="tavily")["text"]
        iteration["search_result"] = result
        
        msg = json.dumps({"query_results": result})
        messages.append({"type": "user", "content": msg})
        response["iterations"].append(iteration)
    
    def start(self, query: str) -> Dict[str, Any]:
        """Avvia il loop agentico self-ask con ricerca web.
        
        Parameters
        ----------
        query : str
            Query dell'utente da processare
        
        Returns
        -------
        Dict[str, Any]
            Risposta dell'agente processata tramite il loop self-ask con ricerca web
        
        Notes
        -----
        Questo agente utilizza un handler personalizzato per gestire le query di ricerca
        web, permettendo al modello di ottenere informazioni aggiornate durante il processo.
        """
        custom_handlers = {"search_query": self._handle_search_query}
        return self._run_react_loop(query, "self_ask_with_search", custom_handlers)

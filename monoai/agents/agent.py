from ..models import Model
from .agentic_loop import (
    FunctionCallingAgenticLoop, 
    ReactAgenticLoop, 
    ReactWithFCAgenticLoop,
    PlanAndExecuteAgenticLoop,
    ProgrammaticAgenticLoop,
    ReflexionAgenticLoop,
    SelfAskAgenticLoop,
    SelfAskWithSearchLoop,
    _AgenticLoop
)

from ..prompts import Prompt


class Agent:
    """Agente AI che implementa diversi paradigmi di ragionamento.
    
    Questa classe fornisce un'interfaccia unificata per creare e utilizzare
    agenti AI con diversi paradigmi di ragionamento. L'agente può essere
    configurato per utilizzare function calling, React, plan-and-execute,
    e altri approcci agentici, oppure un paradigma personalizzato.
    
    Attributes
    ----------
    _model : Model
        Modello AI da utilizzare per l'esecuzione
    _loop : Any
        Istanza del loop agentico specifico per il paradigma scelto
    """
    
    def __init__(self, model: Model, tools=None, paradigm="function_calling", 
                 agent_prompt=None, debug=False, max_iter=None):
        """Inizializza l'agente con il modello e la configurazione specificata.
        
        Parameters
        ----------
        model : Model
            Modello AI da utilizzare per l'esecuzione dell'agente
        tools : list, optional
            Lista degli strumenti disponibili per l'agente. Default è None.
        paradigm : str or _AgenticLoop, optional
            Paradigma di ragionamento da utilizzare. Può essere:
            
            **Stringhe predefinite:**
            - "function_calling": Utilizza OpenAI function calling
            - "react": Approccio React standard
            - "react_with_function_calling": Combina React e function calling
            - "plan-and-execute": Paradigma plan-and-execute
            - "programmatic": Approccio programmatico
            - "reflexion": Paradigma con riflessione
            - "self_ask": Paradigma self-ask
            - "self_ask_with_search": Paradigma self-ask con ricerca web
            
            **Oggetto personalizzato:**
            - Un'istanza di una classe derivata da _AgenticLoop
            
            Default è "function_calling".
        agent_prompt : str, optional
            Prompt personalizzato per l'agente. Se None, viene utilizzato
            il prompt di default per il paradigma scelto. Default è None.
        debug : bool, optional
            Flag per abilitare la stampa di debug durante l'esecuzione.
            Default è False.
        max_iter : int, optional
            Numero massimo di iterazioni consentite per l'agente.
            Se None, non ci sono limiti. Default è None.
        
        Raises
        ------
        ValueError
            Se il paradigma specificato non è supportato o non è valido
        TypeError
            Se viene passato un oggetto personalizzato che non deriva da _AgenticLoop
        """
        self._model = model

        # Gestione paradigma personalizzato
        if isinstance(paradigm, _AgenticLoop):
            # Verifica che l'oggetto personalizzato sia valido
            if not hasattr(paradigm, 'start') or not callable(paradigm.start):
                raise TypeError("Il paradigma personalizzato deve avere un metodo 'start' callable")
            self._loop = paradigm
        else:
            # Paradigmi predefiniti
            loop_kwargs = self._model, agent_prompt, debug, max_iter
            
            if paradigm == "function_calling":
                self._loop = FunctionCallingAgenticLoop(*loop_kwargs)
            elif paradigm == "react":
                self._loop = ReactAgenticLoop(*loop_kwargs)
            elif paradigm == "react_with_function_calling":
                self._loop = ReactWithFCAgenticLoop(*loop_kwargs)
            elif paradigm == "plan-and-execute":
                self._loop = PlanAndExecuteAgenticLoop(*loop_kwargs)
            elif paradigm == "programmatic":
                self._loop = ProgrammaticAgenticLoop(*loop_kwargs)
            elif paradigm == "reflexion":
                self._loop = ReflexionAgenticLoop(*loop_kwargs)
            elif paradigm == "self_ask":
                self._loop = SelfAskAgenticLoop(*loop_kwargs)
            elif paradigm == "self_ask_with_search":
                self._loop = SelfAskWithSearchLoop(*loop_kwargs)
            else:
                raise ValueError(f"Paradigma '{paradigm}' non supportato. "
                               f"Paradigmi disponibili: function_calling, react, "
                               f"react_with_function_calling, plan-and-execute, "
                               f"programmatic, reflexion, self_ask, self_ask_with_search, "
                               f"oppure un oggetto personalizzato derivato da _AgenticLoop")
        
        if tools is not None:
            self._loop.register_tools(tools)
        
    def run(self, prompt: str | Prompt):
        """Esegue l'agente con il prompt specificato.
        
        Parameters
        ----------
        prompt : str
            Prompt o query dell'utente da processare
        
        Returns
        -------
        Dict[str, Any]
            Risposta dell'agente che include:
            - prompt: Query originale
            - iterations: Lista delle iterazioni processate
            - response: Risposta finale (se presente)
        
        Notes
        -----
        Questo metodo delega l'esecuzione al loop agentico specifico
        configurato durante l'inizializzazione. Il comportamento esatto
        dipende dal paradigma scelto (predefinito o personalizzato).
        """
        
        return self._loop.start(prompt)
C'è ancora il problema che l'agente ReAct restituisce thoguht e action in un'unica risposta.
Il formato dei tool non è strutturato bene


  return self.__pydantic_serializer__.to_python(
/Users/giuseppegullo/Progetti/COICOI/venv/lib/python3.13/site-packages/pydantic/_internal/_repr.py:63: RuntimeWarning: coroutine '_FunctionCallingMixin._call_mcp_tool.<locals>.ensure_connected_and_call' was never awaited
  return join_str.join(repr(v) if a is None else f'{a}={v!r}' for a, v in self.__repr_args__())
RuntimeWarning: Enable tracemalloc to get the object allocation traceback
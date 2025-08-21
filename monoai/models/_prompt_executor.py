from typing import Dict, Union, AsyncGenerator
from ..prompts.prompt import Prompt
from ..prompts.prompt_chain import PromptChain
from ..prompts.iterative_prompt import IterativePrompt
from ..tools._tool_parser import ToolParser
from ..conf import Conf

class PromptExecutorMixin:
    """Mixin class to handle prompt execution."""

    
    async def _execute_stream(self, prompt: Union[str, Prompt, PromptChain]) -> AsyncGenerator[Dict, None]:
        """
        Execute a prompt asynchronously with streaming.
        
        Args:
            prompt: The prompt to execute (string, Prompt, or PromptChain)
            
        Yields:
            Dictionary containing streaming response chunks
        """

        # Handle RAG if available
        if hasattr(self, '_rag') and self._rag:
            response = self._rag.query(prompt)
            if len(response["documents"]) > 0:
                documents = response.get('documents', [])
                documents = [s for doc_list in documents for s in doc_list]
                documents = '\n'.join(documents)
                prompt += Conf()["default_prompt"]["rag"] + documents

        # Handle different prompt types
        if isinstance(prompt, PromptChain):
            async for chunk in self._execute_chain_stream(prompt):
                yield chunk
        elif isinstance(prompt, IterativePrompt):
            async for chunk in self._execute_iterative_stream(prompt):
                yield chunk
        elif isinstance(prompt, Prompt):
            async for chunk in self._completion_stream(str(prompt), response_type=prompt.response_type):
                yield chunk
        else:
            async for chunk in self._completion_stream(prompt):
                yield chunk
    
    def _execute(self, prompt: Union[str, Prompt, PromptChain]) -> Dict:
        """
        Execute a prompt synchronously.
        
        Args:
            prompt: The prompt to execute (string, Prompt, or PromptChain)
            agent: The agent to use for execution
            
        Returns:
            Dictionary containing the response
        """

        if self._rag:
            response = self._rag.query(prompt)
            if len(response["documents"])>0:
                documents = response.get('documents', [])
                documents = [s for doc_list in documents for s in doc_list]
                documents = '\n'.join(documents)
                prompt += Conf()["default_prompt"]["rag"] + documents

        if isinstance(prompt, PromptChain):
            return self._execute_chain(prompt)
        elif isinstance(prompt, IterativePrompt):
            return self._execute_iterative(prompt)
        elif isinstance(prompt, Prompt):
            return self._completion(str(prompt), response_type=prompt.response_type)
        else:
            return self._completion(prompt)

    async def _execute_chain_async(self, chain: PromptChain) -> Dict:
        """
        Execute a prompt chain asynchronously.
        
        Args:
            chain: The prompt chain to execute
            agent: The agent to use for execution
            
        Returns:
            Dictionary containing the final response
        """
        response = None
        for i in range(chain._size):
            current_prompt = chain._format(i, response.output if response else None)
            response = self._completion(current_prompt)
        return response

    async def _execute_chain_stream(self, chain: PromptChain) -> AsyncGenerator[Dict, None]:
        """
        Execute a prompt chain asynchronously with streaming.
        
        Args:
            chain: The promptChain to execute
            
        Yields:
            Streaming response chunks from the final prompt in the chain
        """
        response = None
        for i in range(chain._size):
            current_prompt = chain._format(i, response.output if response else None)
            if i == chain._size - 1:  # Last prompt in chain
                async for chunk in self._completion_stream(current_prompt):
                    yield chunk
            else:  # Execute non-streaming for intermediate prompts
                response = self._completion(current_prompt)

    def _execute_chain(self, chain: PromptChain) -> Dict:
        """
        Execute a prompt chain synchronously.
        
        Args:
            chain: The prompt chain to execute
            agent: The agent to use for execution
            
        Returns:
            Dictionary containing the final response
        """
        response = None
        for i in range(chain._size):
            current_prompt = chain._format(i, response["choices"][0]["message"]["content"] if response else None)
            response = self._completion(current_prompt)
        return response
    
    def _execute_iterative(self, prompt: IterativePrompt) -> Dict:
        """
        Execute an iterative prompt synchronously.
        
        Args:
            prompt: The iterative prompt to execute
            agent: The agent to use for execution
            
        Returns:
            Dictionary containing the final response
        """
        response = ""
        memory = ""
        for i in range(prompt._size):
            if i > 0 and prompt._has_memory:
                if prompt._retain_all:
                    memory += current_response
                else:
                    memory = current_response   
                current_prompt = prompt._format(i, memory)
            else:
                current_prompt = prompt._format(i)
            current_response = self._completion(current_prompt)

            response += current_response
        return response

    async def _execute_iterative_stream(self, prompt: IterativePrompt) -> AsyncGenerator[Dict, None]:
        """
        Execute an iterative prompt asynchronously with streaming.
        
        Args:
            prompt: The iterative prompt to execute
            
        Yields:
            Streaming response chunks from the final iteration
        """
        response = ""
        memory = ""
        for i in range(prompt._size):
            if i > 0 and prompt._has_memory:
                if prompt._retain_all:
                    memory += current_response
                else:
                    memory = current_response   
                current_prompt = prompt._format(i, memory)
            else:
                current_prompt = prompt._format(i)
            
            if i == prompt._size - 1:  # Last iteration
                async for chunk in self._completion_stream(current_prompt):
                    yield chunk
            else:  # Execute non-streaming for intermediate iterations
                current_response = self._completion(current_prompt)
                response += current_response

    def _completion(self, prompt: str|list, response_type: str = None) -> Dict:
        
        from pydantic import BaseModel

        class Response(BaseModel):
            response: response_type

        if response_type!=None:
            response_type = Response

        self._disable_logging()

        url = None
        model = self.provider+"/"+self.model
        
        if hasattr(self, "url") and self.url != None:
            url = self.url+"/v"+str(self.version)
            model = "hosted_vllm/"+model
        
        tools = None

        if hasattr(self, "_tools"):
            tools = []
            tp = ToolParser()
            for tool in self._tools:
                tools.append(tp.parse(tool))      

        if isinstance(prompt, str):
            messages = [{ "content": prompt,"role": "user"}]
        else:
            messages = prompt

        from litellm import completion

        return completion(model=model, 
                          messages=messages, 
                          response_format=response_type,
                          base_url = url,
                          tools=tools,
                          max_tokens=self._max_tokens)


    async def _completion_stream(self, prompt: str|list, response_type: str = None) -> AsyncGenerator[Dict, None]:
        """
        Execute a streaming completion.
        
        Args:
            prompt: The prompt to execute
            response_type: Optional response type
            
        Yields:
            Streaming response chunks
        """
        url = None
        model = self.provider+"/"+self.model
        if hasattr(self, "url") and self.url != None:
            url = self.url+"/v"+str(self.version)
            model = "hosted_vllm/"+self.model
        
        if isinstance(prompt, str):
            messages = [{ "content": prompt,"role": "user"}]
        else:
            messages = prompt

        self._disable_logging()
        from litellm import acompletion

        response = await acompletion(
            model=model, 
            messages=messages, 
            response_format=response_type,
            base_url=url,
            stream=True,
            max_tokens=self._max_tokens
        )
        
        async for chunk in response:
            yield chunk
            

    def _disable_logging(self):
        import logging
        loggers = [
            "LiteLLM Proxy",
            "LiteLLM Router",
            "LiteLLM",
            "httpx"
        ]

        for logger_name in loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.CRITICAL + 1) 


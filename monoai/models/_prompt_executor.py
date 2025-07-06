from typing import Dict, Union
from ..prompts.prompt import Prompt
from ..prompts.prompt_chain import PromptChain
from ..prompts.iterative_prompt import IterativePrompt
from litellm import completion

class PromptExecutorMixin:
    """Mixin class to handle prompt execution."""
    
    async def _execute_async(self, prompt: Union[str, Prompt, PromptChain]) -> Dict:
        """
        Execute a prompt asynchronously.
        
        Args:
            prompt: The prompt to execute (string, Prompt, or PromptChain)
            agent: The agent to use for execution
            
        Returns:
            Dictionary containing the response
        """
                
        if isinstance(prompt, PromptChain):
            return await self._execute_chain_async(prompt)
        return self._completion(prompt)
    
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
            print(response)
            if len(response["documents"])>0:
                documents = response.get('documents', [])
                documents = [s for doc_list in documents for s in doc_list]
                documents = '\n'.join(documents)
                prompt += "\nUse also the following information to answer the question: " + documents

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

    def _completion(self, prompt: str, response_type: str = None) -> Dict:
        
        from pydantic import BaseModel

        class Response(BaseModel):
            response: response_type

        if response_type!=None:
            response_type = Response
        
        messages = [{ "content": prompt,"role": "user"}]
        return completion(model=self.provider+"/"+self.model, 
                          messages=messages, 
                          response_format=response_type,
                          max_tokens=self._max_tokens)

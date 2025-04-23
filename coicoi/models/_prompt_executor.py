from typing import Dict, Union
from ..prompts.prompt import Prompt
from ..prompts.prompt_chain import PromptChain
from ..prompts.iterative_prompt import IterativePrompt
from pydantic_ai import Agent

class PromptExecutorMixin:
    """Mixin class to handle prompt execution."""
    
    async def _execute_async(self, prompt: Union[str, Prompt, PromptChain], agent: Agent) -> Dict:
        """
        Execute a prompt asynchronously.
        
        Args:
            prompt: The prompt to execute (string, Prompt, or PromptChain)
            agent: The agent to use for execution
            
        Returns:
            Dictionary containing the response
        """
        if isinstance(prompt, PromptChain):
            return await self._execute_chain_async(prompt, agent)
        return await agent.run(prompt)

    def _execute(self, prompt: Union[str, Prompt, PromptChain], agent: Agent) -> Dict:
        """
        Execute a prompt synchronously.
        
        Args:
            prompt: The prompt to execute (string, Prompt, or PromptChain)
            agent: The agent to use for execution
            
        Returns:
            Dictionary containing the response
        """
        if isinstance(prompt, PromptChain):
            return self._execute_chain(prompt, agent)
        elif isinstance(prompt, IterativePrompt):
            return self._execute_iterative(prompt, agent)
        elif isinstance(prompt, Prompt):
            return agent.run_sync(str(prompt), output_type=prompt.response_type).output
        else:
            return agent.run_sync(str(prompt)).output

    async def _execute_chain_async(self, chain: PromptChain, agent: Agent) -> Dict:
        """
        Execute a prompt chain asynchronously.
        
        Args:
            chain: The prompt chain to execute
            agent: The agent to use for execution
            
        Returns:
            Dictionary containing the final response
        """
        response = None
        for i in range(chain.size):
            current_prompt = chain.format(i, response.output if response else None)
            response = await agent.run(current_prompt)
        return response

    def _execute_chain(self, chain: PromptChain, agent: Agent) -> Dict:
        """
        Execute a prompt chain synchronously.
        
        Args:
            chain: The prompt chain to execute
            agent: The agent to use for execution
            
        Returns:
            Dictionary containing the final response
        """
        response = None
        for i in range(chain.size):
            current_prompt = chain.format(i, response.output if response else None)
            response = agent.run_sync(current_prompt)
        return response.output
    
    def _execute_iterative(self, prompt: IterativePrompt, agent: Agent) -> Dict:
        """
        Execute an iterative prompt synchronously.
        
        Args:
            prompt: The iterative prompt to execute
            agent: The agent to use for execution
            
        Returns:
            Dictionary containing the final response
        """
        response = ""
        for i in range(prompt.size):
            if i > 0 and prompt.has_memory:
                current_prompt = prompt.format(i, current_response)
            else:
                current_prompt = prompt.format(i)
            current_response = agent.run_sync(current_prompt).output
            response += current_response
        return response


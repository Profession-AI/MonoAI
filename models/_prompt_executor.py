from typing import Dict, Union
from prompts.prompt import Prompt
from prompts.prompt_chain import PromptChain
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
        return agent.run_sync(prompt)

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
            current_prompt = chain.format(i, response.data if response else None)
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
            current_prompt = chain.format(i, response.data if response else None)
            response = agent.run_sync(current_prompt)
        return response.data
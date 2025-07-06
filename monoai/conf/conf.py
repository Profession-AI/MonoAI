import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from yaml.parser import ParserError

class Conf:
    """
    A singleton class for managing configuration in a programmatic way.
    Configuration defined here overrides the configuration in monoai.yaml file.

    Examples
    --------
    ```
    from monoai.conf import Conf
    
    # override the base model
    Conf()["base_model"] = {
        "provider": "openai",
        "model": "gpt-4o-mini"
    }
    
    # set prompts path programmatically
    Conf()["prompts_path"] = "prompts"
    ```
    """
    
    _instance = None
    _DEFAULT_CONFIG = {
        'keysfile_path': "providers.keys",
        'prompts_path': ""
    }
    
    def __new__(cls):
        """
        Create or return the singleton instance.

        Returns
        -------
        Config
            The singleton instance
        """
        if cls._instance is None:
            cls._instance = super(Conf, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """
        Initialize the Config instance.
        
        Loads and validates the configuration file, setting up defaults
        and performing environment variable interpolation.
        """
        self._config_path = Path('monoai.yaml')
        self._config = self._DEFAULT_CONFIG.copy()
        self._load_config()
    
    def _load_config(self) -> None:
        """
        Load configuration from the YAML file.
        
        Handles file reading, YAML parsing, environment variable interpolation,
        and configuration validation.

        Raises
        ------
        FileNotFoundError
            If the configuration file doesn't exist
        yaml.ParserError
            If the YAML syntax is invalid
        ValueError
            If the configuration structure is invalid
        """
        try:
            if self._config_path.exists():
                with open(self._config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                if file_config:
                    self._merge_config(self._config, file_config)            
            
        except ParserError as e:
            raise ValueError(f"Invalid YAML syntax in {self._config_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    def _merge_config(self, base: Dict, override: Dict) -> None:
        """
        Recursively merge override configuration into base configuration.

        Parameters
        ----------
        base : Dict
            The base configuration dictionary
        override : Dict
            The override configuration dictionary
        """
        for key, value in override.items():
            if (
                key in base and 
                isinstance(base[key], dict) and 
                isinstance(value, dict)
            ):
                self._merge_config(base[key], value)
            else:
                base[key] = value
        
        
    def __getitem__(self, key: str) -> Any:
        return self._config[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        self._config[key] = value
    
    
    
    
"""
Config singleton class for managing configuration from coiconf.yaml.

This module provides a singleton ConfigManager class that handles loading,
validating, and accessing configuration settings from a YAML file. It supports
environment variable interpolation, default values, and configuration validation.

Examples
--------
Basic usage:
    >>> config = Config()
    >>> openai_key = config.get('api_keys.openai')
    >>> model_name = config.get('models.default')

With default value:
    >>> debug_mode = config.get('settings.debug', default=False)

Checking existence:
    >>> if config.has('api_keys.anthropic'):
    ...     anthropic_key = config.get('api_keys.anthropic')
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from yaml.parser import ParserError

class Config:
    """
    A singleton class for managing application configuration from YAML.
    
    This class implements the Singleton pattern to provide centralized access
    to configuration settings. It handles loading and parsing the coiconf.yaml
    file, with support for environment variable interpolation and validation.

    Attributes
    ----------
    _instance : Config
        The singleton instance of the class
    _config : Dict
        The loaded configuration dictionary
    _config_path : Path
        Path to the configuration file
    
    Examples
    --------
    Basic configuration access:
        >>> config = Config()
        >>> model = config.get('models.default')
        >>> print(model)  # 'gpt-4'
    """
    
    _instance = None
    _DEFAULT_CONFIG = {
        'keysfile_path': "providers.keys",
        'prompts_path': "",
        'base_model':{
            "provider": "openai",
            "model": "gpt-4o-mini"
        }
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
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """
        Initialize the Config instance.
        
        Loads and validates the configuration file, setting up defaults
        and performing environment variable interpolation.
        """
        self._config_path = Path('coiconf.yaml')
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
        
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its dot-notation path.

        Parameters
        ----------
        key_path : str
            Dot-notation path to the configuration value
        default : Any, optional
            Default value if the key doesn't exist

        Returns
        -------
        Any
            The configuration value or default

        Examples
        --------
        Get with default:
            >>> timeout = config.get('settings.timeout', default=30)
            >>> print(timeout)  # 30
        """
        try:
            value = self._config
            for key in key_path.split('.'):
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def has(self, key_path: str) -> bool:
        """
        Check if a configuration key exists.

        Parameters
        ----------
        key_path : str
            Dot-notation path to check

        Returns
        -------
        bool
            True if the key exists, False otherwise

        Examples
        --------
        Check key existence:
            >>> if config.has('api_keys.openai'):
            ...     key = config.get('api_keys.openai')
        """
        try:
            value = self._config
            for key in key_path.split('.'):
                value = value[key]
            return True
        except (KeyError, TypeError):
            return False
    
    def reload(self) -> None:
        """
        Reload the configuration from file.

        This method reloads the configuration file, useful when the file
        has been modified externally.

        Examples
        --------
        Reload configuration:
            >>> config.reload()
        """
        self._config = self._DEFAULT_CONFIG.copy()
        self._load_config()
    
    def get_all(self) -> Dict:
        """
        Get the complete configuration dictionary.

        Returns
        -------
        Dict
            The complete configuration dictionary

        Examples
        --------
        Get all config:
            >>> all_config = config.get_all()
            >>> print(all_config)
        """
        return self._config.copy()
    
    
if __name__ == "__main__":
    config = Config()
    print(config.get_all())

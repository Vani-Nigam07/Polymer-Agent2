"""
Config Management for TransPolymer
==================================

This module provides utilities for loading and managing configuration files.
It handles both absolute and relative paths correctly.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from .path_utils import PathManager, resolve_path


class ConfigManager:
    """
    Manages configuration loading and validation for TransPolymer.
    
    Handles loading YAML config files with proper path resolution.
    """
    
    _configs_cache: Dict[str, Dict[str, Any]] = {}
    
    @classmethod
    def load_config(cls, config_path: Union[str, Path], use_cache: bool = True) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            config_path: Path to config file (can be relative or absolute)
            use_cache: Whether to cache loaded configs
        
        Returns:
            Dictionary with configuration
        
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        config_path = resolve_path(config_path)
        
        # Check cache
        if use_cache and str(config_path) in cls._configs_cache:
            return cls._configs_cache[str(config_path)]
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        
        # Resolve paths in config
        config = cls._resolve_config_paths(config)
        
        if use_cache:
            cls._configs_cache[str(config_path)] = config
        
        return config
    
    @classmethod
    def _resolve_config_paths(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve any relative paths in the configuration.
        
        Looks for keys ending with '_path', '_file', 'model_path', etc.
        """
        path_keys = [
            'path', 'file', 'model_path', 'save_path', 'checkpoint_path',
            'data_path', 'config_path', 'vocab_file', 'train_file', 'test_file'
        ]
        
        resolved_config = {}
        for key, value in config.items():
            if value is None:
                resolved_config[key] = value
            elif isinstance(value, str) and any(key.endswith(pk) or key.lower().endswith(pk) for pk in path_keys):
                # This looks like a path - resolve it
                try:
                    resolved_config[key] = str(resolve_path(value))
                except Exception:
                    # If resolution fails, keep original
                    resolved_config[key] = value
            elif isinstance(value, dict):
                resolved_config[key] = cls._resolve_config_paths(value)
            elif isinstance(value, list):
                resolved_config[key] = [
                    cls._resolve_config_paths(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                resolved_config[key] = value
        
        return resolved_config
    
    @classmethod
    def clear_cache(cls):
        """Clear the configuration cache."""
        cls._configs_cache.clear()


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function for loading a configuration file.
    
    See ConfigManager.load_config for details.
    """
    return ConfigManager.load_config(config_path)

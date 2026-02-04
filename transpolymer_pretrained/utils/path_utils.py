"""
Path Utilities for TransPolymer
===============================

This module provides utilities for resolving paths relative to the TransPolymer package,
regardless of where the module is imported from.

All paths are resolved relative to the package root directory.
"""

from pathlib import Path
from typing import Union, Optional


class PathManager:
    """
    Manages all paths for the TransPolymer package.
    
    This ensures that all relative paths work correctly regardless of where
    the module is called from or how it's installed.
    """
    
    # Package root is the parent of this utils directory
    PACKAGE_ROOT = Path(__file__).parent.parent.parent
    
    @classmethod
    def get_package_root(cls) -> Path:
        """Get the TransPolymer package root directory."""
        return cls.PACKAGE_ROOT
    
    @classmethod
    def get_checkpoint_dir(cls) -> Path:
        """Get the checkpoint directory."""
        ckpt_dir = cls.PACKAGE_ROOT / "ckpt"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return ckpt_dir
    
    @classmethod
    def get_config_dir(cls) -> Path:
        """Get the config directory."""
        config_dir = cls.PACKAGE_ROOT / "transpolymer" / "configs"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir
    
    @classmethod
    def get_data_dir(cls) -> Path:
        """Get the data directory."""
        data_dir = cls.PACKAGE_ROOT / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        return data_dir
    
    @classmethod
    def get_runs_dir(cls) -> Path:
        """Get the runs directory for TensorBoard logs."""
        runs_dir = cls.PACKAGE_ROOT / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        return runs_dir
    
    @classmethod
    def resolve_path(cls, path: Union[str, Path], relative_to: Optional[str] = None) -> Path:
        """
        Resolve a path, handling both absolute and relative paths.
        
        If the path is already absolute, it's returned as-is.
        If the path is relative, it's resolved relative to the package root.
        
        Args:
            path: The path to resolve
            relative_to: Optional subdirectory to resolve relative to ('config', 'checkpoint', 'data', 'runs')
        
        Returns:
            The resolved absolute Path
        
        Examples:
            >>> resolve_path('config_finetune.yaml')  # Resolves to package_root/transpolymer/configs/...
            >>> resolve_path('ckpt/model.pt')  # Resolves relative to package root
            >>> resolve_path('/absolute/path/model.pt')  # Returns as-is
        """
        path = Path(path)
        
        # If absolute, return as-is
        if path.is_absolute():
            return path
        
        # If relative, resolve from package root or specified subdirectory
        if relative_to == 'config':
            return cls.get_config_dir() / path
        elif relative_to == 'checkpoint':
            return cls.get_checkpoint_dir() / path
        elif relative_to == 'data':
            return cls.get_data_dir() / path
        elif relative_to == 'runs':
            return cls.get_runs_dir() / path
        else:
            # Default: resolve relative to package root
            return cls.PACKAGE_ROOT / path
    
    @classmethod
    def get_checkpoint_path(cls, filename: str) -> Path:
        """Get full path to a checkpoint file."""
        return cls.get_checkpoint_dir() / filename
    
    @classmethod
    def get_config_path(cls, filename: str) -> Path:
        """Get full path to a config file."""
        return cls.get_config_dir() / filename
    
    @classmethod
    def get_data_path(cls, filename: str) -> Path:
        """Get full path to a data file."""
        return cls.get_data_dir() / filename


# Convenience function
def resolve_path(path: Union[str, Path], relative_to: Optional[str] = None) -> Path:
    """
    Convenience function for resolving paths.
    
    See PathManager.resolve_path for details.
    """
    return PathManager.resolve_path(path, relative_to)

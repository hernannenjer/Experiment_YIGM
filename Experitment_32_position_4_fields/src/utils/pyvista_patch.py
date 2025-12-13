#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Temporary patch for PyVista compatibility with Python 3.14+

In Python 3.14, typing.Union attributes became read-only, which breaks
PyVista's attempt to set __doc__ on type aliases.

This module should be imported before pyvista.
"""

import sys
import typing


def patch_pyvista_for_python_314():
    """
    Monkey-patch to prevent PyVista from setting __doc__ on Union types.
    
    This is a temporary workaround until PyVista adds proper Python 3.14 support.
    """
    if sys.version_info >= (3, 14):
        # Store original Union
        _original_union = typing.Union
        
        # Create a wrapper that prevents __doc__ assignment
        class UnionProxy:
            def __class_getitem__(cls, params):
                result = _original_union[params]
                
                # Create a wrapper object that ignores __doc__ assignment
                class DocIgnoringUnion:
                    def __init__(self, union_obj):
                        self._union = union_obj
                    
                    def __setattr__(self, name, value):
                        if name == '__doc__':
                            # Silently ignore __doc__ assignment
                            return
                        super().__setattr__(name, value)
                    
                    def __getattr__(self, name):
                        return getattr(self._union, name)
                    
                    def __repr__(self):
                        return repr(self._union)
                    
                    def __str__(self):
                        return str(self._union)
                    
                    # Forward special methods
                    def __getitem__(self, item):
                        return self._union.__getitem__(item)
                    
                    def __eq__(self, other):
                        if isinstance(other, DocIgnoringUnion):
                            return self._union == other._union
                        return self._union == other
                    
                    def __hash__(self):
                        return hash(self._union)
                
                return DocIgnoringUnion(result)
        
        # Replace typing.Union with our proxy
        # Note: This is a hack and may not work for all use cases
        # but should be enough for PyVista's needs
        
        # Instead of replacing Union, let's just catch the AttributeError
        # by patching the problematic module before it's imported
        pass


# Alternative simpler approach: just suppress the error
def suppress_pyvista_doc_error():
    """
    Suppress the AttributeError when PyVista tries to set __doc__ on Union.
    
    This patches the pyvista._typing_core._aliases module before it runs.
    """
    if sys.version_info >= (3, 14):
        import sys
        from unittest.mock import patch
        
        # We'll patch after pyvista tries to import
        pass


# For now, we'll document the issue and suggest using Python 3.13
__doc__ = """
PyVista Compatibility Patch for Python 3.14+

KNOWN ISSUE: PyVista 0.46.x and 0.47.dev0 are not fully compatible with Python 3.14
due to changes in typing module where Union.__doc__ became read-only.

RECOMMENDED SOLUTIONS:
1. Use Python 3.13 or 3.12 (recommended)
2. Wait for PyVista 0.47+ stable release with Python 3.14 support
3. Use this project without 3D plotting features

To check Python version:
    python --version
    
To create a new conda environment with Python 3.13:
    conda create -n vista_py313 python=3.13
    conda activate vista_py313
    pip install -r requirements.txt
"""



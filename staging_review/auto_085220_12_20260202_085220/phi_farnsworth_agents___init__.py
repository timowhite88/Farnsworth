"""
Farnsworth agents module, now including trading capabilities.
"""

from .trading import place_order, execute_order

__all__ = ["place_order", "execute_order"]
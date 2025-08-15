"""
Tools are used to extend the capabilities of the model.
"""

from .domain_whois import domain_whois
from .websearch import search_web

__all__ = ["domain_whois", "search_web"]
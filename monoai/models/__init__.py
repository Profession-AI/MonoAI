"""
Models are the core of MonoAI. They are responsible for executing prompts and returning responses.
"""

from .model import Model
from .multi_model import MultiModel
from .collaborative_model import CollaborativeModel
from .image_model import ImageModel

__all__ = ['Model', 'MultiModel', 'CollaborativeModel', 'ImageModel'] 
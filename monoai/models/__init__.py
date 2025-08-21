"""
Models are the core of MonoAI. They are responsible for executing prompts and returning responses.

This package uses lazy loading to avoid importing heavy optional dependencies
at module import time. Classes are imported only when accessed.
"""

__all__ = ['Model', "HostedModel", 'MultiModel', 'CollaborativeModel', 'ImageModel', 'VoiceModel']

def __getattr__(name):
	if name == 'Model':
		from .model import Model
		return Model
	elif name == 'HostedModel':
		from .hosted_model import HostedModel
		return HostedModel
	elif name == 'MultiModel':
		from .multi_model import MultiModel
		return MultiModel
	elif name == 'CollaborativeModel':
		from .collaborative_model import CollaborativeModel
		return CollaborativeModel
	elif name == 'ImageModel':
		from .image_model import ImageModel
		return ImageModel
	elif name == 'VoiceModel':
		from .voice_model import VoiceModel
		return VoiceModel
	raise AttributeError(f"module 'monoai.models' has no attribute {name!r}")
"""Host-side Gemini router: the same file the official AgentBundle adapter serves from.

Loaded by path so harness and official evaluation share one implementation.
"""
import importlib.util
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / 'official/xpolicylab/AgentBundle/gemini_router.py'
_spec = importlib.util.spec_from_file_location('robodojo_gemini_router', SOURCE)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Every public name of the shared router (constants, validators, GeminiRouter, load_key).
globals().update({k: v for k, v in vars(_module).items() if not k.startswith('_')})
GeminiRouter = _module.GeminiRouter
MODEL = _module.MODEL
MAX_PRICE = _module.MAX_PRICE
MAX_REQUEST_BYTES = _module.MAX_REQUEST_BYTES
load_key = _module.load_key
validate_image = _module.validate_image
request_options = _module.request_options

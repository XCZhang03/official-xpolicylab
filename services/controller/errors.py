"""Actionable public errors without host paths, credentials or provider bodies."""
import re

from services.mcp_contract import Rejected
from services.robodojo.action_validation import ActionValidationError


def public_failure(exc, *, operation=None, stage='request', uncertain=False):
    if stage == 'provider_request':
        # Provider exceptions can embed request headers and arbitrary response bodies.
        status = getattr(exc, 'code', None)
        reason = ('Provider request timed out; its billing reservation is retained.'
                  if isinstance(exc, TimeoutError) else
                  f'Provider request failed ({type(exc).__name__}' +
                  (f', HTTP {status}' if isinstance(status, int) else '') +
                  '); its billing reservation is retained. Private service logs contain diagnostics.')
    else:
        reason = str(exc) or type(exc).__name__
        reason = re.sub(r'https?://\S+', '[redacted URL]', reason)
        reason = re.sub(r'(?<!\w)/(?:[^\s,;\]\)\"\']+)', '[host path]', reason)
        reason = re.sub(r'(?i)(bearer\s+|(?:api[_-]?key|authorization|token)\s*[=:]\s*)[^\s,;]+', r'\1[redacted]', reason)
        reason = re.sub(r'\bsk-[A-Za-z0-9_-]+', '[redacted key]', reason)
        reason = reason[:2000]
    # Validation and contract rejections happen before any robot dispatch.
    validation = isinstance(exc, (ActionValidationError, Rejected))
    return {'type': type(exc).__name__, 'operation': operation, 'stage': stage,
            'reason': reason, 'control_uncertain': bool(uncertain),
            'no_action_executed': not uncertain and (validation or stage in {'request', 'validation', 'api_validation', 'api_budget',
                                                  'provider_request', 'api_accounting'}),
            'recovery': ('Inspect status and observations; do not retry motion in this uncertain episode.'
                         if uncertain else 'Correct the reported problem before retrying; check remaining budgets.')}

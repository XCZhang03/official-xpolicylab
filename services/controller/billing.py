"""Host-only Gemini dollar accounting. Integer nanodollars avoid float drift."""
from decimal import Decimal, InvalidOperation, ROUND_CEILING

NANODOLLARS = 1_000_000_000
SESSION_LIMIT = 10 * NANODOLLARS
from .gemini import MAX_PRICE  # One price cap for reservations and provider requests.


def nanodollars(cost):
    if type(cost) not in (int, float):
        raise ValueError('Missing or invalid provider cost')
    try:
        value = Decimal(str(cost))
        if not value.is_finite() or value < 0:
            raise ValueError('Invalid provider cost')
        return int((value * NANODOLLARS).to_integral_value(rounding=ROUND_CEILING))
    except (InvalidOperation, OverflowError) as error:
        raise ValueError('Invalid provider cost') from error


def reserve_cost(token_reservation, max_output_tokens):
    # The existing conservative input bound includes text, image pixels and
    # protocol/schema overhead. Reasoning is covered by the output token limit.
    prompt = token_reservation - max_output_tokens
    if type(prompt) is not int or prompt < 0 or type(max_output_tokens) is not int or max_output_tokens < 1:
        raise ValueError('Invalid cost reservation')
    return (prompt * nanodollars(MAX_PRICE['prompt']) // 1_000_000
            + max_output_tokens * nanodollars(MAX_PRICE['completion']) // 1_000_000
            + 8 * nanodollars(MAX_PRICE['image']))

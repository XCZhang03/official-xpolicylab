"""Local/raw images cross MCP as bounded bytes, never host paths."""
import base64
import io

import pytest
from PIL import Image

from services.controller.gemini import GeminiRouter, MAX_IMAGE_BYTES


def picture(fmt='PNG', size=(12, 8)):
    out = io.BytesIO()
    Image.new('RGB', size, (30, 60, 90)).save(out, format=fmt)
    return out.getvalue()


def prepare(parts):
    return GeminiRouter('unused').prepare(
        {'messages': [{'role': 'user', 'content': parts}]})


def image_block(raw, mime='image/png'):
    return {'type': 'image', 'mimeType': mime, 'data': base64.b64encode(raw).decode()}


@pytest.mark.parametrize('fmt,mime', [('PNG', 'image/png'), ('JPEG', 'image/jpeg')])
def test_raw_and_local_image_roundtrip(tmp_path, fmt, mime):
    raw = picture(fmt)
    path = tmp_path/'image-without-extension'
    path.write_bytes(raw)
    block = image_block(path.read_bytes(), mime)
    assert block['mimeType'] == mime
    payload, reservation = prepare([block])
    assert payload['messages'][0]['content'][0]['image_url']['url'] == (
        f'data:{mime};base64,' + base64.b64encode(raw).decode())
    assert str(path) not in str(payload)
    assert reservation >= 65536


@pytest.mark.parametrize('part', [
    {'type': 'image', 'observation_id': 'old-reference', 'camera': 'cam_high'},
    {'type': 'image', 'path': '/etc/passwd'},
    {'type': 'image', 'url': 'http://127.0.0.1/private'},
    {'type': 'image', 'mimeType': 'image/png', 'data': 'not-base64!'},
    {'type': 'image', 'mimeType': 'image/png', 'data': ''},
    {'type': 'image', 'mimeType': 'text/plain', 'data': 'aGVsbG8='},
])
def test_reject_invalid_images_without_resolution(part):
    with pytest.raises(ValueError):
        prepare([part])


def test_reject_mismatch_truncation_animation_and_dimensions():
    block = image_block(picture())
    with pytest.raises(ValueError):
        prepare([{**block, 'mimeType': 'image/jpeg'}])
    with pytest.raises((ValueError, OSError, SyntaxError)):
        prepare([{**block, 'data': base64.b64encode(picture()[:-20]).decode()}])
    with pytest.raises(ValueError):
        prepare([image_block(picture(size=(4097, 4096)))])
    out = io.BytesIO()
    Image.new('RGB', (2, 2)).save(out, format='PNG', save_all=True,
                                append_images=[Image.new('RGB', (2, 2), 'red')])
    with pytest.raises(ValueError):
        prepare([image_block(out.getvalue())])


def test_image_count_and_byte_limits(monkeypatch):
    raw = picture()
    with pytest.raises(ValueError):
        prepare([image_block(raw)] * 9)
    with pytest.raises(ValueError):
        prepare([image_block(b'\x89PNG\r\n\x1a\n' + b'x' * MAX_IMAGE_BYTES)])
    with pytest.raises(ValueError):
        prepare([{'type': 'image', 'mimeType': 'image/png',
                  'data': 'A' * (4 * ((MAX_IMAGE_BYTES + 2) // 3) + 1)}])
    monkeypatch.setattr('services.controller.gemini._module.MAX_IMAGE_TOTAL_BYTES', len(raw))
    with pytest.raises(ValueError):
        prepare([image_block(raw)] * 2)

"""Probe bundle: exercises the official-mode contract the way real bundles parse replies."""
import base64
import io
import json

from PIL import Image


def meta_and_images(reply):
    meta = json.loads(next(b['text'] for b in reply['content'] if b['type'] == 'text'))
    images = {a['camera']: Image.open(io.BytesIO(base64.b64decode(reply['content'][a['content_index']]['data'])))
              for a in meta['attachments']}
    return meta, images


def main(ctx):
    meta, images = meta_and_images(ctx.call('robodojo_observe'))
    print('PROBE observe', meta['step_id'], sorted(images), [im.size for im in images.values()], len(meta['states']), flush=True)
    try:
        ctx.call('robodojo_free_space_move', arm='left', target={})
    except PermissionError as exc:
        print('PROBE refused:', str(exc)[:90], flush=True)
    hold = meta['states']
    for chunk in range(3):
        meta, _ = meta_and_images(ctx.call('robodojo_step', actions=[hold, hold]))
        print('PROBE stepped', chunk, 'step_id', meta['step_id'], flush=True)
    (ctx.output_dir / 'probe.json').write_text(json.dumps({'final_step': meta['step_id']}))
    print('PROBE done', flush=True)

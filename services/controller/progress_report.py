"""Readable, offline progress views; JSON remains the evaluation source of truth."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import time
import uuid


def timestamp(value):
    return datetime.fromtimestamp(value, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def detail(row, root):
    message = row.get('error') or row.get('error_type') or row.get('reason') or '—'
    if row['formal_outcome'] == 'error':
        try:
            relative = Path(row['artifacts']['stderr']).relative_to('runtime/autonomous_controller')
            path = (Path(row['source_session']) / 'published' / relative).resolve()
            if path.is_relative_to(root.resolve()):
                with path.open('rb') as stream:
                    stream.seek(max(0, path.stat().st_size - 4096))
                    lines = stream.read(4096).decode('utf-8', errors='replace').splitlines()
                message = next((line for line in reversed(lines) if line.strip()), message)
        except (OSError, KeyError, ValueError):
            pass
    return str(message)[:800]


def render(data, root, *, owner_alive=True):
    status = data['status']
    if status == 'running' and not owner_alive:
        status = 'Coordinator stopped — last saved snapshot'
    completed, total = data['completed_episodes'], data['episode_count']
    successes = data['success_count']
    interim = f'{successes / completed:.1%} ({successes}/{completed})' if completed else 'Not available yet'
    final = f"{data['success_rate']:.1%}" if data['success_rate'] is not None else 'Pending — not a final result'
    summary = [
        ('Status', status), ('Completed', f'{completed} / {total}'),
        ('Successes', str(successes)), ('Unsuccessful', str(data['unsuccessful_count'])),
        ('Errors', str(data['error_count'])), ('Interrupted', str(data['interrupted_count'])),
        ('Running / reserved', str(len(data['active']))), ('Queued', str(len(data['pending_layouts']))),
        ('Success among completed', interim), ('Final success rate', final),
    ]
    score = data.get('score_summary') or {}
    if score.get('average_score_percent') is not None:
        summary.append(('Average native score', f"{score['average_score_percent']:.1f}/100"))
        summary.append(('Score coverage', f"{score['recorded_score_count']} / {score['episode_count']} (missing/invalid = 0)"))
    rows = {r['layout_id']: (r['formal_outcome'], r.get('end_step_id', '—'), detail(r, root))
            for r in data['episodes']}
    for r in data['active'].values():
        rows[r['layout_id']] = (r['status'], '—', f"Started {timestamp(r['started'])}")
    for layout in data['pending_layouts']:
        rows[layout] = ('queued', '—', 'Waiting for a worker')
    subtitle = f"Evaluation collection {data['eval_seed']} · Updated {timestamp(data['updated'])}"
    note = ('Errors include controller exceptions; they do not necessarily indicate infrastructure failures. '
            'The final rate uses all scheduled episodes. No started layout is retried.')
    def md(value):
        return escape(str(value)).replace('|', '&#124;').replace('\n', ' ').replace('\r', ' ')
    markdown = ['# Formal evaluation progress', '', subtitle, '', '| Metric | Value |', '| --- | --- |']
    markdown += [f'| {k} | {md(v)} |' for k, v in summary]
    markdown += ['', note, '', '## Episodes', '', '| Layout | Outcome | Final step | Detail |',
                 '| --- | --- | --- | --- |']
    table = []
    colors = {'success': 'success', 'error': 'error', 'unsuccessful': 'warning',
              'interrupted': 'warning', 'running': 'running', 'reserved': 'running'}
    for layout, (label, step, message) in sorted(rows.items()):
        markdown.append(f'| {layout} | {md(label)} | {md(step)} | {md(message)} |')
        table.append(f'<tr><td>{layout}</td><td><span class="badge {colors.get(label, "")}">'
                     f'{escape(label)}</span></td><td>{escape(str(step))}</td><td>{escape(message)}</td></tr>')
    markdown += ['', '[Raw JSON](formal-report.json) · [Browser view](progress.html)', '']
    cards = ''.join(f'<section><small>{escape(k)}</small><strong>{escape(v)}</strong></section>'
                    for k, v in summary)
    refresh = '<meta http-equiv="refresh" content="10">' if status == 'running' else ''
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
{refresh}<title>Formal evaluation progress</title><style>
body{{margin:0;background:#f3f6fa;color:#17283b;font:15px/1.6 system-ui,sans-serif}}
main{{max-width:1200px;margin:40px auto;padding:0 24px}}h1{{margin-bottom:0;font-size:30px}}
.muted,small{{color:#56677a}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(205px,1fr));gap:12px;margin:24px 0}}
section{{background:white;border:1px solid #dce4ed;border-radius:12px;padding:16px}}
strong,small{{display:block}}strong{{font-size:21px;margin-top:8px}}progress{{width:100%;height:16px;accent-color:#2876d3}}
.table{{overflow:auto;background:white;border:1px solid #dce4ed;border-radius:12px}}
table{{width:100%;border-collapse:collapse;text-align:left}}th,td{{padding:12px 16px;border-bottom:1px solid #e5ebf2}}
th{{background:#eaf0f7}}td:last-child{{min-width:280px;overflow-wrap:anywhere}}td{{vertical-align:top}}
.badge{{padding:3px 9px;border-radius:6px;background:#edf0f4;white-space:nowrap}}
.success{{background:#d8f3e5;color:#155c37}}.error{{background:#ffe0e0;color:#8e2020}}
.warning{{background:#fff0cc;color:#705009}}.running{{background:#deedff;color:#175594}}
a{{color:#185eab}}footer{{margin:24px 0}}p{{overflow-wrap:anywhere}}
</style></head><body><main><h1>Formal evaluation progress</h1>
<p class="muted">{escape(subtitle)}</p><progress aria-label="Completed episodes" value="{completed}" max="{total}"></progress>
<div class="cards">{cards}</div><p class="muted">{escape(note)}</p>
<h2>Episodes</h2><div class="table"><table><thead><tr><th>Layout</th><th>Outcome</th><th>Final step</th><th>Detail</th></tr></thead>
<tbody>{''.join(table)}</tbody></table></div>
<footer>Refreshes every 10 seconds while running. Last saved update is shown above.
<a href="formal-report.json">Raw JSON</a> · <a href="progress.md">Markdown</a></footer>
</main></body></html>'''
    return '\n'.join(markdown), html


def write_report(root, data, *, owner_alive=True):
    markdown, html = render(data, root, owner_alive=owner_alive)
    for name, content in [('progress.md', markdown), ('progress.html', html)]:
        path = root / name
        temporary = path.with_name(f'.{name}-{uuid.uuid4().hex}')
        try:
            with temporary.open('x', encoding='utf-8') as stream:
                os.chmod(temporary, 0o600)
                stream.write(content)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--watch', action='store_true', help='Update an already-running batch without restarting it')
    args = parser.parse_args()
    while True:
        data = json.loads((args.directory / 'formal-report.json').read_text())
        lifecycle = json.loads((args.directory / 'agent-lifecycle.json').read_text())
        alive = Path(f"/proc/{int(lifecycle['pid'])}").exists()
        write_report(args.directory, data, owner_alive=alive)
        if not args.watch or data['status'] != 'running' or not alive:
            break
        time.sleep(10)


if __name__ == '__main__':
    main()

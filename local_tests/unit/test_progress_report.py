from services.controller.progress_report import render, write_report


def data():
    return dict(status='running', completed_episodes=2, episode_count=50,
                success_count=1, unsuccessful_count=0, error_count=1, interrupted_count=0,
                active={'2': dict(layout_id=2, status='running', started=1)}, pending_layouts=[3],
                success_rate=None, eval_seed=1, updated=2,
                episodes=[dict(layout_id=0, formal_outcome='success', end_step_id=280),
                          dict(layout_id=1, formal_outcome='error', error='<script>x</script>|oops\nnext')])


def test_readable_counts_escaping_and_refresh(tmp_path):
    md, html = render(data(), tmp_path)
    assert '50.0% (1/2)' in md and 'Pending — not a final result' in md
    assert '| 0 | success | 280 |' in md
    assert '<script>' not in html and '<script>' not in md
    assert '&#124;oops next' in md
    assert 'http-equiv="refresh"' in html
    assert 'queued' in html and 'running' in html


def test_terminal_and_dead_coordinator_do_not_refresh(tmp_path):
    d = data()
    _, html = render(d, tmp_path, owner_alive=False)
    assert 'Coordinator stopped' in html and 'http-equiv="refresh"' not in html
    d.update(status='completed', success_rate=.02)
    md, html = render(d, tmp_path)
    assert '2.0%' in md and 'http-equiv="refresh"' not in html


def test_views_are_atomic_and_preserve_source(tmp_path):
    source = tmp_path / 'formal-report.json'
    source.write_text('untouched')
    write_report(tmp_path, data())
    assert (tmp_path / 'progress.md').read_text().startswith('# Formal')
    assert (tmp_path / 'progress.html').read_text().startswith('<!doctype html>')
    assert source.read_text() == 'untouched'
    assert not list(tmp_path.glob('.progress*'))

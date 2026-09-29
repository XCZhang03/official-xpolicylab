// Operator dashboard for official-interface auto-research sessions.
const token = new URLSearchParams(location.hash.slice(1)).get('token') || '';
const headers = {'X-Operator-Token': token};
const $ = id => document.getElementById(id);
const CAMERAS = ['cam_high', 'cam_left_wrist', 'cam_right_wrist'];
const TEXT_FIELDS = new Set(['task', 'sim_gpu', 'research_gpu', 'image', 'demonstration_context', 'exploration_collections', 'agent_cli', 'agent_provider']);

let selected = '', current = null, loaded = false, busy = false, timeouts = {};
let resultsSignature = '', liveEpisode = '', liveKey = '', liveLoaded = '', artifactUrl = null;
let episodesSignature = '', episodesAt = 0, episodesRun = '';
const liveUrls = {};

function text(id, value, className) {
  const node = $(id);
  node.textContent = value ?? '';
  if (className !== undefined) node.className = className;
}

async function api(path, options = {}) {
  const response = await fetch(path, {...options, headers: {...headers, ...(options.headers || {})}, cache: 'no-store'});
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `${response.status} ${response.statusText}`);
  return body;
}

function post(path, value) {
  return api(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(value)});
}

function syncSelect(select, rows, value) {
  const signature = JSON.stringify(rows);
  if (select.dataset.signature !== signature) {
    select.replaceChildren(...rows.map(([id, label]) => new Option(label, id)));
    select.dataset.signature = signature;
  }
  select.value = value;
}

function percent(value) { return value == null ? '—' : (100*value).toFixed(1)+'%'; }

function formPayload() {
  const payload = {};
  for (const element of $('form').elements) {
    if (!element.name) continue;
    payload[element.name] = TEXT_FIELDS.has(element.name) ? element.value.trim() : Number(element.value);
  }
  return payload;
}

function updateTimeouts() {
  const budget = timeouts[$('task').value];
  text('timeouts', budget ? `${budget.native_step_limit} native steps · ${budget.episode_seconds/60} min per rehearsal/formal episode · ${budget.session_seconds/3600} h per session including formal.` : '');
}

function clearLive() {
  for (const camera of CAMERAS) {
    const img = $(camera);
    img.hidden = true; img.removeAttribute('src');
    if (liveUrls[camera]) URL.revokeObjectURL(liveUrls[camera]);
    delete liveUrls[camera];
  }
  liveKey = liveLoaded = '';
}

async function renderLive(session) {
  const live = session?.live_observation;
  const episode = JSON.stringify([session?.run_id, live?.native_run]);
  if (episode !== liveEpisode) { clearLive(); liveEpisode = episode; }
  if (!live?.available) {
    clearLive();
    text('liveStatus', 'No native observation yet.', 'muted');
    text('liveMetrics', '');
    return;
  }
  const age = live.updated_at_unix_s ? Math.max(0, Math.floor(Date.now()/1000 - live.updated_at_unix_s)) : null;
  text('liveStatus', `${live.active ? 'Simulator live' : 'Simulator stopped · last saved observation'} · ${live.episode_mode || 'unknown'} · step ${live.step_id} / ${live.episode_step_limit ?? '—'}${age === null ? '' : ` · saved ${age}s ago`}`, live.active ? 'good' : 'muted');
  const reward = live.reward || {};
  text('liveMetrics', `Episode ${live.episode_id || '—'} · native success ${reward.native_success ?? '—'} · score ${reward.native_score_percent ?? '—'} · terminal ${reward.native_end_flag ?? '—'}`, 'mono muted');
  const cameras = CAMERAS.filter(camera => (live.cameras || []).includes(camera));
  const key = JSON.stringify([episode, live.step_id, cameras]);
  if (key === liveKey || key === liveLoaded) return;
  liveKey = key;
  try {
    const frames = await Promise.all(cameras.map(async camera => {
      const query = new URLSearchParams({run_id: session.run_id, native_run: live.native_run, step_id: live.step_id, camera});
      const response = await fetch('/api/frame?'+query, {headers, cache: 'no-store'});
      if (!response.ok) throw new Error('Live frame unavailable; retrying on refresh.');
      return [camera, await response.blob()];
    }));
    if (liveKey !== key || selected !== session.run_id) return;
    // Swap all cameras together so a view never mixes steps or episodes.
    for (const camera of CAMERAS) {
      if (liveUrls[camera]) URL.revokeObjectURL(liveUrls[camera]);
      delete liveUrls[camera];
      const frame = frames.find(([name]) => name === camera), img = $(camera);
      img.hidden = !frame;
      if (frame) { liveUrls[camera] = URL.createObjectURL(frame[1]); img.src = liveUrls[camera]; }
      else img.removeAttribute('src');
    }
    liveLoaded = key;
  } catch (error) {
    if (liveKey === key) text('liveStatus', error.message, 'muted');
  } finally {
    if (liveKey === key) liveKey = '';
  }
}

function renderResults(session) {
  const signature = JSON.stringify([session?.run_id, session?.results]);
  if (signature === resultsSignature) return;
  resultsSignature = signature;
  const rows = [...(session?.results || [])].reverse().map(row => {
    const tr = document.createElement('tr');
    const cells = [
      `${row.mode} · ${row.formal_episode_index ?? row.episode ?? row.episode_id ?? '—'}`,
      (row.bundle || '—').slice(0, 12),
      row.task_complete === true ? 'true' : row.task_complete === false ? 'false' : 'unknown',
      `${row.reason || '—'} / exit ${row.returncode ?? '—'}${row.error_type ? ' / '+row.error_type : ''}`,
    ].map(value => { const td = document.createElement('td'); td.textContent = value; return td; });
    cells[2].className = row.task_complete === true ? 'good' : row.task_complete === false ? 'bad' : 'muted';
    const td = document.createElement('td'), details = document.createElement('details');
    const summary = document.createElement('summary'), pre = document.createElement('pre');
    summary.textContent = 'Result'; pre.textContent = JSON.stringify(row, null, 2);
    details.append(summary, pre);
    const artifacts = [['stdout', row.artifacts?.stdout], ['stderr', row.artifacts?.stderr],
      ...(row.artifacts?.final_images || []).map((frame, index) => ['final image '+(index+1), frame.path])];
    for (const [label, path] of artifacts) {
      if (!path) continue;
      const button = document.createElement('button');
      button.type = 'button'; button.textContent = label;
      button.addEventListener('click', () => viewArtifact(session.run_id, path));
      details.append(button);
    }
    td.append(details);
    tr.append(...cells, td);
    return tr;
  });
  $('results').replaceChildren(...rows);
}

function renderRelease(release) {
  $('releaseCommands').hidden = !release;
  if (!release) { text('release', 'No rehearsed or submitted bundle yet.', 'muted'); return; }
  text('release', `${release.submitted ? 'Submitted' : 'Newest rehearsed'} bundle ${release.bundle} · sha256 ${release.sha256.slice(0, 16)}… · ${release.path}`, 'mono');
  text('officialCheck', release.official_check);
  text('buildCheckpoint', release.build_checkpoint);
  const counterpart = release.counterpart;
  $('variant').hidden = !counterpart;
  if (!counterpart) return;
  text('variantTask', counterpart.task);
  const combined = release.build_combined;
  $('buildCombined').hidden = $('copyCombined').hidden = !combined;
  if (combined) {
    text('variantNote', `Combined checkpoint with the ${counterpart.submitted ? 'submitted' : 'newest rehearsed'} ${counterpart.task} bundle ${counterpart.bundle} from session ${counterpart.run_id}:`, 'muted');
    text('buildCombined', combined);
  } else {
    text('variantNote', `No session has a rehearsed ${counterpart.task} bundle yet. Without one, a checkpoint scores zero on ${counterpart.task}; run a ${counterpart.task} session, or list both task names for one bundle (task,${counterpart.task}=<bundle>) if it handles both.`, 'warn');
  }
}

function render(data) {
  timeouts = data.task_timeouts || {};
  if (!loaded) {
    $('task').replaceChildren(...data.tasks.map(task => new Option(task, task)));
    $('form').elements.demonstration_context.replaceChildren(...data.demonstration_contexts.map(kind => new Option(kind, kind)));
    for (const element of $('form').elements) if (element.name && element.name in data.profile) element.value = data.profile[element.name];
    $('prepare').disabled = false;
    loaded = true;
    updateTimeouts();
  }
  const session = data.selected;
  current = session; selected = session?.run_id || '';
  syncSelect($('session'), data.sessions.length ? data.sessions.map(row => [row.run_id, `${row.task} · ${row.run_id} · ${row.status}`]) : [['', 'No sessions']], selected);
  text('sTask', session?.task || '—');
  text('sStatus', session?.status || '—');
  text('sStage', session?.stage || '—');
  text('sManual', session ? String(session.manual_success) : '—');
  text('sEpisodes', session ? `${session.exploration_used} / ${session.exploration_limit}` : '—');
  const batch = session?.formal_batch;
  text('sFormal', batch ? `${batch.success_count} / ${batch.episode_count} (${percent(batch.success_rate)})` : '—');
  const score = batch?.score_summary;
  text('sScore', score ? (score.average_score_percent != null ? score.average_score_percent.toFixed(1)+'%' : `pending · ${score.recorded_score_count} recorded`) : '—');
  text('sTokens', session?.usage?.total_session_usage?.total_tokens?.toLocaleString() ?? '—');
  const gemini = session?.gemini;
  text('sGemini', gemini ? (gemini.blocked ? 'Blocked' : `$${gemini.reported_usd.toFixed(3)} / $${gemini.limit_usd}`) : '—');
  $('sGemini').title = gemini ? `Reserved $${gemini.reserved_usd.toFixed(4)}; remaining $${gemini.remaining_usd.toFixed(3)}. Codex costs are separate.` : '';
  text('sGeminiCalls', gemini ? `${gemini.calls.development} / ${gemini.calls.formal}` : '—');
  const agent = session?.agent;
  text('sAgent', !agent ? '—' : agent.cli !== 'claude' ? agent.cli
    : `claude · ${agent.provider}${agent.provider === 'claude-login' ? ` · ${agent.account || 'no experiment token'}${agent.pending ? ' (at launch)' : ''}` : ''}`);
  text('command', session?.command || 'No session selected.', session ? '' : 'muted');
  $('copy').disabled = !session;
  $('launch').disabled = !session?.launchable;
  $('stop').disabled = !session?.alive;
  text('paths', session ? `Workspace: ${session.workspace}\nArtifacts: ${session.artifacts}` : '');
  text('agentLog', session?.agent_log || 'No captured log. Terminal launches print to their terminal.');
  text('frontendLog', session?.frontend_log || 'No frontend log yet.');
  text('configuration', JSON.stringify({configuration: session?.configuration, bundles: session?.bundles,
    rehearsed_bundles: session?.rehearsed_bundles}, null, 2));
  renderRelease(session?.release);
  refreshEpisodes(session?.run_id !== episodesRun);
  renderResults(session);
  renderLive(session);
}

async function refresh() {
  if (busy) return;
  busy = true;
  const requested = selected;
  try {
    const query = new URLSearchParams(history() ? {history: '1'} : {});
    if (requested) query.set('run_id', requested);
    const data = await api('/api/state?'+query);
    if (requested === selected) render(data);
  } catch (error) {
    text('message', error.message, 'bad');
  } finally {
    busy = false;
  }
}

async function viewArtifact(runId, path) {
  try {
    const response = await fetch('/api/artifact?'+new URLSearchParams({run_id: runId, path}), {headers, cache: 'no-store'});
    if (!response.ok) throw new Error((await response.json()).error);
    if (artifactUrl) { URL.revokeObjectURL(artifactUrl); artifactUrl = null; }
    const image = response.headers.get('Content-Type').startsWith('image/');
    $('artifactImage').hidden = !image; $('artifactText').hidden = image;
    if (image) { artifactUrl = URL.createObjectURL(await response.blob()); $('artifactImage').src = artifactUrl; }
    else text('artifactText', await response.text());
    text('artifactPath', `${runId} / ${path}`, 'mono');
    $('artifactViewer').open = true;
  } catch (error) {
    text('message', error.message, 'bad');
  }
}

$('task').addEventListener('change', updateTimeouts);
$('form').addEventListener('input', () => text('message', 'Unsaved setup edits; Prepare creates a new session and leaves the selected one unchanged.', 'muted'));
$('form').addEventListener('submit', async event => {
  event.preventDefault();
  $('prepare').disabled = true;
  try {
    const prepared = await post('/api/prepare', formPayload());
    selected = prepared.run_id;
    text('message', 'Configuration prepared; nothing is running yet. Launch it below or copy the terminal command.', 'good');
    await refresh();
  } catch (error) {
    text('message', error.message, 'bad');
  } finally {
    $('prepare').disabled = false;
  }
});
$('session').addEventListener('change', () => { selected = $('session').value; resultsSignature = ''; refresh(); });
$('refresh').addEventListener('click', refresh);
document.addEventListener('click', async event => {
  const source = event.target.closest('[data-copy]');
  if (!source) return;
  try { await navigator.clipboard.writeText($(source.dataset.copy).textContent); text('message', 'Copied.', 'good'); }
  catch (_) { text('message', 'Clipboard unavailable; select the text and copy it.', 'muted'); }
});
for (const [id, action] of [['launch', 'launch'], ['stop', 'stop']]) {
  $(id).addEventListener('click', async () => {
    if (!current) return;
    const runId = current.run_id;
    const question = action === 'launch'
      ? `Launch the paid Codex session for ${current.task}? It uses the prepared configuration, not unsaved form edits.`
      : 'Stop this session and close its environment? Budgets are not restored and no formal retry is permitted.';
    if (!confirm(question)) return;
    $(id).disabled = true;
    try { text('message', (await post('/api/'+action, {run_id: runId})).status, 'good'); }
    catch (error) { text('message', error.message, 'bad'); }
    await refresh();
  });
}
function history() { return $('history').checked; }

function percentScore(value) { return value == null ? '—' : (100*value).toFixed(1); }

function playEpisode(session, row, index) {
  const url = '/api/video?'+new URLSearchParams({run_id: session, episode: row.episode, ...(history() ? {history: '1'} : {})});
  const video = $('video');
  video.hidden = false;
  video.src = url;
  video.playbackRate = Number($('speed').value);
  video.play().catch(() => {});
  $('download').href = url;
  $('download').download = `${session}-${index}-${row.mode}.mp4`;
  $('download').hidden = false;
  text('playing', `Episode ${index} · ${row.mode}${row.formal_episode_index ? ' '+row.formal_episode_index : ''} · ${row.episode_id || ''}`, 'muted');
}

async function refreshEpisodes(force = false) {
  // Episode lists change slowly; poll them every 10 s, or at once on a new selection.
  if (!selected || (!force && selected === episodesRun && Date.now() - episodesAt < 10000)) return;
  const session = selected;
  let rows;
  try {
    rows = await api('/api/episodes?'+new URLSearchParams({run_id: session, ...(history() ? {history: '1'} : {})}));
  } catch (error) {
    text('message', error.message, 'bad');
    return;
  }
  if (session !== selected) return;
  if (session !== episodesRun) {
    $('video').pause(); $('video').removeAttribute('src'); $('video').hidden = true;
    $('download').hidden = true; text('playing', '');
  }
  episodesRun = session; episodesAt = Date.now();
  const signature = JSON.stringify(rows);
  if (signature === episodesSignature) return;
  episodesSignature = signature;
  $('episodes').replaceChildren(...rows.map((row, i) => {
    const tr = document.createElement('tr');
    const result = row.live ? 'running' : row.native_success === true ? 'success' : row.native_success === false ? 'not successful' : 'unknown';
    const cells = [i + 1, row.mode + (row.formal_episode_index ? ' '+row.formal_episode_index : ''),
      (row.episode_id || '—').slice(0, 12), `${row.steps ?? '—'}${row.step_limit ? ' / '+row.step_limit : ''}`,
      result, percentScore(row.native_score)].map(value => {
        const td = document.createElement('td'); td.textContent = value; return td; });
    cells[4].className = result === 'success' ? 'good' : result === 'not successful' ? 'bad' : 'muted';
    const td = document.createElement('td'), button = document.createElement('button');
    button.type = 'button';
    // The simulator finalizes an episode's MP4 when the episode closes.
    button.textContent = row.live ? 'Recording…' : row.video ? `Play (${(row.video_bytes / 1048576).toFixed(1)} MB)` : 'No video';
    button.disabled = row.live || !row.video;
    button.addEventListener('click', () => playEpisode(session, row, i + 1));
    td.append(button);
    tr.append(...cells, td);
    return tr;
  }));
  if (!rows.length) {
    const tr = document.createElement('tr'), td = document.createElement('td');
    td.colSpan = 7; td.className = 'muted'; td.textContent = 'No simulator episodes recorded yet.';
    tr.append(td); $('episodes').append(tr);
  }
}

$('speed').addEventListener('change', () => { $('video').playbackRate = Number($('speed').value); });
$('history').addEventListener('change', () => { selected = ''; resultsSignature = ''; episodesSignature = ''; episodesRun = ''; refresh(); });
// #history=1 opens the dashboard with earlier sessions listed.
$('history').checked = new URLSearchParams(location.hash.slice(1)).get('history') === '1';
if (!token) text('message', 'Missing operator token: open the URL printed by the dashboard server.', 'bad');
fetch('/api/session', {method: 'POST', headers}).catch(() => {});  // Cookie for <video> requests.
refresh();
setInterval(refresh, 1000);

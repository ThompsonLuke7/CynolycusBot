(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const meta = {
    spy: ['SP', 'Intraday · execution', '#8fb6ee', 'trading'],
    swing: ['SW', '30 minute · execution', '#a8b9ec', 'trading'],
    htf: ['HT', 'Higher timeframe · signals', '#bea4e3', 'trading'],
    momentum: ['MO', 'Momentum · execution', '#91ddc5', 'trading'],
    amethyst: ['AM', 'Dealer intelligence', '#bba2e4', 'research'],
    dealer: ['DP', 'Positioning · analytics', '#e2bc89', 'research'],
    dealer_ranker: ['DR', 'Dealer · ranking', '#d8b88f', 'trading'],
    meta: ['MR', 'Governed · ranking', '#94c8d2', 'trading'],
    intraday_structure: ['IS', 'Intraday · structure', '#b6cda2', 'trading'],
    library: ['LI', 'Research · library', '#a4bbc8', 'research'],
    trades: ['TR', 'Execution · journal', '#a4bbc8', 'research']
  };
  const read = (key, fallback) => { try { return JSON.parse(localStorage.getItem(key)) ?? fallback; } catch (_) { return fallback; } };
  const save = (key, value) => { try { localStorage.setItem(key, JSON.stringify(value)); } catch (_) { /* Preferences are optional. */ } };
  const storedIntent = read('cyno-hub-live', {});
  const liveState = storedIntent && typeof storedIntent === 'object' ? storedIntent : {};
  const initial = JSON.parse($('dashboard-config').textContent);
  let dashboards = initial, snapshot = null, filter = 'all', stale = true, polling = false;
  let lastUpdated = null, allPending = false;
  const pending = new Set(), cards = new Map(), navLinks = new Map();
  let showPreviews = read('cyno-hub-previews-v2', false) === true;
  const format = (n, digits = 0) => n == null || !Number.isFinite(Number(n)) ? '—' : Number(n).toLocaleString('en-US', {maximumFractionDigits: digits});
  const money = (n, digits = 2) => n == null || !Number.isFinite(Number(n)) ? '—' : (n < 0 ? '−$' : '$') + Math.abs(n).toLocaleString('en-US', {minimumFractionDigits: digits, maximumFractionDigits: digits});
  const tone = n => n == null ? '' : n < 0 ? 'negative' : n > 0 ? 'positive' : '';
  const attention = d => !d.up || /error|fail|warming|stopping/i.test(d.state || '');
  const description = d => meta[d.key] || ['•', 'System module', '#94a4a9', 'research'];
  const moduleURL = d => {
    // Use the browser-visible host, including IPv6, with the configured module port.
    const url = new URL(location.href);
    url.protocol = 'http:'; url.port = String(d.port); url.pathname = '/'; url.search = ''; url.hash = '';
    return url.href;
  };
  const setMessage = text => { $('msg').textContent = text; $('msg').hidden = !text; };
  function createCard(d) {
    const card = document.createElement('article');
    card.className = 'module-card'; card.dataset.key = d.key;
    card.innerHTML = `<header><span class="module-icon" aria-hidden="true"></span><div class="module-title"><h3></h3><small></small></div><a class="open-link" target="_blank" rel="noopener">↗</a></header>
      <div class="module-content"><div class="status-row"><span class="pill status-pill"></span><span class="pill account-pill"></span></div><p class="module-detail"></p>
      <div class="performance"><span><strong class="tracked">—</strong>Tracked P/L</span><span><strong class="win-rate">—</strong>Win rate</span><span><strong class="closed">—</strong>Closed</span></div></div>
      <div class="module-footer"><label class="intent"><input type="checkbox"> Real money</label><span class="managed"></span><div class="card-controls"><button class="button secondary start"></button><button class="button stop">Stop</button></div></div><div class="thumb" hidden></div>`;
    card.querySelector('.intent input').addEventListener('change', event => {
      liveState[d.key] = event.target.checked; save('cyno-hub-live', liveState); updateCard(dashboards.find(item => item.key === d.key));
    });
    card.querySelector('.start').addEventListener('click', () => control(d.key, 'start'));
    card.querySelector('.stop').addEventListener('click', () => control(d.key, 'stop'));
    cards.set(d.key, card); $('grid').append(card);
    return card;
  }
  function updateCard(d) {
    const card = cards.get(d.key) || createCard(d), info = description(d), q = selector => card.querySelector(selector);
    card.style.setProperty('--module-color', info[2]);
    q('.module-icon').textContent = info[0]; q('h3').textContent = d.name; q('.module-title small').textContent = info[1];
    q('.open-link').href = moduleURL(d); q('.open-link').setAttribute('aria-label', 'Open ' + d.name + ' dashboard');
    const status = d.up ? (d.state || 'idle') : snapshot ? 'Unavailable' : 'Waiting';
    q('.status-pill').textContent = status;
    q('.status-pill').className = 'pill status-pill ' + (!snapshot ? '' : !d.up || /error|fail/i.test(status) ? 'bad' : /running|ready/i.test(status) ? 'good' : 'warning');
    const account = d.account_type || (d.tradeable ? 'Account unknown' : info[3] === 'research' ? 'Analytics' : 'Managed');
    q('.account-pill').textContent = account;
    q('.account-pill').classList.toggle('real', account === 'real money');
    q('.module-detail').textContent = d.up ? d.detail || 'No additional status reported.' : d.error || 'Waiting for module status.';
    q('.module-detail').title = q('.module-detail').textContent;
    const p = d.up ? d.performance || {} : {};
    q('.tracked').textContent = p.closed_trades > 0 ? money(p.tracked_pnl) : '—';
    q('.tracked').className = 'tracked ' + (p.closed_trades > 0 ? tone(p.tracked_pnl) : '');
    q('.win-rate').textContent = p.win_rate == null ? '—' : format(p.win_rate, 1) + '%';
    q('.closed').textContent = format(p.closed_trades);
    q('.intent').hidden = !d.tradeable;
    q('.intent input').checked = liveState[d.key] === true;
    q('.intent input').disabled = !d.live_available || stale || pending.has(d.key) || allPending;
    q('.intent').title = d.live_available ? 'Account intent for the next start. Starting with real money requires confirmation.' : 'Real-money execution is unavailable for this module.';
    q('.managed').hidden = d.tradeable;
    q('.managed').textContent = d.policy_mode ? 'Policy · ' + d.policy_mode : d.startable ? 'Analytics session' : 'Managed externally';
    q('.start').hidden = !d.startable; q('.stop').hidden = !d.stoppable;
    q('.start').textContent = ['momentum', 'dealer_ranker'].includes(d.key) ? 'Run pass' : 'Start';
    q('.start').disabled = q('.stop').disabled = !d.up || stale || pending.has(d.key) || allPending;
    q('.thumb').hidden = !showPreviews;
    if (showPreviews && !q('iframe')) {
      const frame = document.createElement('iframe'); frame.src = moduleURL(d); frame.loading = 'lazy'; frame.title = d.name + ' live preview'; frame.tabIndex = -1;
      q('.thumb').append(frame);
    } else if (!showPreviews && q('iframe')) q('iframe').remove();
  }
  function applyFilters() {
    const term = $('module-search').value.trim().toLowerCase(); let shown = 0;
    dashboards.forEach(d => {
      const info = description(d);
      const match = (filter === 'all' || (filter === 'attention' ? snapshot && attention(d) : info[3] === filter)) && (d.name + ' ' + info[1]).toLowerCase().includes(term);
      cards.get(d.key).hidden = !match; if (match) shown++;
    });
    $('empty').hidden = shown > 0;
    $('module-count').textContent = format(shown) + ' / ' + dashboards.length;
  }
  function render() {
    dashboards.forEach(updateCard);
    for (const [key, card] of cards) if (!dashboards.some(d => d.key === key)) { card.remove(); cards.delete(key); }
    dashboards.forEach(d => {
      let a = navLinks.get(d.key);
      if (!a) {
        a = document.createElement('a'); a.target = '_blank'; a.rel = 'noopener';
        a.append(document.createElement('span'), document.createElement('span'));
        navLinks.set(d.key, a); $('module-nav').append(a);
      }
      a.href = moduleURL(d);
      a.firstChild.className = 'nav-dot' + (snapshot && !d.up ? ' down' : '');
      a.lastChild.textContent = d.name;
    });
    for (const [key, a] of navLinks) if (!dashboards.some(d => d.key === key)) { a.remove(); navLinks.delete(key); }
    $('directory-count').textContent = dashboards.length;
    const totals = snapshot?.totals || {}, anyReporting = dashboards.some(d => d.up);
    $('equity').textContent = money(totals.equity, 0);
    $('positions').textContent = format(anyReporting ? totals.open_positions : null);
    $('upl').textContent = money(anyReporting ? totals.unrealized_pl : null);
    $('upl').className = tone(anyReporting ? totals.unrealized_pl : null);
    $('account-positions').textContent = totals.account_positions == null ? 'Attributed to reporting modules' : format(totals.account_positions) + ' total in shared account';
    const up = dashboards.filter(d => d.up).length;
    $('availability').textContent = snapshot ? up + ' / ' + dashboards.length : '—';
    $('health-track').replaceChildren(...dashboards.map(d => { const i = document.createElement('i'); if (!d.up) i.className = 'down'; return i; }));
    $('health-note').textContent = !snapshot ? 'Waiting for the first snapshot' : up === dashboards.length ? 'All module endpoints responding' : (dashboards.length - up) + ' unavailable · totals may be incomplete';
    $('attention-count').textContent = snapshot ? dashboards.filter(attention).length : '—';
    $('start-all').disabled = stale || allPending || pending.size > 0 || !dashboards.some(d => d.up && d.startable);
    applyFilters();
  }
  async function tick() {
    if (polling) return; polling = true; $('refresh').disabled = true;
    const abort = new AbortController(), timeout = setTimeout(() => abort.abort(), 12000);
    try {
      const response = await fetch('/api/state', {cache: 'no-store', signal: abort.signal});
      if (!response.ok) throw new Error('HTTP ' + response.status);
      const data = await response.json();
      if (!Array.isArray(data.dashboards) || !data.totals) throw new Error('Invalid snapshot');
      snapshot = data; dashboards = data.dashboards; stale = false; lastUpdated = new Date();
      $('connection').textContent = 'Connected · 5s refresh'; $('connection').className = 'connection good';
      $('stale-notice').hidden = true;
      $('updated').textContent = 'Updated ' + lastUpdated.toLocaleTimeString('en-US') + ' · refreshes every 5s';
    } catch (_) {
      stale = true; $('connection').textContent = 'Connection interrupted'; $('connection').className = 'connection bad';
      $('stale-notice').hidden = false;
      $('stale-notice').textContent = snapshot ? 'Snapshot is stale. Showing the last update from ' + lastUpdated.toLocaleTimeString('en-US') + '. Controls are paused; reconnecting automatically.' : 'The hub is unavailable. Waiting for a snapshot; reconnecting automatically.';
    } finally { clearTimeout(timeout); polling = false; $('refresh').disabled = false; render(); }
  }
  async function post(path, body) {
    const response = await fetch('/api/' + path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    return response.json();
  }
  async function control(key, action) {
    const d = dashboards.find(item => item.key === key);
    if (!d || stale || allPending || pending.has(key) || !d.up) return;
    if (action === 'start' ? !d.startable : !d.stoppable) return;
    const live = d.tradeable && d.live_available && liveState[key] === true;
    if (action === 'start' && live && !confirm('Start ' + d.name + ' on the REAL-MONEY account?')) return;
    pending.add(key); render(); setMessage((action === 'start' ? 'Starting ' : 'Stopping ') + d.name + '…');
    try {
      const result = await post(action, action === 'start' ? {key, live} : {key});
      setMessage(result.ok ? d.name + ': request completed.' : d.name + ': ' + (result.error || 'request failed.'));
    } catch (error) { setMessage(d.name + ': response unavailable (' + error.message + '). Check the module before retrying; the request may have reached it.'); }
    finally { pending.delete(key); render(); tick(); }
  }
  $('start-all').addEventListener('click', async () => {
    if (stale || allPending || pending.size) return;
    const liveMap = Object.fromEntries(dashboards.filter(d => d.startable && !['meta', 'momentum', 'dealer_ranker'].includes(d.key)).map(d => [d.key, d.tradeable && d.live_available && liveState[d.key] === true]));
    if (Object.values(liveMap).some(Boolean) && !confirm('Start all sessions — some modules are set to REAL MONEY. Continue?')) return;
    allPending = true; render(); setMessage('Starting continuous sessions… Scheduled passes remain on their schedule.');
    try {
      const result = await post('start-all', {live_map: liveMap});
      if (!Array.isArray(result.results)) throw new Error(result.error || 'Invalid control response');
      const failed = result.results.filter(r => !r.ok), skipped = result.results.filter(r => r.skipped);
      setMessage(failed.length ? 'Some sessions could not start: ' + failed.map(r => r.key + ' (' + r.error + ')').join(', ') : 'Session requests completed. ' + skipped.length + ' scheduled passes skipped.');
    } catch (error) { setMessage('Response unavailable (' + error.message + '). Check module status before retrying; requests may have reached the server.'); }
    finally { allPending = false; render(); tick(); }
  });
  $('previews-toggle').checked = showPreviews;
  $('previews-toggle').addEventListener('change', event => { showPreviews = event.target.checked; save('cyno-hub-previews-v2', showPreviews); render(); });
  $('refresh').addEventListener('click', tick);
  $('module-search').addEventListener('input', applyFilters);
  document.querySelectorAll('[data-filter]').forEach(button => button.addEventListener('click', () => {
    filter = button.dataset.filter;
    document.querySelectorAll('[data-filter]').forEach(item => item.setAttribute('aria-pressed', String(item === button))); applyFilters();
  }));
  $('reset-filters').addEventListener('click', () => { $('module-search').value = ''; document.querySelector('[data-filter="all"]').click(); });
  document.addEventListener('keydown', event => { if (event.key === '/' && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) { event.preventDefault(); $('module-search').focus(); } });
  const updateClock = () => { $('clock').textContent = new Date().toLocaleString('en-US', {timeZone: 'America/New_York', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'}) + ' ET'; };
  updateClock(); setInterval(updateClock, 30000); render(); tick(); setInterval(tick, 5000);
})();

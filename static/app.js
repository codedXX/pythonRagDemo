// Question-answering interface.
//
// Talks only to /api/*. No provider endpoint or credential is present here; the
// browser never learns what the server talks to.

const els = {
  messages: document.getElementById('messages'),
  composer: document.getElementById('composer'),
  question: document.getElementById('question'),
  send: document.getElementById('send'),
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('file-input'),
  uploadList: document.getElementById('upload-list'),
  docList: document.getElementById('doc-list'),
  refreshDocs: document.getElementById('refresh-docs'),
  healthPanel: document.getElementById('health-panel'),
  healthToggle: document.getElementById('health-toggle'),
  debugCheck: document.getElementById('debug-check'),
};

// The client owns the transcript so the server can stay stateless.
const history = [];
let busy = false;

// ---------------------------------------------------------------- utilities

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function escapeHtml(text) {
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

/**
 * Minimal Markdown renderer.
 *
 * Everything is escaped before any markup is emitted, because the text comes from
 * uploaded documents and model output and must never be interpreted as HTML.
 */
function renderMarkdown(source) {
  const blocks = [];
  // Fenced code and tables are extracted first so inline processing cannot break
  // their contents.
  let text = source.replace(/```([\w+-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
    blocks.push(
      `<pre><code>${escapeHtml(code.replace(/\n$/, ''))}</code></pre>`
    );
    return `\u0000BLOCK${blocks.length - 1}\u0000`;
  });

  const lines = text.split('\n');
  const out = [];
  let listType = null;
  let table = null;

  const closeList = () => {
    if (listType) {
      out.push(`</${listType}>`);
      listType = null;
    }
  };
  const closeTable = () => {
    if (table) {
      out.push('<table>');
      table.forEach((row, index) => {
        const tag = index === 0 ? 'th' : 'td';
        out.push(`<tr>${row.map((cell) => `<${tag}>${inline(cell)}</${tag}>`).join('')}</tr>`);
      });
      out.push('</table>');
      table = null;
    }
  };

  function inline(value) {
    return escapeHtml(value)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[\s(])\*([^*\n]+)\*/g, '$1<em>$2</em>')
      .replace(/\[(\d+)\]/g, '<span class="citation">[$1]</span>')
      .replace(/\[([^\]]+)\]\((https?:[^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  }

  for (const raw of lines) {
    const line = raw;
    if (/^\u0000BLOCK\d+\u0000$/.test(line.trim())) {
      closeList();
      closeTable();
      out.push(line.trim());
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line)) {
      const cells = line.trim().replace(/^\||\|$/g, '').split('|').map((c) => c.trim());
      // Skip the separator row (---) of a Markdown table.
      if (!cells.every((c) => /^:?-{2,}:?$/.test(c))) {
        closeList();
        table = table || [];
        table.push(cells);
        continue;
      }
      continue;
    }
    closeTable();

    if (!line.trim()) {
      closeList();
      continue;
    }
    const heading = line.match(/^(#{1,6})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = Math.min(3, heading[1].length);
      out.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }
    if (/^\s*>\s?/.test(line)) {
      closeList();
      out.push(`<blockquote>${inline(line.replace(/^\s*>\s?/, ''))}</blockquote>`);
      continue;
    }
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    const ordered = line.match(/^\s*\d+[.)]\s+(.*)$/);
    if (bullet || ordered) {
      const wanted = bullet ? 'ul' : 'ol';
      if (listType !== wanted) {
        closeList();
        out.push(`<${wanted}>`);
        listType = wanted;
      }
      out.push(`<li>${inline((bullet || ordered)[1])}</li>`);
      continue;
    }
    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  closeTable();

  let html = out.join('\n');
  html = html.replace(/\u0000BLOCK(\d+)\u0000/g, (_, index) => blocks[Number(index)]);
  return html;
}

// -------------------------------------------------------------- health panel

function statusClass(status) {
  if (status === 'healthy') return 'health-ok';
  if (status === 'misconfigured') return 'health-warn';
  return 'health-bad';
}

async function loadHealth() {
  try {
    const response = await fetch('/api/health');
    const data = await response.json();
    els.healthPanel.replaceChildren();

    for (const dep of data.dependencies || []) {
      const row = el('div', 'health-row');
      row.append(el('span', 'health-name', dep.name.replace(/_/g, ' ')));
      row.append(el('span', statusClass(dep.status), dep.status));
      els.healthPanel.append(row);
      if (dep.status !== 'healthy' && dep.message) {
        els.healthPanel.append(el('div', 'health-msg', dep.message));
      }
    }
    if (data.dimension) {
      const row = el('div', 'health-row');
      row.append(el('span', 'health-name', 'vector dim'));
      row.append(el('span', 'health-ok', String(data.dimension)));
      els.healthPanel.append(row);
    }
  } catch (error) {
    els.healthPanel.replaceChildren(el('div', 'health-msg', `无法获取状态：${error.message}`));
  }
}

els.healthToggle.addEventListener('click', async () => {
  const hidden = els.healthPanel.classList.toggle('hidden');
  if (!hidden) await loadHealth();
});

/** Surface a dependency problem before the user tries to act. */
async function warnIfUnhealthy() {
  try {
    const data = await (await fetch('/api/health')).json();
    const broken = (data.dependencies || []).filter((d) => d.status === 'misconfigured');
    if (broken.length === 0) return false;
    els.healthPanel.classList.remove('hidden');
    await loadHealth();
    return true;
  } catch {
    return false;
  }
}

// ----------------------------------------------------------------- documents

function describeDoc(doc) {
  if (doc.state === 'failed') return doc.error || '处理失败';
  if (doc.state === 'staging') return '处理中…';
  return `${doc.parent_count} 段 / ${doc.child_count} 块`;
}

async function loadDocuments() {
  try {
    const data = await (await fetch('/api/docs')).json();
    els.docList.replaceChildren();
    if (!data.documents || data.documents.length === 0) {
      els.docList.append(el('li', 'doc-sub', '还没有文档'));
      return;
    }
    for (const doc of data.documents) {
      const item = el('li', `doc-item ${doc.state}`);
      const meta = el('div', 'doc-meta');
      meta.append(el('span', 'doc-name', doc.name));
      meta.append(el('span', 'doc-sub', describeDoc(doc)));
      item.append(meta);

      const remove = el('button', 'doc-remove', '×');
      remove.title = '删除';
      remove.disabled = doc.state === 'staging';
      remove.addEventListener('click', () => removeDocument(doc, remove));
      item.append(remove);
      els.docList.append(item);
    }
  } catch (error) {
    els.docList.replaceChildren(el('li', 'doc-sub', `加载失败：${error.message}`));
  }
}

async function removeDocument(doc, button) {
  button.disabled = true;
  try {
    const response = await fetch(`/api/docs/${encodeURIComponent(doc.doc_id)}`, {
      method: 'DELETE',
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    await loadDocuments();
    await loadHealth();
  } catch (error) {
    button.disabled = false;
    button.title = `删除失败：${error.message}`;
  }
}

els.refreshDocs.addEventListener('click', loadDocuments);

// -------------------------------------------------------------------- upload

function uploadRow(name) {
  const item = el('li', 'upload-item');
  item.dataset.filename = name;

  const head = el('div', 'upload-item-head');
  head.append(el('span', 'upload-name', name));
  head.append(el('span', 'upload-state', '等待中'));
  item.append(head);

  const track = el('div', 'progress-track');
  const bar = el('div', 'progress-bar');
  track.append(bar);
  item.append(track);

  const detail = el('p', 'upload-detail', '');
  item.append(detail);

  els.uploadList.append(item);
  return { item, state: head.lastElementChild, bar, detail };
}

const STAGE_LABEL = {
  parsing: '解析中',
  chunking: '分块中',
  embedding: '向量化中',
  persisting: '入库中',
  done: '完成',
  failed: '失败',
};

async function uploadFiles(files) {
  if (!files || files.length === 0) return;

  const allowed = [];
  for (const file of files) {
    const ext = (file.name.split('.').pop() || '').toLowerCase();
    if (!['pdf', 'md', 'markdown'].includes(ext)) {
      const row = uploadRow(file.name);
      row.item.classList.add('failed');
      row.state.textContent = '不支持';
      row.detail.textContent = `不支持 .${ext}；请上传 PDF 或 Markdown（.md/.markdown）`;
      continue;
    }
    allowed.push(file);
  }
  if (allowed.length === 0) return;

  const form = new FormData();
  for (const file of allowed) form.append('files', file);

  const rows = new Map();
  for (const file of allowed) rows.set(file.name, uploadRow(file.name));

  try {
    const response = await fetch('/api/docs', { method: 'POST', body: form });
    if (!response.ok || !response.body) {
      throw new Error(`HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        handleUploadEvent(frame, rows);
        boundary = buffer.indexOf('\n\n');
      }
    }
  } catch (error) {
    for (const [name, row] of rows) {
      if (!row.item.classList.contains('done')) {
        row.item.classList.add('failed');
        row.state.textContent = '失败';
        row.detail.textContent = `请求失败：${error.message}`;
      }
    }
  } finally {
    await loadDocuments();
    await loadHealth();
  }
}

function parseFrame(frame) {
  let event = 'message';
  const dataLines = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event: ')) event = line.slice(7).trim();
    else if (line.startsWith('data: ')) dataLines.push(line.slice(6));
  }
  if (dataLines.length === 0) return null;
  try {
    return { event, data: JSON.parse(dataLines.join('\n')) };
  } catch {
    return null;
  }
}

function handleUploadEvent(frame, rows) {
  const parsed = parseFrame(frame);
  if (!parsed) return;
  const { event, data } = parsed;
  const row = rows.get(data.name);
  if (!row) return;

  if (event === 'progress') {
    const label = STAGE_LABEL[data.stage] || data.stage;
    row.state.textContent = label;
    if (typeof data.fraction === 'number') {
      row.bar.style.width = `${Math.round(data.fraction * 100)}%`;
    }
    row.detail.textContent = data.message || '';
    if (data.stage === 'embedding' && data.total) {
      row.bar.style.width = `${Math.round((data.current / data.total) * 100)}%`;
    }
  } else if (event === 'done') {
    row.item.classList.remove('failed');
    row.item.classList.add('done');
    row.bar.style.width = '100%';
    row.state.textContent = '完成';
    row.detail.textContent = `${data.parents} 段 / ${data.children} 块，用时 ${data.duration}s`;
  } else if (event === 'failed') {
    row.item.classList.add('failed');
    row.state.textContent = '失败';
    row.detail.textContent = `${STAGE_LABEL[data.stage] || data.stage}：${data.error}`;
  } else if (event === 'rejected') {
    row.item.classList.add('failed');
    row.state.textContent = '拒绝';
    row.detail.textContent = data.error;
  }
}

els.dropzone.addEventListener('click', () => els.fileInput.click());
els.dropzone.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' || event.key === ' ') {
    event.preventDefault();
    els.fileInput.click();
  }
});
els.fileInput.addEventListener('change', () => {
  uploadFiles(els.fileInput.files);
  els.fileInput.value = '';
});

for (const type of ['dragenter', 'dragover']) {
  els.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    els.dropzone.classList.add('dragover');
  });
}
for (const type of ['dragleave', 'drop']) {
  els.dropzone.addEventListener(type, (event) => {
    event.preventDefault();
    els.dropzone.classList.remove('dragover');
  });
}
els.dropzone.addEventListener('drop', (event) => {
  uploadFiles(event.dataTransfer?.files);
});

// ---------------------------------------------------------------------- chat

function addTurn(question) {
  const empty = els.messages.querySelector('.empty-state');
  if (empty) empty.remove();

  const turn = el('div', 'turn');
  const q = el('div', 'turn-question');
  q.append(el('span', '', question));
  turn.append(q);

  const answer = el('div', 'turn-answer streaming');
  const body = el('div', 'answer-body');
  answer.append(body);
  turn.append(answer);

  els.messages.append(turn);
  els.messages.scrollTop = els.messages.scrollHeight;
  return { turn, answer, body };
}

/**
 * Strip formatting noise from a snippet shown in a citation card.
 *
 * The chunk text deliberately keeps its heading line, because a parent that opens
 * with its heading is better context for the model. In a compact citation card that
 * line reads as literal "## ..." markup, so it is removed here rather than in the
 * stored text.
 */
function snippetText(text) {
  return text
    .replace(/^#{1,6}\s+.*$/gm, '')
    .replace(/\n{2,}/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function renderSources(container, data) {
  if (!data.sources || data.sources.length === 0) {
    // The answer text already says there is no content; a second notice saying the
    // same thing is pure noise.
    return;
  }

  const block = el('div', 'sources');
  block.append(el('div', 'sources-title', '来源'));
  const list = el('ul', 'source-list');

  for (const source of data.sources) {
    const item = el('li', 'source-item');
    const head = el('div', 'source-head');
    head.append(el('span', 'source-index', `[${source.index}]`));
    head.append(el('span', 'source-name', source.source));
    const where = [];
    if (source.page) where.push(`第 ${source.page} 页`);
    if (source.breadcrumb) where.push(source.breadcrumb);
    if (source.score !== null && source.score !== undefined) where.push(`相关度 ${source.score}`);
    if (where.length) head.append(el('span', 'source-where', where.join(' · ')));
    item.append(head);

    if (source.matched_text) {
      item.append(el('p', 'source-snippet', snippetText(source.matched_text)));
    }
    list.append(item);
  }
  block.append(list);

  if (data.rerank_error) {
    block.append(el('p', 'notice', `重排已跳过：${data.rerank_error}`));
  }
  container.append(block);
}

function renderDebug(container, data) {
  const details = el('details', 'debug');
  details.append(el('summary', '', '检索调试'));

  const stages = el('div', 'debug-stages');
  for (const stage of ['dense', 'sparse', 'fused', 'reranked']) {
    const block = el('div', 'debug-stage');
    block.append(el('h4', '', stage));
    const table = el('table');
    const head = el('tr');
    for (const label of ['id', 'channels', 'rank', 'score']) head.append(el('th', '', label));
    table.append(head);
    for (const row of (data[stage] || []).slice(0, 6)) {
      const tr = el('tr');
      tr.append(el('td', '', String(row.id).slice(-8)));
      tr.append(el('td', '', (row.channels || []).join('+')));
      const rank = stage === 'fused' ? row.fused_rank : stage === 'reranked' ? row.rerank_rank : stage === 'dense' ? row.dense_rank : row.sparse_rank;
      tr.append(el('td', '', String(rank ?? '－')));
      const score = stage === 'reranked' ? row.rerank_score : row.fused_score;
      tr.append(el('td', '', score === null || score === undefined ? '－' : Number(score).toFixed(4)));
      table.append(tr);
    }
    block.append(table);
    stages.append(block);
  }
  details.append(stages);
  container.append(details);
}

async function ask(question) {
  if (busy) return;
  busy = true;
  els.send.disabled = true;
  els.question.disabled = true;

  const { turn, answer, body } = addTurn(question);
  let answerText = '';
  let gotSources = false;
  let notes = null;

  const note = (message, isError = false) => {
    if (!notes) {
      notes = el('p', `notice${isError ? ' error' : ''}`, message);
      answer.append(notes);
    } else {
      notes.textContent = message;
      notes.classList.toggle('error', isError);
    }
  };

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        history,
        debug: els.debugCheck.checked,
      }),
    });

    if (!response.ok || !response.body) {
      const detail = await response.text();
      throw new Error(detail || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let boundary = buffer.indexOf('\n\n');
      while (boundary !== -1) {
        const frame = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf('\n\n');

        const parsed = parseFrame(frame);
        if (!parsed) continue;
        const { event, data } = parsed;

        if (event === 'status') {
          note(data.stage === 'rewriting' ? '正在理解问题…' : '正在检索…');
        } else if (event === 'rewritten') {
          note(`检索用问题：${data.query}`);
        } else if (event === 'sources') {
          gotSources = true;
          if (notes) { notes.remove(); notes = null; }
          renderSources(answer, data);
        } else if (event === 'token') {
          answerText += data.text;
          body.innerHTML = renderMarkdown(answerText);
          els.messages.scrollTop = els.messages.scrollHeight;
        } else if (event === 'debug') {
          renderDebug(answer, data);
        } else if (event === 'error') {
          answer.classList.remove('streaming');
          note(data.message, true);
        } else if (event === 'done') {
          answer.classList.remove('streaming');
          if (data.answer) {
            answerText = data.answer;
            body.innerHTML = renderMarkdown(answerText);
          }
          if (data.truncated) note('回答被截断：生成中断。', true);
          if (!gotSources && !data.empty) note('本次没有命中任何来源。');
        }
      }
    }
  } catch (error) {
    answer.classList.remove('streaming');
    note(`请求失败：${error.message}`, true);
  } finally {
    if (answerText && !answer.classList.contains('streaming')) {
      history.push({ question, answer: answerText });
      // The server keeps no session, so trim here to match its window.
      while (history.length > 6) history.shift();
    }
    busy = false;
    els.send.disabled = false;
    els.question.disabled = false;
    els.question.focus();
  }
}

els.composer.addEventListener('submit', (event) => {
  event.preventDefault();
  const question = els.question.value.trim();
  if (!question) return;
  els.question.value = '';
  els.question.style.height = 'auto';
  ask(question);
});

els.question.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    els.composer.requestSubmit();
  }
});

els.question.addEventListener('input', () => {
  els.question.style.height = 'auto';
  els.question.style.height = `${Math.min(200, els.question.scrollHeight)}px`;
});

// ---------------------------------------------------------------------- boot

loadDocuments();
warnIfUnhealthy();
els.question.focus();
const sidebar = document.getElementById('sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const mobileSidebarToggle = document.getElementById('mobile-sidebar-toggle');
const sidebarBackdrop = document.getElementById('sidebar-backdrop');
const navButtons = document.querySelectorAll('.nav-item[data-view]');
const chatView = document.getElementById('chat-view');
const documentsView = document.getElementById('documents-view');
const evaluationView = document.getElementById('evaluation-view');
const evalFrame = document.getElementById('eval-frame');

const messages = document.getElementById('messages');
const EMPTY_STATE_HTML = document.getElementById('empty-state').outerHTML;

const WELCOME_MESSAGES = [
  'Ready when you are.',
  'What are we digging into today?',
  'Ask me anything about your documents.',
  'What would you like to know?',
  'Let’s find what you’re looking for.',
];

function setRandomGreeting() {
  const heading = document.getElementById('empty-heading');
  if (!heading) return;
  heading.textContent = WELCOME_MESSAGES[Math.floor(Math.random() * WELCOME_MESSAGES.length)];
}

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const uploadStatus = document.getElementById('upload-status');
const stagedFileEl = document.getElementById('staged-file');
const stagedFileNameEl = document.getElementById('staged-file-name');
const stagedFileSizeEl = document.getElementById('staged-file-size');
const stagedFileCancelBtn = document.getElementById('staged-file-cancel');
const stagedFileSubmitBtn = document.getElementById('staged-file-submit');
const questionInput = document.getElementById('question');
const askBtn = document.getElementById('ask-btn');
const modelTrigger = document.getElementById('model-trigger');
const modelTriggerLabel = document.getElementById('model-trigger-label');
const modelMenu = document.getElementById('model-menu');
const themeToggle = document.getElementById('theme-toggle');

const pagePreviewPanel = document.getElementById('page-preview-panel');
const pagePreviewClose = document.getElementById('page-preview-close');
const pagePreviewTitle = document.getElementById('page-preview-title');
const pagePreviewImg = document.getElementById('page-preview-img');
const pagePreviewImageWrap = document.getElementById('page-preview-image-wrap');
const pagePreviewViewPageBtn = document.getElementById('page-preview-view-page-btn');

const imagesModal = document.getElementById('images-modal');
const imagesBackdrop = document.getElementById('images-backdrop');
const imagesClose = document.getElementById('images-close');
const imagesTitle = document.getElementById('images-title');
const imagesBody = document.getElementById('images-body');

const docSearch = document.getElementById('doc-search');
const docsRefreshBtn = document.getElementById('docs-refresh-btn');
const docsTableBody = document.getElementById('docs-table-body');
const docsEmpty = document.getElementById('docs-empty');
const docsPagination = document.getElementById('docs-pagination');

const PAGE_SIZE = 10;
let allDocuments = [];
let currentPage = 1;
let modelOptionsReady = false;
let isAsking = false;
let modelCatalog = [];
let selectedModelId = null;

const MODEL_STORAGE_KEY = 'generation-model';
const LEGACY_MODEL_STORAGE_KEY = 'openrouter-model';

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str || '';
  return div.innerHTML;
}

/* ---------- Model selection ---------- */

async function loadModels() {
  try {
    const res = await fetch('/api/models');
    if (!res.ok) throw new Error('Could not load models.');
    const data = await res.json();
    const models = Array.isArray(data.models) ? data.models : [];
    if (!models.length) throw new Error('No models are configured.');

    modelCatalog = models;
    const availableIds = new Set(models.filter(m => m.available !== false).map(m => m.id));

    const savedModel = localStorage.getItem(MODEL_STORAGE_KEY)
      || localStorage.getItem(LEGACY_MODEL_STORAGE_KEY);
    const preferredModel = availableIds.has(savedModel)
      ? savedModel
      : (availableIds.has(data.default_model) ? data.default_model : availableIds.values().next().value);

    if (preferredModel) {
      selectedModelId = preferredModel;
      modelOptionsReady = true;
      localStorage.setItem(MODEL_STORAGE_KEY, preferredModel);
      localStorage.removeItem(LEGACY_MODEL_STORAGE_KEY);
    } else {
      selectedModelId = null;
      modelOptionsReady = false;
    }

    renderModelMenu();
    updateModelTrigger();
    modelTrigger.disabled = isAsking || !modelOptionsReady;
    askBtn.disabled = isAsking || !modelOptionsReady;
  } catch (err) {
    modelCatalog = [];
    selectedModelId = null;
    modelOptionsReady = false;
    modelTriggerLabel.textContent = 'Models unavailable';
    modelTrigger.title = 'The model list could not be loaded.';
    modelTrigger.disabled = true;
    askBtn.disabled = true;
  }
}

function updateModelTrigger() {
  const model = modelCatalog.find(m => m.id === selectedModelId);
  modelTriggerLabel.textContent = model
    ? `${model.provider} · ${model.name}`
    : 'No model ready — see setup';
  modelTrigger.title = model
    ? `${model.blurb}${model.detail ? ` — ${model.detail}` : ''}`
    : 'Start the portable local runtime or configure an OpenRouter key.';
}

function renderModelMenu() {
  modelMenu.innerHTML = '';
  for (const model of modelCatalog) {
    const available = model.available !== false;
    const selected = model.id === selectedModelId;
    const badge = !available
      ? '<span class="model-option-badge unavailable">Unavailable</span>'
      : (model.local ? '<span class="model-option-badge">Local</span>' : '');
    const check = selected
      ? '<svg class="model-option-check" width="16" height="16" viewBox="0 0 24 24" fill="none"><path d="M5 13l4 4L19 7" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>'
      : '';

    const opt = document.createElement('button');
    opt.type = 'button';
    opt.className = `model-option${selected ? ' selected' : ''}`;
    opt.setAttribute('role', 'option');
    opt.setAttribute('aria-selected', String(selected));
    opt.dataset.id = model.id;
    opt.disabled = !available;
    if (model.detail) opt.title = model.detail;
    opt.innerHTML = `
      <span class="model-option-text">
        <span class="model-option-name">${escapeHtml(model.name)}${badge}</span>
        <span class="model-option-desc">${escapeHtml(model.blurb || '')}</span>
      </span>
      ${check}
    `;
    modelMenu.appendChild(opt);
  }
}

function openModelMenu() {
  if (modelTrigger.disabled) return;
  modelMenu.hidden = false;
  modelTrigger.setAttribute('aria-expanded', 'true');
}

function closeModelMenu() {
  modelMenu.hidden = true;
  modelTrigger.setAttribute('aria-expanded', 'false');
}

modelTrigger.addEventListener('click', () => {
  if (modelMenu.hidden) openModelMenu(); else closeModelMenu();
});

modelMenu.addEventListener('click', (e) => {
  const opt = e.target.closest('.model-option');
  if (!opt || opt.disabled) return;
  selectedModelId = opt.dataset.id;
  localStorage.setItem(MODEL_STORAGE_KEY, selectedModelId);
  updateModelTrigger();
  renderModelMenu();
  closeModelMenu();
});

document.addEventListener('click', (e) => {
  if (!modelMenu.hidden && !e.target.closest('.model-picker')) closeModelMenu();
});
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !modelMenu.hidden) closeModelMenu();
});

/* ---------- Sidebar toggle ---------- */

const MOBILE_SIDEBAR_QUERY = '(max-width: 860px)';

function openMobileSidebar() {
  sidebar.classList.add('mobile-open');
  sidebarBackdrop.classList.add('visible');
}
function closeMobileSidebar() {
  sidebar.classList.remove('mobile-open');
  sidebarBackdrop.classList.remove('visible');
}

sidebarToggle.addEventListener('click', () => {
  if (window.matchMedia(MOBILE_SIDEBAR_QUERY).matches) {
    closeMobileSidebar();
  } else {
    sidebar.classList.toggle('collapsed');
  }
});
mobileSidebarToggle.addEventListener('click', openMobileSidebar);
sidebarBackdrop.addEventListener('click', closeMobileSidebar);

/* ---------- View switching ---------- */

// The evaluation report is a separate document loaded in an <iframe>, its
// own dark-only stylesheet has no access to this page's data-theme — pass
// the current theme along as a query param so the report can match it
// (eval/run_eval.py's write_html reads this before first paint).
function evalReportUrl() {
  const theme = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  return `/api/eval/report?theme=${theme}`;
}

function showView(view) {
  chatView.hidden = view !== 'chat';
  documentsView.hidden = view !== 'documents';
  evaluationView.hidden = view !== 'evaluation';
  navButtons.forEach(btn => btn.classList.toggle('active', btn.dataset.view === view));
  if (view === 'documents') refreshDocuments();
  if (view === 'evaluation') evalFrame.src = evalReportUrl();
  closeMobileSidebar();
}

document.getElementById('nav-documents').addEventListener('click', () => showView('documents'));
document.getElementById('nav-evaluation').addEventListener('click', () => showView('evaluation'));
document.getElementById('eval-back-btn').addEventListener('click', () => showView('chat'));
document.getElementById('nav-new-chat').addEventListener('click', () => {
  messages.innerHTML = EMPTY_STATE_HTML;
  setRandomGreeting();
  chatView.classList.add('centered');
  questionInput.value = '';
  showView('chat');
  autoResize();
});

/* ---------- Theme toggle ---------- */

themeToggle.addEventListener('click', () => {
  const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  localStorage.setItem('theme', next);
  // Keep an already-open Evaluation report in sync rather than leaving it
  // on whichever theme was active when it first loaded.
  if (!evaluationView.hidden) evalFrame.src = evalReportUrl();
});

/* ---------- Documents: fetch, search, paginate, render ---------- */

function statusClass(status) {
  return { ready: 'status-ready', processing: 'status-processing', failed: 'status-failed' }[status] || '';
}

function formatUploaded(epochSeconds) {
  if (!epochSeconds) return '—';
  return new Date(epochSeconds * 1000).toLocaleString(undefined, {
    year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  });
}

async function refreshDocuments() {
  const res = await fetch('/api/documents');
  allDocuments = await res.json();
  allDocuments.sort((a, b) => (b.uploaded_at || 0) - (a.uploaded_at || 0));
  currentPage = 1;
  renderDocsTable();
}

function renderDocsTable() {
  const term = docSearch.value.trim().toLowerCase();
  const filtered = term
    ? allDocuments.filter(d => d.filename.toLowerCase().includes(term))
    : allDocuments;

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  currentPage = Math.min(currentPage, totalPages);
  const pageItems = filtered.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);

  docsTableBody.innerHTML = '';
  docsEmpty.hidden = filtered.length !== 0;

  for (const doc of pageItems) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td class="doc-name-cell">
        <span class="doc-file-title">${escapeHtml(doc.filename)}</span>
        ${doc.error ? `<div class="doc-error">${escapeHtml(doc.error)}</div>` : ''}
      </td>
      <td><span class="status ${statusClass(doc.status)}">${doc.status}</span></td>
      <td>${doc.chunk_count || 0}</td>
      <td>${doc.image_count
        ? `<button class="images-count-btn" data-id="${doc.document_id}" data-name="${escapeHtml(doc.filename)}">${doc.image_count}</button>`
        : '<span class="images-count-empty">—</span>'}</td>
      <td>${formatUploaded(doc.uploaded_at)}</td>
      <td class="doc-actions">
        ${doc._pending ? '' : `
        <button class="icon-btn danger" title="Delete" data-id="${doc.document_id}" aria-label="Delete document">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none"><path d="M4 7h16M9 7V4h6v3M6 7l1 13a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2l1-13" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"/></svg>
        </button>
        `}
      </td>
    `;
    docsTableBody.appendChild(tr);
  }

  renderPagination(totalPages);
}

function renderPagination(totalPages) {
  if (totalPages <= 1) { docsPagination.innerHTML = ''; return; }
  let html = `<button class="page-btn" data-page="${currentPage - 1}" ${currentPage === 1 ? 'disabled' : ''}>‹</button>`;
  for (let p = 1; p <= totalPages; p++) {
    html += `<button class="page-btn ${p === currentPage ? 'active' : ''}" data-page="${p}">${p}</button>`;
  }
  html += `<button class="page-btn" data-page="${currentPage + 1}" ${currentPage === totalPages ? 'disabled' : ''}>›</button>`;
  docsPagination.innerHTML = html;
}

docsPagination.addEventListener('click', (e) => {
  const btn = e.target.closest('.page-btn');
  if (!btn || btn.disabled) return;
  currentPage = Number(btn.dataset.page);
  renderDocsTable();
});

docSearch.addEventListener('input', () => { currentPage = 1; renderDocsTable(); });
docsRefreshBtn.addEventListener('click', () => { clearUploadStatus(); refreshDocuments(); });

docsTableBody.addEventListener('click', async (e) => {
  const deleteBtn = e.target.closest('.icon-btn.danger');
  if (deleteBtn) {
    deleteBtn.disabled = true;
    await fetch(`/api/documents/${deleteBtn.dataset.id}`, { method: 'DELETE' });
    clearUploadStatus();
    refreshDocuments();
    return;
  }
  const imagesBtn = e.target.closest('.images-count-btn');
  if (imagesBtn) openImages(imagesBtn.dataset.id, imagesBtn.dataset.name);
});

/* ---------- Images gallery ---------- */

function closeImages() {
  imagesModal.hidden = true;
  imagesBody.innerHTML = '';
}

async function openImages(documentId, filename) {
  imagesTitle.textContent = `${filename} — images`;
  imagesBody.innerHTML = '';
  imagesModal.hidden = false;

  const res = await fetch(`/api/documents/${documentId}/images`);
  const images = res.ok ? await res.json() : [];

  if (!images.length) {
    imagesBody.innerHTML = '<div class="images-empty">No images extracted for this document.</div>';
    return;
  }

  imagesBody.innerHTML = images.map(f => `
    <div class="image-card">
      <img src="/api/documents/${documentId}/images/${f.index}" alt="Image ${f.index + 1}" loading="lazy" />
      <div class="image-caption">
        ${f.page ? `<div class="image-page">Page ${f.page}</div>` : ''}
        <div class="image-description">${escapeHtml(f.description || 'No description available.')}</div>
      </div>
    </div>
  `).join('');
}

imagesClose.addEventListener('click', closeImages);
imagesBackdrop.addEventListener('click', closeImages);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !imagesModal.hidden) closeImages();
});

/* ---------- Upload ---------- */

let stagedFile = null;

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function clearUploadStatus() {
  uploadStatus.textContent = '';
  uploadStatus.classList.remove('error');
}

function stageFile(file) {
  if (!file) return;
  stagedFile = file;
  stagedFileNameEl.textContent = file.name;
  stagedFileSizeEl.textContent = formatFileSize(file.size);
  stagedFileEl.hidden = false;
  dropzone.hidden = true;
  clearUploadStatus();
}

function clearStagedFile() {
  stagedFile = null;
  fileInput.value = '';
  stagedFileEl.hidden = true;
  dropzone.hidden = false;
}

async function submitStagedFile(file) {
  const formData = new FormData();
  formData.append('file', file);

  stagedFileSubmitBtn.disabled = true;
  stagedFileCancelBtn.disabled = true;

  // Optimistic row so the file shows as "processing" in the table for the
  // full (synchronous, potentially slow on first run) ingestion request,
  // instead of leaving the table looking unchanged until it completes.
  const pendingId = `pending-${file.name}-${file.size}`;
  allDocuments.unshift({
    document_id: pendingId,
    filename: file.name,
    status: 'processing',
    chunk_count: 0,
    uploaded_at: Date.now() / 1000,
    error: null,
    _pending: true,
  });
  currentPage = 1;
  renderDocsTable();
  clearStagedFile();

  try {
    const res = await fetch('/api/documents', { method: 'POST', body: formData });
    const data = await res.json();
    uploadStatus.textContent = res.ok ? '' : (data.detail || 'Failed to process document.');
    uploadStatus.classList.toggle('error', !res.ok);
  } catch (err) {
    uploadStatus.textContent = 'Network error.';
    uploadStatus.classList.add('error');
  } finally {
    stagedFileSubmitBtn.disabled = false;
    stagedFileCancelBtn.disabled = false;
    refreshDocuments();
  }
}

stagedFileCancelBtn.addEventListener('click', clearStagedFile);
stagedFileSubmitBtn.addEventListener('click', () => { if (stagedFile) submitStagedFile(stagedFile); });

fileInput.addEventListener('change', () => stageFile(fileInput.files[0]));

['dragover', 'dragenter'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); })
);
['dragleave', 'drop'].forEach(evt =>
  dropzone.addEventListener(evt, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); })
);
dropzone.addEventListener('drop', (e) => stageFile(e.dataTransfer.files[0]));

/* ---------- Chat ---------- */

if (window.mermaid) {
  mermaid.initialize({
    startOnLoad: false,
    theme: document.documentElement.getAttribute('data-theme') === 'dark' ? 'dark' : 'default',
  });
}

let chartInstanceId = 0;

// Strips [n] citation markers from display text. They're only needed
// server-side to map claims back to source chunks — the clickable source
// chips already show which documents/pages were used, so showing the raw
// chunk-position number ("[3]") inline just confuses readers.
function stripCitationMarkers(text) {
  return text
    .replace(/\[(\d+)\](?!\()/g, '')
    .replace(/[ \t]+([.,;:!?])/g, '$1')
    .replace(/[ \t]{2,}/g, ' ');
}

// Drops a trailing "Note: ..." caveat paragraph. The model is asked not to
// add one (prompt.yaml), but small local models don't always comply, so
// this is the reliable backstop.
function stripTrailingNote(text) {
  const paragraphs = text.split(/\n\s*\n/);
  const last = paragraphs[paragraphs.length - 1];
  if (last && /^[*_\s]*note:/i.test(last.trim())) paragraphs.pop();
  return paragraphs.join('\n\n');
}

// Renders the assistant's raw Markdown into `container` as sanitized HTML,
// then upgrades any ```mermaid fence into a diagram and any ```chart fence
// (a {type, labels, datasets} JSON spec) into a Chart.js canvas.
function renderAssistantContent(container, text) {
  const html = marked.parse(stripTrailingNote(stripCitationMarkers(text)), { gfm: true, breaks: true });
  container.innerHTML = window.DOMPurify ? DOMPurify.sanitize(html) : html;
  container.classList.add('markdown-rendered');

  container.querySelectorAll('table').forEach((table) => {
    // The "Source" column only ever carried [n] markers, which are always
    // stripped from display — drop the now-empty column entirely.
    const headerCells = [...table.querySelectorAll('thead th')];
    const sourceIdx = headerCells.findIndex((th) => th.textContent.trim().toLowerCase() === 'source');
    if (sourceIdx !== -1) {
      table.querySelectorAll('tr').forEach((row) => row.children[sourceIdx]?.remove());
    }

    const scroller = document.createElement('div');
    scroller.className = 'table-scroll';
    table.replaceWith(scroller);
    scroller.appendChild(table);
  });

  const mermaidBlocks = container.querySelectorAll('pre code.language-mermaid');
  if (mermaidBlocks.length && window.mermaid) {
    mermaidBlocks.forEach((code) => {
      const div = document.createElement('div');
      div.className = 'mermaid';
      div.textContent = code.textContent;
      code.closest('pre').replaceWith(div);
    });
    mermaid.run({ nodes: container.querySelectorAll('.mermaid') });
  }

  if (window.Chart) {
    container.querySelectorAll('pre code.language-chart').forEach((code) => {
      let spec;
      try {
        spec = JSON.parse(code.textContent);
      } catch (_) {
        return; // Leave malformed chart specs as plain code blocks.
      }
      const wrap = document.createElement('div');
      wrap.className = 'chat-chart-wrap';
      const canvas = document.createElement('canvas');
      canvas.id = `chat-chart-${++chartInstanceId}`;
      wrap.appendChild(canvas);
      code.closest('pre').replaceWith(wrap);
      new Chart(canvas.getContext('2d'), {
        type: spec.type || 'bar',
        data: { labels: spec.labels || [], datasets: spec.datasets || [{ data: spec.data || [] }] },
        options: { responsive: true, maintainAspectRatio: false, ...(spec.options || {}) },
      });
    });
  }
}

const TYPE_MS_PER_CHUNK = 5;
const TYPE_CHARS_PER_CHUNK = 4;

function typeText(el, text) {
  el.classList.add('typing');
  return new Promise((resolve) => {
    let i = 0;
    (function tick() {
      i += TYPE_CHARS_PER_CHUNK + Math.floor(Math.random() * TYPE_CHARS_PER_CHUNK);
      el.textContent = text.slice(0, i);
      messages.scrollTop = messages.scrollHeight;
      if (i < text.length) {
        setTimeout(tick, TYPE_MS_PER_CHUNK);
      } else {
        el.classList.remove('typing');
        resolve();
      }
    })();
  });
}

function addMessage(role, text, citations) {
  const stray = document.getElementById('empty-state');
  if (stray) stray.remove();
  chatView.classList.remove('centered');
  const wrap = document.createElement('div');
  wrap.className = `msg ${role}`;
  const citationsHtml = (citations || []).map(c => {
    const page = c.page_start ? (c.page_start === c.page_end ? `p.${c.page_start}` : `pp.${c.page_start}-${c.page_end}`) : '';
    // Two chunks (e.g. a table and a paragraph) can share the same page, so
    // the chip needs more than title+page to read as distinct — fall back
    // to a content-type label (Table/Figure) when there's no heading.
    const typeLabel = c.content_type === 'table' ? 'Table' : c.content_type === 'image' ? 'Figure' : '';
    const disambiguator = c.heading || typeLabel;
    const tooltip = [
      c.heading || '',
      c.document_category ? `Category: ${c.document_category}` : '',
      (c.tags && c.tags.length) ? `Tags: ${c.tags.join(', ')}` : '',
    ].filter(Boolean).join(' — ');
    const hasImage = c.content_type === 'image' && !!c.chunk_id;
    const clickable = (c.document_id && c.page_start && ((c.positions && c.positions.length) || hasImage)) ? ' clickable' : '';
    return `<span class="citation-chip${clickable}" title="${escapeHtml(tooltip)}">${escapeHtml(c.title)} ${page}${disambiguator ? ` · ${escapeHtml(disambiguator)}` : ''}</span>`;
  }).join('');

  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  wrap.appendChild(bubble);
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;

  const appendCitations = () => {
    if (!citationsHtml) return;
    const citEl = document.createElement('div');
    citEl.className = 'citations';
    citEl.innerHTML = citationsHtml;
    // Attach the full citation object directly to its chip element (not
    // serialized into the DOM) so the click handler below can read it back
    // without needing to HTML-escape a JSON blob into an attribute.
    const chips = citEl.querySelectorAll('.citation-chip');
    (citations || []).forEach((c, i) => { chips[i].__citation = c; });
    wrap.appendChild(citEl);
    messages.scrollTop = messages.scrollHeight;
  };

  const render = () => {
    if (role === 'assistant') {
      renderAssistantContent(bubble, text);
      messages.scrollTop = messages.scrollHeight;
    } else {
      bubble.textContent = text;
    }
  };

  const done = (role === 'assistant' ? typeText(bubble, text) : Promise.resolve())
    .then(render)
    .then(appendCitations);

  return { wrap, done };
}

/* ---------- Citation preview (page or source image) ---------- */

let currentPreviewCitation = null;

function closePagePreview() {
  pagePreviewPanel.hidden = true;
  pagePreviewImg.src = '';
  pagePreviewImageWrap.querySelectorAll('.page-preview-highlight').forEach(el => el.remove());
  pagePreviewViewPageBtn.hidden = true;
  currentPreviewCitation = null;
}

function drawPageHighlights(citation) {
  pagePreviewImageWrap.querySelectorAll('.page-preview-highlight').forEach(el => el.remove());
  const [, , , , , pageWidth, pageHeight] = citation.positions[0];
  for (const [pageNo, left, top, right, bottom] of citation.positions) {
    if (pageNo !== citation.page_start) continue;
    const box = document.createElement('div');
    box.className = 'page-preview-highlight';
    box.style.left = `${(left / pageWidth) * 100}%`;
    box.style.top = `${(top / pageHeight) * 100}%`;
    box.style.width = `${((right - left) / pageWidth) * 100}%`;
    box.style.height = `${((bottom - top) / pageHeight) * 100}%`;
    pagePreviewImageWrap.appendChild(box);
  }
}

// Image chunks are minted as `${document_id}::image-{index}` (see
// ingestion/chunker.py build_image_chunks) — the index is the only handle
// we have back to the extracted image file.
function extractImageIndex(citation) {
  const match = /::image-(\d+)$/.exec(citation.chunk_id || '');
  return match ? match[1] : null;
}

function openPagePreview(citation) {
  currentPreviewCitation = citation;
  pagePreviewTitle.textContent = `${citation.title} — page ${citation.page_start}`;
  pagePreviewPanel.hidden = false;
  pagePreviewViewPageBtn.hidden = true;
  pagePreviewImg.onload = () => drawPageHighlights(citation);
  pagePreviewImg.src = `/api/documents/${citation.document_id}/pages/${citation.page_start}`;
}

// Renders the actual extracted figure/table image the chunk was built
// from, instead of the whole page it happens to sit on — with a link back
// to the full page for context.
function openImagePreview(citation) {
  const index = extractImageIndex(citation);
  if (index === null) { openPagePreview(citation); return; }
  currentPreviewCitation = citation;
  pagePreviewImageWrap.querySelectorAll('.page-preview-highlight').forEach(el => el.remove());
  pagePreviewTitle.textContent = `${citation.title} — figure (p.${citation.page_start})`;
  pagePreviewPanel.hidden = false;
  pagePreviewViewPageBtn.hidden = !citation.page_start;
  pagePreviewImg.onload = null;
  pagePreviewImg.src = `/api/documents/${citation.document_id}/images/${index}`;
}

function openCitationPreview(citation) {
  if (citation.content_type === 'image') {
    openImagePreview(citation);
  } else {
    openPagePreview(citation);
  }
}

messages.addEventListener('click', (e) => {
  const chip = e.target.closest('.citation-chip.clickable');
  if (chip && chip.__citation) openCitationPreview(chip.__citation);
});

pagePreviewViewPageBtn.addEventListener('click', () => {
  if (currentPreviewCitation) openPagePreview(currentPreviewCitation);
});

pagePreviewClose.addEventListener('click', closePagePreview);
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && !pagePreviewPanel.hidden) closePagePreview();
});

// Labels for each coarse phase of the query pipeline (embed -> search ->
// rerank -> generate), reported live by the backend as `/api/query` streams
// stage events, so this mirrors the actual [QUERY] trace steps rather than
// a guessed timer.
const THINKING_STAGE_LABELS = {
  thinking: 'Thinking',
  embedding: 'Embedding',
  searching: 'Searching',
  reranking: 'Reranking',
  generating: 'Generating',
};

function addThinkingIndicator() {
  const wrap = document.createElement('div');
  wrap.className = 'msg assistant thinking';
  wrap.innerHTML = `
    <div class="bubble">
      <span class="thinking-logo" aria-hidden="true">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none">
          <path d="M12 3.5c2 0 3.1 1.2 2.9 3.2 1.7-1 3.3-.6 4.3 1.1 1 1.8.5 3.4-1.3 4.2 1.8.8 2.3 2.4 1.3 4.2-1 1.7-2.6 2.1-4.3 1.1.2 2-.9 3.2-2.9 3.2s-3.1-1.2-2.9-3.2c-1.7 1-3.3.6-4.3-1.1-1-1.8-.5-3.4 1.3-4.2-1.8-.8-2.3-2.4-1.3-4.2 1-1.7 2.6-2.1 4.3-1.1C8.9 4.7 10 3.5 12 3.5Z" stroke="currentColor" stroke-width="2.1" stroke-linecap="round" stroke-linejoin="round"/>
          <circle cx="12" cy="12" r="2" fill="currentColor"/>
        </svg>
      </span>
      <span class="thinking-text">${THINKING_STAGE_LABELS.thinking}</span>
    </div>
  `;
  messages.appendChild(wrap);
  messages.scrollTop = messages.scrollHeight;
  return wrap;
}

function setThinkingStage(wrap, stage) {
  const label = wrap.querySelector('.thinking-text');
  if (label) label.textContent = THINKING_STAGE_LABELS[stage] || THINKING_STAGE_LABELS.thinking;
}

const QUESTION_MAX_HEIGHT = 165;

function autoResize() {
  questionInput.style.height = 'auto';
  const fullHeight = questionInput.scrollHeight;
  questionInput.style.height = Math.min(fullHeight, QUESTION_MAX_HEIGHT) + 'px';
  questionInput.style.overflowY = fullHeight > QUESTION_MAX_HEIGHT ? 'auto' : 'hidden';
}

// `/api/query` streams newline-delimited JSON: zero or more `{stage}` events
// as the pipeline progresses, followed by exactly one final `{answer, ...}`
// or `{error}` event. Returns that final event, or null if the stream ended
// without one.
async function readQueryStream(body, onStage) {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let final = null;

  const handleLine = (line) => {
    if (!line) return;
    let event;
    try { event = JSON.parse(line); } catch (_) { return; }
    if (event.stage) onStage(event.stage);
    else final = event;
  };

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newlineIndex;
    while ((newlineIndex = buffer.indexOf('\n')) >= 0) {
      handleLine(buffer.slice(0, newlineIndex).trim());
      buffer = buffer.slice(newlineIndex + 1);
    }
  }
  handleLine(buffer.trim());

  return final;
}

async function ask() {
  if (isAsking || !modelOptionsReady) return;
  const question = questionInput.value.trim();
  if (!question) return;

  const model = selectedModelId || null;
  isAsking = true;
  addMessage('user', question);
  questionInput.value = '';
  autoResize();
  askBtn.disabled = true;
  modelTrigger.disabled = true;
  closeModelMenu();
  const thinking = addThinkingIndicator();
  let response;

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, model }),
    });

    if (!res.ok) {
      let detail = 'Something went wrong.';
      try { detail = (await res.json()).detail || detail; } catch (_) { /* body wasn't JSON */ }
      thinking.remove();
      response = addMessage('assistant', detail, null);
    } else if (!res.body) {
      // Streaming bodies aren't supported here — fall back to a plain parse.
      const data = await res.json();
      thinking.remove();
      response = addMessage('assistant', data.answer, data.citations);
    } else {
      const final = await readQueryStream(res.body, (stage) => setThinkingStage(thinking, stage));
      thinking.remove();
      if (!final) {
        response = addMessage('assistant', 'Something went wrong.', null);
      } else if (final.error) {
        response = addMessage('assistant', final.error, null);
      } else {
        response = addMessage('assistant', final.answer, final.citations);
      }
    }
  } catch (err) {
    thinking.remove();
    response = addMessage('assistant', 'Network error — is the server running?', null);
  }

  try {
    await response.done;
  } finally {
    isAsking = false;
    askBtn.disabled = !modelOptionsReady;
    modelTrigger.disabled = !modelOptionsReady;
    questionInput.focus();
  }
}

askBtn.addEventListener('click', ask);
questionInput.addEventListener('input', autoResize);
questionInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});

setRandomGreeting();
showView('chat');
loadModels();

// Small client for the MEDEC API. No dependencies.
const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const pct = (x) => (x == null ? "-" : (100 * x).toFixed(1) + "%");
const num = (x, d = 3) => (x == null || Number.isNaN(x) ? "-" : Number(x).toFixed(d));

const STATUS_LABEL = { pending: "pendiente", running: "en curso", done: "listo", skipped: "omitido", error: "error" };
let defaults = { weight: 0.7, threshold: 0.5 };
let currentExample = null;
let pollTimer = null;

async function api(path, options = {}) {
  const resp = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`);
  return resp.json();
}

// ------------------------------------------------------------------ status
async function loadStatus() {
  try {
    const s = await api("/api/health");
    defaults = s.defaults;
    const pills = [
      ["BiomedBERT", s.bert],
      ["LLM (Qwen)", s.llm],
      [s.dspy_compiled ? "DSPy compilado" : "DSPy sin compilar", s.dspy_compiled],
      ["Corrector", s.corrector],
    ];
    $("#status").innerHTML = pills.map(([name, on]) => `<span class="pill ${on ? "on" : "off"}">${esc(name)}</span>`).join("");
    if ($("#weight").dataset.init !== "1") resetSliders();
    if (!s.bert) $("#use_bert").checked = false;
  } catch (e) {
    $("#status").innerHTML = `<span class="pill off">API sin conexión</span>`;
  }
}

function resetSliders() {
  $("#weight").value = defaults.weight;
  $("#threshold").value = defaults.threshold;
  $("#weight").dataset.init = "1";
  updateSliderLabels();
}

function updateSliderLabels() {
  $("#weight-value").textContent = Number($("#weight").value).toFixed(2);
  $("#threshold-value").textContent = Number($("#threshold").value).toFixed(3);
}

// ---------------------------------------------------------------- examples
async function loadExamples() {
  const list = await api("/api/examples");
  const sel = $("#example");
  sel.innerHTML = `<option value="">(texto propio)</option>` + list.map((e) =>
    `<option value="${esc(e.text_id)}">${esc(e.text_id)} · ${e.error_flag ? "con error" : "sin error"} · ${esc(e.preview)}...</option>`).join("");
}

async function selectExample(textId) {
  currentExample = null;
  if (!textId) return;
  const ex = await api(`/api/examples/${encodeURIComponent(textId)}`);
  currentExample = ex;
  $("#text").value = ex.sentences.map((s, i) => `${i} ${s}`).join("\n");
}

// ---------------------------------------------------------------- analysis
function renderSteps(steps) {
  $("#steps").innerHTML = steps.map((s) =>
    `<li data-status="${s.status}"><span>${esc(s.label)}</span><span class="state">${STATUS_LABEL[s.status]}${s.seconds != null ? ` · ${s.seconds}s` : ""}</span></li>`).join("");
}

async function run() {
  const text = $("#text").value.trim();
  if (!text) return alertResult("Escribe un texto o elige un ejemplo.");
  const body = {
    use_bert: $("#use_bert").checked,
    use_dspy: $("#use_dspy").checked,
    correct: $("#correct").checked,
    weight: Number($("#weight").value),
    threshold: Number($("#threshold").value),
  };
  const exampleText = currentExample ? currentExample.sentences.map((s, i) => `${i} ${s}`).join("\n") : null;
  if (currentExample && text === exampleText) body.text_id = currentExample.text_id;
  else body.text = text;
  // With a single detector the sliders do not apply: BERT uses its own tuned threshold, DSPy its own answer
  if (!(body.use_bert && body.use_dspy)) { delete body.weight; delete body.threshold; }

  $("#run").disabled = true;
  $("#result").classList.add("hidden");
  try {
    const { job_id } = await api("/api/analyze", { method: "POST", body: JSON.stringify(body) });
    poll(job_id, Date.now());
  } catch (e) {
    $("#run").disabled = false;
    alertResult(e.message);
  }
}

function poll(jobId, started) {
  clearTimeout(pollTimer);
  api(`/api/jobs/${jobId}`).then((job) => {
    renderSteps(job.steps);
    const elapsed = ((Date.now() - started) / 1000).toFixed(0);
    $("#job-info").textContent = job.status === "queued"
      ? `En cola (${job.queue_position} trabajos antes) · ${elapsed}s`
      : `Trabajo ${job.id} · ${elapsed}s`;
    if (job.status === "done") { $("#run").disabled = false; renderResult(job.result); }
    else if (job.status === "error") { $("#run").disabled = false; alertResult(job.error); }
    else pollTimer = setTimeout(() => poll(jobId, started), 1500);
  }).catch((e) => { $("#run").disabled = false; alertResult(e.message); });
}

function alertResult(msg) {
  const box = $("#result");
  box.classList.remove("hidden");
  box.innerHTML = `<p class="error-msg">${esc(msg)}</p>`;
}

function bar(cls, value) {
  if (value == null) return "";
  return `<div class="bar ${cls}" title="${cls}: ${num(value)}"><span style="width:${Math.max(0, Math.min(1, value)) * 100}%"></span></div>`;
}

function renderResult(r) {
  const box = $("#result");
  box.classList.remove("hidden");
  const gold = r.gold;
  let html = r.error_flag
    ? `<div class="verdict error">Hay un error en la oración ${r.error_sentence_id}</div>`
    : `<div class="verdict ok">El texto se considera correcto</div>`;

  if (gold) {
    const flagHit = gold.error_flag === r.error_flag;
    const sidHit = gold.error_sentence_id === r.error_sentence_id;
    html += `<div class="gold">Etiqueta real: ${gold.error_flag ? `error en la oración ${gold.error_sentence_id}` : "sin error"} ·
      A <b class="${flagHit ? "hit" : "miss"}">${flagHit ? "acierto" : "fallo"}</b> ·
      B <b class="${sidHit ? "hit" : "miss"}">${sidHit ? "acierto" : "fallo"}</b></div>`;
  }

  const d = r.decision;
  html += `<div class="hint">Decisión: ${d.mode === "combined" ? `combinación con w = ${num(d.weight, 2)} y umbral ${num(d.threshold)}` : d.mode === "bert" ? `solo BERT, umbral ${num(d.threshold)}` : "solo el LLM (su propia respuesta)"}</div>`;
  html += `<div class="legend">${r.p_bert ? `<span><i style="background:var(--bert)"></i>p_bert</span>` : ""}${r.s_dspy ? `<span><i style="background:var(--dspy)"></i>s_dspy</span>` : ""}<span><i style="background:var(--combined)"></i>puntaje final</span>${gold && gold.error_flag ? `<span>borde punteado = oración con error real</span>` : ""}</div>`;
  html += `<div class="sentences">` + r.sentences.map((s, i) => {
    const cls = [r.error_flag && i === r.error_sentence_id ? "flagged" : "", gold && gold.error_sentence_id === i ? "gold-error" : ""].join(" ");
    return `<div class="sentence ${cls}"><span class="sid">${i}</span><span class="stext">${esc(s)}</span>
      <div class="bars">${bar("bert", r.p_bert?.[i])}${bar("dspy", r.s_dspy?.[i])}${bar("combined", r.scores[i])}</div></div>`;
  }).join("") + `</div>`;

  if (r.corrected_sentence) {
    html += `<h3>Corrección propuesta</h3><div class="diff">` + r.diff.map((p) =>
      p.op === "equal" ? esc(p.text) : p.op === "delete" ? `<del>${esc(p.text)}</del>` : `<ins>${esc(p.text)}</ins>`).join(" ") + `</div>`;
    if (r.quality) {
      html += `<p class="hint">ROUGE-L frente a la original: ${num(r.quality.rouge_l)}${r.quality.doubtful ? ` · <b class="error-msg">dudosa (${r.quality.reason === "identical" ? "quedó igual" : "cambió demasiado"})</b>` : " · pasa el control de calidad"}</p>`;
    }
    if (gold && gold.corrected_sentence) html += `<p class="hint">Corrección de referencia: ${esc(gold.corrected_sentence)}</p>`;
    if (r.few_shot_examples?.length) {
      html += `<details><summary>Ejemplos de train usados en el prompt (${r.few_shot_examples.length})</summary><ul>` +
        r.few_shot_examples.map((e) => `<li>${esc(e.error_sentence)} → <b>${esc(e.corrected_sentence)}</b> <small>(similitud ${num(e.similarity, 2)})</small></li>`).join("") + `</ul></details>`;
    }
  }
  if (r.dspy) {
    html += `<details><summary>Razonamiento del LLM (DSPy)</summary><p class="hint">Respuesta: ${r.dspy.failed ? "falló el formato" : r.dspy.has_error ? `error en la oración ${r.dspy.error_sentence_id}, confianza ${num(r.dspy.confidence, 2)}` : "sin error"}</p><pre>${esc(r.dspy.reasoning || "(vacío)")}</pre></details>`;
  }
  box.innerHTML = `<h2>Resultado</h2>` + html;
}

// ----------------------------------------------------------------- metrics
function table(rows, columns) {
  if (!rows?.length) return "<p class='hint'>Sin datos.</p>";
  return `<div class="table-wrap"><table><thead><tr>${columns.map(([, label]) => `<th>${label}</th>`).join("")}</tr></thead><tbody>` +
    rows.map((r) => `<tr>${columns.map(([key, , fmt]) => `<td class="${fmt ? "num" : ""}">${fmt ? fmt(r[key]) : esc(r[key])}</td>`).join("")}</tr>`).join("") +
    `</tbody></table></div>`;
}

async function loadMetrics() {
  const box = $("#metrics");
  try {
    const m = await api("/api/metrics");
    const t = m.test;
    if (!t) { box.innerHTML = "<p>Todavía no hay resultados (ejecuta scripts/run_evaluate.py).</p>"; return; }
    let html = `<h2>Resultados en el conjunto de prueba (${t.n_texts} textos)</h2>`;
    html += `<h3>Detección: subtareas A y B (Accuracy)</h3>` + table(t.detection, [
      ["system", "Sistema"], ["flag_acc", "A: ¿hay error?", pct], ["sentence_acc", "B: ¿qué oración?", pct], ["sentence_acc_on_errors", "B solo en textos con error", pct]]);
    if (t.correction?.length) {
      html += `<h3>Corrección: subtarea C</h3><p class="hint">"Oficial": si ambos son NA vale 1 y si solo uno es NA vale 0. "Pares": solo textos donde la referencia y la predicción tienen corrección.</p>` +
        table(t.correction, [["system", "Sistema"], ["rouge1_official", "ROUGE-1 oficial", (x) => num(x)], ["bertscore_official", "BERTScore oficial", (x) => num(x)],
          ["rouge1_pairs", "ROUGE-1 pares", (x) => num(x)], ["bertscore_pairs", "BERTScore pares", (x) => num(x)], ["n_pairs", "Pares", (x) => x ?? "-"]]);
    }
    if (m.combine?.combined_by_weight) {
      html += `<h3>Búsqueda del peso w en validación</h3><p class="hint">w = 1 es solo BERT y w = 0 es solo DSPy. Mejor umbral para cada w.</p>` +
        table(m.combine.combined_by_weight, [["w", "w", (x) => num(x, 1)], ["threshold", "Umbral", (x) => num(x)], ["flag_acc", "A", pct], ["sentence_acc", "B", pct]]);
    }
    if (t.error_analysis) {
      html += `<h3>Análisis de errores del pipeline (B en textos con error)</h3>` +
        table(t.error_analysis.by_position, [["position", "Posición"], ["mean", "Aciertos", pct], ["count", "Textos", (x) => x]]) +
        table(t.error_analysis.by_error_type, [["error_type", "Tipo aproximado"], ["mean", "Aciertos", pct], ["count", "Textos", (x) => x]]);
    }
    box.innerHTML = html;
  } catch (e) {
    box.innerHTML = `<p class="error-msg">${esc(e.message)}</p>`;
  }
}

// -------------------------------------------------------------------- init
document.querySelectorAll(".tab").forEach((btn) => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${btn.dataset.tab}`));
  if (btn.dataset.tab === "metrics") loadMetrics();
}));
$("#example").addEventListener("change", (e) => selectExample(e.target.value));
$("#random").addEventListener("click", () => {
  const opts = [...$("#example").options].filter((o) => o.value);
  if (!opts.length) return;
  const pick = opts[Math.floor(Math.random() * opts.length)].value;
  $("#example").value = pick;
  selectExample(pick);
});
$("#weight").addEventListener("input", updateSliderLabels);
$("#threshold").addEventListener("input", updateSliderLabels);
$("#reset").addEventListener("click", resetSliders);
$("#run").addEventListener("click", run);

renderSteps([["prepare", "1-2. Preparar oraciones"], ["bert", "3a. BiomedBERT"], ["dspy", "3b. Qwen + DSPy"],
  ["combine", "3c. Combinar y decidir"], ["correct", "4. Corregir"], ["quality", "5. Revisar corrección"]]
  .map(([id, label]) => ({ id, label, status: "pending", seconds: null })));
loadStatus();
loadExamples().catch(() => {});
setInterval(loadStatus, 30000);

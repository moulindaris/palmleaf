// main.js — handles upload form, pipeline display, and UI interactions

const form        = document.getElementById("uploadForm");
const fileInput   = document.getElementById("imageInput");
const fileLabel   = document.getElementById("fileLabel");
const fileDropZone = fileInput?.closest(".file-label");
const submitBtn   = document.getElementById("submitBtn");
const submitLabel = submitBtn?.textContent;
const statusMsg   = document.getElementById("statusMsg");
const progressBar = document.getElementById("progressBar");
const results     = document.getElementById("results");
const previewBox  = document.getElementById("uploadPreview");
const previewImg  = document.getElementById("previewImage");
const previewName = document.getElementById("previewName");
const previewMeta = document.getElementById("previewMeta");
const modelStatus = document.getElementById("modelStatus");
const batchResults = document.getElementById("batchResults");
const titleInput = form?.querySelector('[name="title"]');
const modeInputs = form?.querySelectorAll('[name="uploadMode"]');
const MAX_FILE_SIZE = 32 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = ["jpg", "jpeg", "png", "tif", "tiff", "bmp", "pdf"];
let previewUrl = null;

function setStatus(message, state = "") {
  if (!statusMsg) return;
  statusMsg.textContent = message;
  statusMsg.dataset.state = state;
}

function setPipelineStage(activeStage, completed = false) {
  const stages = ["upload", "restore", "ocr", "context"];
  const activeIndex = stages.indexOf(activeStage);
  document.querySelectorAll(".pipeline-step").forEach((stage, index) => {
    stage.classList.toggle("active", !completed && index === activeIndex);
    stage.classList.toggle("complete", completed || (activeIndex > index));
  });
}

function formatBytes(bytes) {
  if (!bytes) return "0 bytes";
  const units = ["bytes", "KB", "MB"];
  const power = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / Math.pow(1024, power)).toFixed(power ? 1 : 0)} ${units[power]}`;
}

function validateAndPreview(file) {
  if (!file) return false;
  const extension = file.name.split(".").pop().toLowerCase();
  if (!ACCEPTED_EXTENSIONS.includes(extension)) {
    setStatus("Choose a JPG, PNG, TIFF, BMP, or PDF file to continue.", "error");
    fileInput.value = "";
    return false;
  }
  if (file.size > MAX_FILE_SIZE) {
    setStatus("This file is larger than 32 MB. Choose a smaller scan or PDF.", "error");
    fileInput.value = "";
    return false;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  const isPdf = extension === "pdf";
  if (isPdf) {
    previewImg.hidden = true;
    previewImg.removeAttribute("src");
  } else {
    previewImg.hidden = false;
    previewUrl = URL.createObjectURL(file);
    previewImg.src = previewUrl;
  }
  previewName.textContent = file.name;
  previewMeta.textContent = isPdf
    ? `${formatBytes(file.size)} · original PDF preserved; page 1 analyzed`
    : `${formatBytes(file.size)} · ready to preserve`;
  previewBox.classList.add("visible");
  fileLabel.textContent = isPdf ? "✓ PDF selected" : "✓ Image selected";
  setStatus("");
  setPipelineStage("upload");
  return true;
}

function isMultipleMode() {
  return form?.querySelector('[name="uploadMode"]:checked')?.value === "multiple";
}

function updateUploadMode() {
  const multiple = isMultipleMode();
  fileInput.multiple = multiple;
  titleInput.required = false;
  titleInput.disabled = multiple;
  titleInput.placeholder = multiple ? "Titles will use each filename" : "Blank uses the uploaded file name";
  submitBtn.textContent = multiple ? "🚀 Process Selected Files" : submitLabel;
  fileLabel.textContent = multiple
    ? "📂 Drop Manuscript Files Here or Click to Browse"
    : "📂 Drop Manuscript File Here or Click to Browse";
  previewBox.classList.remove("visible");
  fileInput.value = "";
  batchResults.classList.add("hidden");
  setStatus("");
}

updateUploadMode();
modeInputs?.forEach(input => input.addEventListener("change", updateUploadMode));

// Show selected filename
fileInput?.addEventListener("change", () => {
  const files = [...fileInput.files];
  if (!files.length) return;
  if (files.some(file => !validateAndPreview(file))) return;
  if (isMultipleMode()) {
    fileLabel.textContent = `✓ ${files.length} files selected`;
    previewName.textContent = `${files.length} manuscript files selected`;
    previewMeta.textContent = "Each record title uses its filename; PDFs analyze page 1";
    const firstIsPdf = files[0].name.toLowerCase().endsWith(".pdf");
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    previewImg.hidden = firstIsPdf;
    if (firstIsPdf) previewImg.removeAttribute("src");
    else {
      previewUrl = URL.createObjectURL(files[0]);
      previewImg.src = previewUrl;
    }
    previewBox.classList.add("visible");
  }
});

["dragenter", "dragover"].forEach(eventName => fileDropZone?.addEventListener(eventName, event => {
  event.preventDefault();
  fileDropZone.classList.add("dragging");
}));
[("dragleave"), ("drop")].forEach(eventName => fileDropZone?.addEventListener(eventName, event => {
  event.preventDefault();
  fileDropZone.classList.remove("dragging");
}));
fileDropZone?.addEventListener("drop", event => {
  const files = [...event.dataTransfer.files];
  if (!files.length) return;
  const transfer = new DataTransfer();
  (isMultipleMode() ? files : files.slice(0, 1)).forEach(file => transfer.items.add(file));
  fileInput.files = transfer.files;
  fileInput.dispatchEvent(new Event("change"));
});

fetch("/api/model-status")
  .then(response => response.ok ? response.json() : Promise.reject())
  .then(status => {
    const ready = [status.restorer, status.classifier, status.ocr_corrector].filter(Boolean).length;
    const modernizer = status.modernizer;
    modelStatus.textContent = modernizer?.available
      ? `Tamil modernization ready · ${modernizer.model} via ${modernizer.provider || "hosted model"}`
      : `Tamil modernization fallback active · ${modernizer?.message || "add a Groq API key to the server .env"}`;
  })
  .catch(() => { if (modelStatus) modelStatus.textContent = "AI model status available after server startup"; });

form?.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (isMultipleMode()) {
    await processBatch([...fileInput.files]);
    return;
  }
  if (!validateAndPreview(fileInput.files[0])) return;
  submitBtn.disabled = true;
  submitBtn.textContent = "⏳ Preserving file…";
  const selectedFile = fileInput.files[0];
  const selectedPdf = selectedFile?.name.toLowerCase().endsWith(".pdf");
  setStatus(selectedPdf
    ? "Preserving the original PDF and preparing its first page for analysis…"
    : "Preparing your scan for restoration, Tamil text reading, and historical analysis…", "working");
  progressBar.classList.remove("hidden");
  results.classList.add("hidden");
  setPipelineStage("restore");

  try {
    const res  = await fetch("/api/process", { method: "POST", body: new FormData(form) });
    const data = await res.json();

    if (data.error) {
      setStatus(`${data.error} Please check the scan and try again.`, "error");
      return;
    }

    // Phase 1 — Images
    setImg("imgOriginal",     data.original,     "dlOriginal", data.original_url);
    setImg("imgPreprocessed", data.preprocessed, "dlPre",      data.preprocessed_url);
    setImg("imgRestored",     data.restored,      "dlRes",      data.restored_url);

    // Phase 7 — Classification
    const cls = data.classification;
    document.getElementById("classificationResult").innerHTML = `
      <div class="info-item"><span>✍️ Style</span><strong>${cls.style}</strong></div>
      <div class="info-item"><span>📅 Period</span><strong>${cls.period}</strong></div>
      <div class="info-item"><span>🎯 Confidence</span><strong>${(cls.confidence*100).toFixed(0)}%</strong></div>
      <div class="info-item"><span>🖊️ Ink Density</span><strong>${cls.ink_density}</strong></div>
      <div class="info-item"><span>🔲 Texture Score</span><strong>${cls.texture_score}</strong></div>
    `;

    setPipelineStage("ocr");
    // Phase 2 — Text
    document.getElementById("rawText").textContent   = data.raw_text   || "[No text]";
    document.getElementById("cleanText").value = data.clean_text || "[No text]";
    const uncertainWords = document.getElementById("uncertainWords");
    if (uncertainWords) {
      uncertainWords.replaceChildren();
      (data.correction?.uncertain_words || []).forEach(word => {
        const mark = document.createElement("mark");
        mark.className = "uncertain-word";
        mark.textContent = word;
        uncertainWords.appendChild(mark);
      });
    }
    const correctionWarning = document.getElementById("correctionWarning");
    if (correctionWarning) correctionWarning.textContent = data.correction?.warning || "";
    const retryOcr = document.getElementById("retryOcrButton");
    if (retryOcr) {
      retryOcr.dataset.manuscriptId = data.id;
      retryOcr.disabled = false;
    }
    const ocrConfidence = document.getElementById("ocrConfidence");
    if (ocrConfidence && data.correction) {
      ocrConfidence.textContent = `${Math.round((data.correction.confidence || 0) * 100)}% estimated`;
      ocrConfidence.title = data.correction.confidence_kind || "Estimated OCR quality, not a calibrated probability.";
    }
    const warning = document.getElementById("correctionWarning");
    if (warning) warning.textContent = data.correction?.warning || "";

    // Download text
    const blob = new Blob([data.clean_text], { type: "text/plain;charset=utf-8" });
    document.getElementById("dlText").href = URL.createObjectURL(blob);

    // Text stats
    const d = data.text_diff;
    document.getElementById("textStats").innerHTML = `
      <span>📊 Raw: ${d.raw_chars} chars</span>
      <span>✅ Cleaned: ${d.clean_chars} chars</span>
      <span>🗑️ Removed: ${d.removed_chars} noise chars</span>
      <span>📄 Lines: ${d.clean_lines}</span>
    `;

    setPipelineStage("context");
    // Phase 8 — Knowledge
    const kn = data.knowledge;
    const tagHtml = kn.tags.length
      ? kn.tags.map(([t, c]) => `<span class="tag cat-${c.toLowerCase()}">${t} <small>(${c})</small></span>`).join("")
      : "<span class='muted'>No specific keywords detected</span>";
    document.getElementById("knowledgeResult").innerHTML = `
      <div class="info-grid">
        <div class="info-item"><span>📂 Category</span><strong>${kn.category}</strong></div>
        <div class="info-item"><span>🏛️ Dynasty</span><strong>${kn.dynasty}</strong></div>
        <div class="info-item"><span>📅 Period</span><strong>${kn.period}</strong></div>
      </div>
      <div class="tag-cloud" style="margin-top:0.75rem">${tagHtml}</div>
    `;

    // Save confirmation
    document.getElementById("saveMsg").textContent =
      `✅ Manuscript saved to repository (ID: ${data.id})`;
    const link = document.getElementById("viewDetailLink");
    link.href = `/manuscript/${data.id}`;
    link.classList.remove("hidden");

    results.classList.remove("hidden");
    setPipelineStage("context", true);
    const successMessage = data.source_pdf_url
      ? `PDF saved with ${data.source_pdf_page_count} page(s). Page 1 was analyzed; download the original from the record.`
      : "Your manuscript is now in the archive. Explore its restored scan and extracted text below.";
    setStatus(data.processing_note
      ? `${data.processing_note} ${successMessage}`
      : successMessage, "success");
    results.scrollIntoView({ behavior: "smooth" });

  } catch (err) {
    setStatus("We couldn’t reach the processing service. Check that the app is running, then try again.", "error");
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = submitLabel;
    progressBar.classList.add("hidden");
  }
});

async function processBatch(files) {
  if (!files.length) {
    setStatus("Choose at least one manuscript file to continue.", "error");
    return;
  }
  submitBtn.disabled = true;
  submitBtn.textContent = "⏳ Processing files…";
  progressBar.classList.remove("hidden");
  results.classList.add("hidden");
  batchResults.classList.remove("hidden");
  batchResults.replaceChildren();
  const heading = document.createElement("h3");
  heading.textContent = "Batch processing results";
  const table = document.createElement("table");
  table.className = "meta-table";
  table.innerHTML = "<thead><tr><th>File</th><th>Status</th><th>Category</th><th>Record</th></tr></thead>";
  const body = document.createElement("tbody");
  table.appendChild(body);
  batchResults.append(heading, table);
  let saved = 0;

  for (const [index, file] of files.entries()) {
    setStatus(`Processing file ${index + 1} of ${files.length}: ${file.name}`, "working");
    const row = document.createElement("tr");
    const nameCell = document.createElement("td");
    nameCell.textContent = file.name;
    const statusCell = document.createElement("td");
    const categoryCell = document.createElement("td");
    const linkCell = document.createElement("td");
    row.append(nameCell, statusCell, categoryCell, linkCell);
    body.appendChild(row);

    try {
      const payload = new FormData(form);
      payload.delete("image");
      payload.delete("uploadMode");
      payload.set("image", file);
      payload.set("title", file.name.replace(/\.[^.]+$/, ""));
      const response = await fetch("/api/process", { method: "POST", body: payload });
      const data = await response.json();
      if (!response.ok || data.error) throw new Error(data.error || "Processing failed");
      statusCell.textContent = "Saved";
      categoryCell.textContent = data.knowledge?.category || "—";
      const link = document.createElement("a");
      link.href = `/manuscript/${data.id}`;
      link.textContent = "View record";
      linkCell.appendChild(link);
      saved++;
    } catch (error) {
      statusCell.textContent = "Failed";
      statusCell.style.color = "var(--red-glow)";
      categoryCell.textContent = error.message;
      linkCell.textContent = "—";
    }
  }
  setStatus(`Finished: ${saved} of ${files.length} file(s) saved to the repository.`, saved ? "success" : "error");
  submitBtn.disabled = false;
  submitBtn.textContent = submitLabel;
  progressBar.classList.add("hidden");
}

function setImg(imgId, src, dlId, altSrc) {
  const img = document.getElementById(imgId);
  const dl  = document.getElementById(dlId);
  const targetSrc = src || altSrc;
  if (img && targetSrc) {
    img.src = targetSrc;
    img.style.display = "block";
  }
  if (dl) {
    dl.href = altSrc || src || "#";
    dl.style.display = "inline-flex";
  }
}

function renderModernTamilSuggestion(suggestion, fallbackText) {
  if (window.renderTamilResult) {
    window.renderTamilResult({ ...suggestion, corrected_text: suggestion?.corrected_text || fallbackText });
    return;
  }
  const text = document.getElementById("modernTamilText");
  const confidence = document.getElementById("modernTamilConfidence");
  const changes = document.getElementById("modernTamilChanges");
  const notice = document.getElementById("modernTamilNotice");
  if (!text || !confidence || !changes) return;

  const result = suggestion || {};
  text.textContent = result.modern_text || fallbackText || "[No readable Tamil text]";
  const edits = Array.isArray(result.correction_changes) ? result.correction_changes : [];
  confidence.textContent = `${Math.round((result.confidence || 0) * 100)}% estimated · review recommended`;
  if (notice) notice.textContent = result.warning || "Partial results are preserved; review the source and uncertain words.";
  changes.replaceChildren();
  if (!edits.length) {
    const note = document.createElement("span");
    note.className = "modern-change";
    note.textContent = "The text is preserved unchanged because no verified rule matched.";
    changes.appendChild(note);
    return;
  }
  edits.forEach(edit => {
    const chip = document.createElement("span");
    chip.className = "modern-change";
    const from = document.createElement("span");
    from.textContent = edit.source;
    const arrow = document.createTextNode(" → ");
    const to = document.createElement("b");
    to.textContent = edit.target;
    chip.append(from, arrow, to);
    changes.appendChild(chip);
  });
}

function copyText() {
  const field = document.getElementById("cleanText");
  const text = field?.value || field?.textContent;
  if (!text) return;
  if (!navigator.clipboard?.writeText) {
    const btn = document.querySelector(".copy-btn");
    if (btn) btn.textContent = "Copy unavailable here";
    return;
  }
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector(".copy-btn");
    if (!btn) return;
    const originalLabel = btn.textContent;
    btn.textContent = "✓ Copied";
    btn.classList.add("copied");
    setTimeout(() => {
      btn.textContent = originalLabel;
      btn.classList.remove("copied");
    }, 2000);
  }).catch(() => {
    const btn = document.querySelector(".copy-btn");
    if (btn) btn.textContent = "Select text to copy";
  });
}

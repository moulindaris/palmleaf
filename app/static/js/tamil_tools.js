(() => {
  const correctedField = document.getElementById("cleanText");
  const convertButton = document.getElementById("convertModernTamilButton");
  const retryButton = document.getElementById("retryOcrButton");
  const loading = document.getElementById("modernTamilLoading");

  function showUncertain(words) {
    const container = document.getElementById("uncertainWords");
    if (!container) return;
    container.replaceChildren();
    if (!words?.length) return;
    const label = document.createElement("strong");
    label.textContent = "Review uncertain OCR:";
    container.appendChild(label);
    words.forEach(word => {
      const chip = document.createElement("mark");
      chip.className = "uncertain-word";
      chip.textContent = word;
      container.appendChild(chip);
    });
  }

  function renderResult(result, setCorrection = true) {
    if (!result) return;
    if (setCorrection && correctedField && typeof result.corrected_text === "string") {
      correctedField.value = result.corrected_text;
    }
    const modern = document.getElementById("modernTamilText");
    if (modern) modern.textContent = result.modern_text || result.corrected_text || "";
    const confidence = document.getElementById("modernTamilConfidence");
    if (confidence) {
      const amount = Math.round((Number(result.confidence) || 0) * 100);
      const source = result.model_used ? (result.model || "Groq AI") : "rule-based fallback";
      confidence.textContent = `${amount}% estimated · ${source}`;
      confidence.title = result.confidence_label || "Estimated text quality, not a calibrated probability.";
    }
    const ocrConfidence = document.getElementById("ocrConfidence");
    if (ocrConfidence && (result.ocr_confidence != null || result.confidence != null)) {
      ocrConfidence.textContent = `${Math.round(Number(result.ocr_confidence ?? result.confidence) * 100)}% estimated`;
      ocrConfidence.title = result.confidence_label || "Estimated text quality, not a calibrated probability.";
    }
    const warning = document.getElementById("correctionWarning");
    const note = document.getElementById("modernTamilNotice");
    if (warning) warning.textContent = result.warning || "";
    if (note) note.textContent = result.warning || (result.model_used
      ? `${result.model || "Groq AI"} generated a modern-Tamil suggestion. Compare it with the OCR and review flagged words.`
      : "Groq modernization did not complete; this is the best recoverable OCR result. Review it against the source.");
    showUncertain(result.uncertain_words || []);
    const changes = document.getElementById("modernTamilChanges");
    if (changes) {
      changes.replaceChildren();
      (result.correction_changes || result.modernization_changes || []).forEach(change => {
        const item = document.createElement("span");
        item.className = "modern-change";
        item.textContent = `${change.source} → ${change.target}`;
        changes.appendChild(item);
      });
    }
    const originalText = document.getElementById("rawText") || document.getElementById("originalOcrText");
    const download = document.getElementById("dlText");
    if (download && correctedField) {
      download.href = URL.createObjectURL(new Blob([correctedField.value], { type: "text/plain;charset=utf-8" }));
    }
    if (originalText && !originalText.textContent.trim()) originalText.textContent = "No text recovered";
  }

  async function postJson(url, payload) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || `Request failed (${response.status}).`);
    return data;
  }

  convertButton?.addEventListener("click", async () => {
    const text = correctedField?.value || "";
    convertButton.disabled = true;
    if (loading) loading.textContent = "Converting with Groq AI…";
    try {
      const result = await postJson("/api/modernize", { text });
      renderResult(result, true);
      if (loading) loading.textContent = result.model_used
        ? "Conversion complete. Review the uncertain words before relying on the reading."
        : (result.warning || "Groq modernization did not complete. Showing the best recoverable OCR result.");
    } catch (error) {
      if (loading) loading.textContent = error.message || "Conversion failed. Your edited text is still available.";
    } finally {
      convertButton.disabled = false;
    }
  });

  retryButton?.addEventListener("click", async () => {
    const id = retryButton.dataset.manuscriptId || window.palmleafManuscriptId;
    if (!id) {
      if (loading) loading.textContent = "Save this manuscript first, then retry OCR.";
      return;
    }
    retryButton.disabled = true;
    const oldLabel = retryButton.textContent;
    retryButton.textContent = "Retrying OCR…";
    if (loading) loading.textContent = "Restoring the saved scan and rerunning Tamil OCR…";
    try {
      const result = await postJson("/api/retry-ocr", { id });
      const original = document.getElementById("rawText") || document.getElementById("originalOcrText");
      if (original) original.textContent = result.raw_text || "No Tamil text detected";
      if (correctedField) correctedField.value = result.clean_text || "";
      showUncertain(result.correction?.uncertain_words || []);
      const warning = document.getElementById("correctionWarning");
      if (warning) warning.textContent = result.correction?.warning || "";
      const ocrConfidence = document.getElementById("ocrConfidence");
      if (ocrConfidence && result.correction?.confidence != null) {
        ocrConfidence.textContent = `${Math.round(Number(result.correction.confidence) * 100)}% estimated`;
      }
      if (loading) loading.textContent = "OCR retry complete. You can edit the corrected text before conversion.";
    } catch (error) {
      if (loading) loading.textContent = error.message || "OCR retry failed. Your current text was kept.";
    } finally {
      retryButton.disabled = false;
      retryButton.textContent = oldLabel;
    }
  });

  window.renderTamilResult = renderResult;
})();

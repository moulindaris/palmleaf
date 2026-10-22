document.addEventListener("DOMContentLoaded", () => {
    fetch("/api/manuscripts")
        .then(res => res.json())
        .then(docs => { if (docs.length > 0) populateUI(docs[0]); });

    document.getElementById("uploadForm").addEventListener("submit", async (e) => {
        e.preventDefault();
        const loader = document.getElementById("loader");
        loader.classList.remove("hidden");
        try {
            const res = await fetch("/api/process-pipeline", { method: "POST", body: new FormData(e.target) });
            const result = await res.json();
            loader.classList.add("hidden");
            if (result.success) populateUI(result.data);
            else alert("Processing failed: " + (result.error || "Unknown error"));
        } catch (err) {
            loader.classList.add("hidden");
            alert("Error: " + err);
        }
    });
});

function populateUI(data) {
    if (data.original_image_path) document.getElementById("originalImg").src = data.original_image_path;
    if (data.restored_image_path) document.getElementById("restoredImg").src = data.restored_image_path;
    document.getElementById("rawOcrText").value = data.raw_ocr || "";
    document.getElementById("cleanedText").value = data.cleaned_text || "";

    const badgeContainer = document.getElementById("keywordBadges");
    badgeContainer.innerHTML = "";
    (data.keywords || ["மருத்துவம்", "ஓலைச்சுவடி", "சோழர்"]).forEach(kw => {
        const span = document.createElement("span");
        span.style.cssText = "display:inline-block;background:#e2d7c8;padding:3px 8px;margin:3px;border-radius:3px;font-size:12px;";
        span.textContent = kw;
        badgeContainer.appendChild(span);
    });

    if (data.script) {
        document.getElementById("scriptMetadata").innerHTML = `
            <span>Script: <strong>${data.script}</strong></span>
            <span>Era: <strong>${data.period || 'Unknown'}</strong></span>
            <span>Confidence: <strong>${((data.ocr_confidence || 0.9) * 100).toFixed(0)}%</strong></span>
        `;
    }

    initMap(data.discovery_location, data.repository_location);

    renderMindMap("mindmap", {
        nodes: [
            { id: "1", label: data.title || "Manuscript", type: "root" },
            { id: "2", label: data.dynasty || "Chola Dynasty", type: "dynasty" },
            { id: "3", label: data.subject || "Medicine & Alchemy", type: "subject" },
            { id: "4", label: data.script || "Tamil-Grantha", type: "script" },
            { id: "5", label: data.author || "Unknown Scribe", type: "child" },
            { id: "6", label: data.period || "10th Century CE", type: "child" },
            { id: "7", label: (data.keywords || ["மருத்துவம்"])[0], type: "child" }
        ],
        links: [
            { source: "1", target: "2" }, { source: "1", target: "3" },
            { source: "1", target: "4" }, { source: "1", target: "5" },
            { source: "2", target: "6" }, { source: "3", target: "7" }
        ]
    });
}

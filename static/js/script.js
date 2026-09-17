/**
 * DermoAI — Clinical Dermoscopic Diagnostic Assistant
 * Interactive Client Script
 */

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropZone = document.getElementById("dropZone");
    const imageInput = document.getElementById("imageInput");
    const browseBtn = document.getElementById("browseBtn");
    const dropzoneIdle = document.getElementById("dropzoneIdle");
    const dropzonePreview = document.getElementById("dropzonePreview");
    const previewImg = document.getElementById("previewImg");
    const previewFilename = document.getElementById("previewFilename");
    const changeImageBtn = document.getElementById("changeImageBtn");

    const predictBtn = document.getElementById("predictBtn");
    const resetBtn = document.getElementById("resetBtn");
    const sampleChips = document.querySelectorAll(".sample-chip");

    const resultsEmpty = document.getElementById("resultsEmpty");
    const resultsLoading = document.getElementById("resultsLoading");
    const resultsActive = document.getElementById("resultsActive");

    const verdictCard = document.getElementById("verdictCard");
    const verdictPill = document.getElementById("verdictPill");
    const verdictHeading = document.getElementById("verdictHeading");
    const verdictSummary = document.getElementById("verdictSummary");

    const metricConfidence = document.getElementById("metricConfidence");
    const metricConfidenceSub = document.getElementById("metricConfidenceSub");
    const metricRiskTier = document.getElementById("metricRiskTier");
    const probPercent = document.getElementById("probPercent");
    const probBarFill = document.getElementById("probBarFill");
    const recommendationList = document.getElementById("recommendationList");
    const printReportBtn = document.getElementById("printReportBtn");
    const printableReport = document.getElementById("printableReport");

    let currentFile = null;
    let currentAnalysisData = null;

    // --------------------------------------------------------------------------
    // File Input & Drag-and-Drop Handling
    // --------------------------------------------------------------------------

    browseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        imageInput.click();
    });

    dropZone.addEventListener("click", () => {
        if (!currentFile) {
            imageInput.click();
        }
    });

    changeImageBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        imageInput.click();
    });

    imageInput.addEventListener("change", () => {
        if (imageInput.files && imageInput.files[0]) {
            handleSelectedFile(imageInput.files[0]);
        }
    });

    // Drag-over styling
    ["dragenter", "dragover"].forEach(evtName => {
        dropZone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evtName => {
        dropZone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("dragover");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const dt = e.dataTransfer;
        if (dt && dt.files && dt.files[0]) {
            handleSelectedFile(dt.files[0]);
        }
    });

    // Clipboard paste support
    window.addEventListener("paste", (e) => {
        const items = (e.clipboardData || e.originalEvent.clipboardData).items;
        for (let item of items) {
            if (item.type.indexOf("image") === 0) {
                const blob = item.getAsFile();
                handleSelectedFile(blob, "pasted-dermoscopy.png");
                break;
            }
        }
    });

    function handleSelectedFile(file, customName) {
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file (JPG, PNG, or WEBP).");
            return;
        }

        currentFile = file;
        const filename = customName || file.name || "lesion-image.jpg";
        previewFilename.innerText = filename;

        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            dropzoneIdle.classList.add("hidden");
            dropzonePreview.classList.remove("hidden");
            predictBtn.removeAttribute("disabled");
        };
        reader.readAsDataURL(file);
    }

    // --------------------------------------------------------------------------
    // 1-Click Interactive Sample Gallery
    // --------------------------------------------------------------------------

    sampleChips.forEach(chip => {
        chip.addEventListener("click", async () => {
            const sampleName = chip.getAttribute("data-sample");
            const sampleUrl = `/static/samples/${sampleName}`;

            try {
                // Fetch image from static folder as Blob
                const res = await fetch(sampleUrl);
                if (!res.ok) throw new Error("Could not load sample image");
                const blob = await res.blob();
                const file = new File([blob], sampleName, { type: blob.type || "image/jpeg" });
                
                handleSelectedFile(file, sampleName);

                // Auto-run analysis for instant feedback
                setTimeout(() => {
                    runPrediction();
                }, 150);

            } catch (err) {
                console.error("Error loading sample:", err);
                alert("Could not load sample: " + sampleName);
            }
        });
    });

    // --------------------------------------------------------------------------
    // Reset Action
    // --------------------------------------------------------------------------

    resetBtn.addEventListener("click", () => {
        currentFile = null;
        currentAnalysisData = null;
        imageInput.value = "";
        previewImg.src = "";
        dropzoneIdle.classList.remove("hidden");
        dropzonePreview.classList.add("hidden");
        predictBtn.setAttribute("disabled", "true");

        resultsLoading.classList.add("hidden");
        resultsActive.classList.add("hidden");
        resultsEmpty.classList.remove("hidden");
    });

    // --------------------------------------------------------------------------
    // Prediction Action
    // --------------------------------------------------------------------------

    predictBtn.addEventListener("click", () => {
        runPrediction();
    });

    async function runPrediction() {
        if (!currentFile) {
            alert("Please upload or select a dermoscopic image first.");
            return;
        }

        // Switch to Loading View
        resultsEmpty.classList.add("hidden");
        resultsActive.classList.add("hidden");
        resultsLoading.classList.remove("hidden");
        predictBtn.setAttribute("disabled", "true");

        const formData = new FormData();
        formData.append("file", currentFile);

        try {
            const response = await fetch("/api/predict", {
                method: "POST",
                body: formData
            });

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server responded with status ${response.status}`);
            }

            const data = await response.json();

            if (data.error) {
                throw new Error(data.error);
            }

            currentAnalysisData = data;
            renderResults(data);

        } catch (err) {
            console.error("Prediction error:", err);
            alert("Analysis failed: " + err.message);
            resultsLoading.classList.add("hidden");
            resultsEmpty.classList.remove("hidden");
        } finally {
            predictBtn.removeAttribute("disabled");
        }
    }

    // --------------------------------------------------------------------------
    // Render Results & Metrics
    // --------------------------------------------------------------------------

    function renderResults(data) {
        resultsLoading.classList.add("hidden");
        resultsActive.classList.remove("hidden");

        const isCancerous = (data.prediction === "Cancerous");
        const confPercent = (data.confidence * 100).toFixed(1);
        const maligProb = (data.malignancy_probability * 100).toFixed(1);

        // Update Verdict Card
        verdictCard.className = "verdict-card " + (isCancerous ? "verdict-malignant" : "verdict-benign");
        verdictPill.innerText = isCancerous ? "Potential Malignancy" : "Benign Lesion";
        verdictHeading.innerText = isCancerous ? "Cancerous / High Concern" : "Non-Cancerous / Benign";
        verdictSummary.innerText = data.summary || (isCancerous
            ? "Automated feature patterns align with malignant melanoma or high-grade cutaneous neoplasm. Specialist evaluation advised."
            : "Features are consistent with benign melanocytic nevi, seborrheic keratosis, or non-malignant skin patterns.");

        // Metric boxes
        metricConfidence.innerText = `${confPercent}%`;
        metricConfidenceSub.innerText = isCancerous ? "Confidence: Cancerous" : "Confidence: Benign";

        metricRiskTier.innerText = data.risk_tier || (isCancerous ? "High" : "Low");
        metricRiskTier.className = "metric-val risk-val " + 
            (data.risk_tier === "High Risk" ? "risk-high" : (data.risk_tier === "Moderate Risk" ? "risk-moderate" : "risk-low"));

        // Probability meter
        probPercent.innerText = `${maligProb}%`;
        probBarFill.style.width = `${Math.min(100, Math.max(0, maligProb))}%`;

        // Recommendations List
        recommendationList.innerHTML = "";
        const recs = data.clinical_recommendations || getDefaultRecommendations(isCancerous);
        recs.forEach(rec => {
            const li = document.createElement("li");
            li.innerText = rec;
            recommendationList.appendChild(li);
        });
    }

    function getDefaultRecommendations(isCancerous) {
        if (isCancerous) {
            return [
                "Schedule an in-person clinical dermatoscopy with a board-certified dermatologist.",
                "Perform full-body cutaneous examination to check for secondary atypical lesions.",
                "Avoid unnecessary trauma or manipulation of the pigmented lesion.",
                "Consider excisional biopsy with appropriate histological margins if clinically indicated."
            ];
        } else {
            return [
                "No immediate markers of malignancy detected in dermoscopic pattern.",
                "Continue routine self-examination following the ABCDE criteria quarterly.",
                "Protect skin with broad-spectrum SPF 30+ sunscreen and UV-protective clothing.",
                "Consult a physician promptly if the lesion begins bleeding, itching, or rapidly evolving."
            ];
        }
    }

    // --------------------------------------------------------------------------
    // Printable Clinical Summary Report
    // --------------------------------------------------------------------------

    printReportBtn.addEventListener("click", () => {
        if (!currentAnalysisData) return;

        const dateStr = new Date().toLocaleString();
        const imgSrc = previewImg.src;
        const d = currentAnalysisData;
        const isCancer = (d.prediction === "Cancerous");

        printableReport.innerHTML = `
            <div style="font-family: Arial, sans-serif; max-width: 750px; margin: 0 auto; border: 2px solid #0f172a; padding: 25px; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #0284c7; padding-bottom: 12px; margin-bottom: 18px;">
                    <div>
                        <h2 style="margin: 0; color: #0284c7; font-size: 22px;">DermoAI Clinical Assessment Report</h2>
                        <p style="margin: 3px 0 0; font-size: 12px; color: #64748b;">Deep Learning Dermoscopy & Cutaneous Lesion Triage</p>
                    </div>
                    <div style="text-align: right; font-size: 12px; color: #475569;">
                        <div><strong>Report Date:</strong> ${dateStr}</div>
                        <div><strong>Model:</strong> MobileNetV2 Transfer Learning</div>
                    </div>
                </div>

                <div style="display: flex; gap: 24px; margin-bottom: 20px;">
                    <div style="flex: 0 0 200px; text-align: center;">
                        <img src="${imgSrc}" style="width: 190px; height: 190px; object-fit: cover; border: 1px solid #cbd5e1; border-radius: 6px;">
                        <div style="font-size: 11px; color: #64748b; margin-top: 5px;">Lesion Dermoscopic Capture</div>
                    </div>
                    <div style="flex: 1;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 13px;">
                            <tr style="border-bottom: 1px solid #e2e8f0;">
                                <td style="padding: 6px 0; color: #64748b;"><strong>Classification:</strong></td>
                                <td style="padding: 6px 0; font-weight: bold; color: ${isCancer ? '#dc2626' : '#059669'}; font-size: 16px;">
                                    ${d.prediction} (${isCancer ? 'Malignant Suspicion' : 'Benign Pattern'})
                                </td>
                            </tr>
                            <tr style="border-bottom: 1px solid #e2e8f0;">
                                <td style="padding: 6px 0; color: #64748b;"><strong>Model Confidence:</strong></td>
                                <td style="padding: 6px 0;"><strong>${(d.confidence * 100).toFixed(2)}%</strong></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #e2e8f0;">
                                <td style="padding: 6px 0; color: #64748b;"><strong>Malignancy Risk Index:</strong></td>
                                <td style="padding: 6px 0;"><strong>${(d.malignancy_probability * 100).toFixed(2)}%</strong></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #e2e8f0;">
                                <td style="padding: 6px 0; color: #64748b;"><strong>Triage Risk Tier:</strong></td>
                                <td style="padding: 6px 0;"><strong>${d.risk_tier || (isCancer ? 'High Risk' : 'Low Risk')}</strong></td>
                            </tr>
                        </table>
                    </div>
                </div>

                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 6px; padding: 14px; margin-bottom: 20px;">
                    <h4 style="margin: 0 0 8px; font-size: 13px; color: #0284c7; text-transform: uppercase;">Clinical Recommendations</h4>
                    <ul style="margin: 0; padding-left: 20px; font-size: 12px; line-height: 1.5;">
                        ${(d.clinical_recommendations || getDefaultRecommendations(isCancer)).map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>

                <div style="border-top: 1px dashed #cbd5e1; padding-top: 14px; margin-top: 24px; font-size: 10px; color: #94a3b8; line-height: 1.4;">
                    <strong>LEGAL & CLINICAL DISCLAIMER:</strong> This automated analysis is generated via an artificial intelligence neural network for investigational and clinical decision-support only. It does NOT constitute a medical diagnosis, pathology report, or definitive staging. Clinical correlation and histopathological evaluation by a certified dermatopathologist remain the gold standard.
                </div>

                <div style="margin-top: 30px; display: flex; justify-content: space-between; font-size: 12px; color: #475569;">
                    <div>Examining Clinician Signature: _______________________</div>
                    <div>Date: ________________</div>
                </div>
            </div>
        `;

        window.print();
    });
});

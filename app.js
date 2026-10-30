/**
 * ORIGIN X — Interactive Client Controller
 * Adheres to Apple Fluid Motion and Direct Manipulation design patterns.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const dropzonePrompt = document.getElementById("dropzone-prompt");
  const previewContainer = document.getElementById("preview-container");
  const previewImage = document.getElementById("preview-image");
  const btnChangeImage = document.getElementById("btn-change-image");
  
  const fileDetails = document.getElementById("file-details");
  const fileMetaName = document.getElementById("file-meta-name");
  const fileMetaDims = document.getElementById("file-meta-dims");
  const fileMetaSize = document.getElementById("file-meta-size");

  const btnAnalyze = document.getElementById("btn-analyze");
  const btnReset = document.getElementById("btn-reset");

  const resultStatusTag = document.getElementById("result-status-tag");
  const resultsEmpty = document.getElementById("results-empty");
  const resultsLoading = document.getElementById("results-loading");
  const resultsActive = document.getElementById("results-active");
  const resultsError = document.getElementById("results-error");
  const errorTitle = document.getElementById("error-title");
  const errorDetail = document.getElementById("error-detail");

  // Result Elements
  const verdictBadge = document.getElementById("verdict-badge");
  const confidenceTier = document.getElementById("confidence-tier");
  const verdictRationale = document.getElementById("verdict-rationale");
  const probPercentage = document.getElementById("prob-percentage");
  const meterFill = document.getElementById("meter-fill");
  const ciMarker = document.getElementById("confidence-interval-marker");
  const ciText = document.getElementById("ci-text");

  const signalSpectralVal = document.getElementById("signal-spectral-val");
  const signalSpectralDesc = document.getElementById("signal-spectral-desc");
  const signalNoiseVal = document.getElementById("signal-noise-val");
  const signalNoiseDesc = document.getElementById("signal-noise-desc");
  const signalColorVal = document.getElementById("signal-color-val");
  const signalColorDesc = document.getElementById("signal-color-desc");

  let currentFile = null;
  let currentPreviewUrl = null;

  // --------------------------------------------------------------------------
  // Direct Manipulation Drag & Drop Handlers
  // --------------------------------------------------------------------------
  ["dragenter", "dragover"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    });
  });

  dropzone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    if (dt && dt.files && dt.files.length > 0) {
      handleFileSelected(dt.files[0]);
    }
  });

  dropzone.addEventListener("click", (e) => {
    if (e.target !== btnChangeImage) {
      fileInput.click();
    }
  });

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.click();
    }
  });

  fileInput.addEventListener("change", () => {
    if (fileInput.files && fileInput.files.length > 0) {
      handleFileSelected(fileInput.files[0]);
    }
  });

  btnChangeImage.addEventListener("click", (e) => {
    e.stopPropagation();
    fileInput.click();
  });

  // --------------------------------------------------------------------------
  // File Validation and In-Memory Preview
  // --------------------------------------------------------------------------
  function handleFileSelected(file) {
    // Validate type
    const validTypes = ["image/jpeg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      showError("Unsupported Format", `Please upload a JPEG, PNG, or WebP file. (Detected: ${file.type || 'unknown'})`);
      return;
    }

    // Validate size (15 MB)
    if (file.size > 15 * 1024 * 1024) {
      showError("File Too Large", `File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds the 15 MB limit.`);
      return;
    }

    currentFile = file;

    // Clean up previous blob URL
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
    }
    currentPreviewUrl = URL.createObjectURL(file);

    // Update Preview
    previewImage.src = currentPreviewUrl;
    dropzonePrompt.classList.add("hidden");
    previewContainer.classList.remove("hidden");

    // Read Dimensions
    const img = new Image();
    img.onload = () => {
      fileMetaDims.textContent = `${img.naturalWidth} × ${img.naturalHeight} px`;
    };
    img.src = currentPreviewUrl;

    // File Metadata
    fileMetaName.textContent = file.name;
    fileMetaSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    fileDetails.classList.remove("hidden");

    // Enable Buttons
    btnAnalyze.disabled = false;
    btnReset.disabled = false;

    // Reset results panel
    resetResultsView();
  }

  // --------------------------------------------------------------------------
  // Execute Origin Analysis
  // --------------------------------------------------------------------------
  btnAnalyze.addEventListener("click", async () => {
    if (!currentFile) return;

    showLoadingState();

    const formData = new FormData();
    formData.append("file", currentFile);

    try {
      const response = await fetch("/api/v1/analyze", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errJson = await response.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server returned error status ${response.status}`);
      }

      const result = await response.json();
      displayResults(result);
    } catch (err) {
      showError("Analysis Incomplete", err.message);
    }
  });

  // --------------------------------------------------------------------------
  // Render Probabilistic Results
  // --------------------------------------------------------------------------
  function displayResults(data) {
    resultsLoading.classList.add("hidden");
    resultsEmpty.classList.add("hidden");
    resultsError.classList.add("hidden");
    resultsActive.classList.remove("hidden");

    const pred = data.prediction;
    const diag = data.diagnostics;

    // Verdict Badge & Rationale
    verdictBadge.className = "verdict-badge";
    if (pred.is_inconclusive) {
      verdictBadge.textContent = "INCONCLUSIVE";
      verdictBadge.classList.add("inconclusive");
      resultStatusTag.textContent = "Indeterminate";
      resultStatusTag.className = "status-tag idle";
    } else if (pred.verdict === "AI_GENERATED") {
      verdictBadge.textContent = "AI GENERATED";
      verdictBadge.classList.add("ai");
      resultStatusTag.textContent = "Estimated AI";
      resultStatusTag.className = "status-tag active";
    } else {
      verdictBadge.textContent = "HUMAN CREATED";
      verdictBadge.classList.add("human");
      resultStatusTag.textContent = "Estimated Human";
      resultStatusTag.className = "status-tag active";
    }

    confidenceTier.textContent = `Confidence: ${pred.confidence_tier}`;
    verdictRationale.textContent = pred.rationale;

    // Calibrated Gauge
    const probAiPct = (pred.probability_ai * 100).toFixed(1);
    probPercentage.textContent = `${probAiPct}%`;
    meterFill.style.width = `${Math.min(100, Math.max(0, pred.probability_ai * 100))}%`;

    // Confidence Interval Marker
    const ciLow = (pred.confidence_interval[0] * 100).toFixed(1);
    const ciHigh = (pred.confidence_interval[1] * 100).toFixed(1);
    ciText.textContent = `Estimated 95% Confidence Interval: [${ciLow}% – ${ciHigh}%]`;
    ciMarker.style.left = `${ciLow}%`;
    ciMarker.style.width = `${Math.max(2, ciHigh - ciLow)}%`;

    // Diagnostic Signals
    signalSpectralVal.textContent = diag.spectral_profile.high_frequency_ratio.toFixed(3);
    signalSpectralDesc.textContent = diag.spectral_profile.description;

    signalNoiseVal.textContent = diag.noise_consistency.residual_kurtosis.toFixed(1);
    signalNoiseDesc.textContent = diag.noise_consistency.description;

    signalColorVal.textContent = diag.chromatic_profile.high_saturation_ratio.toFixed(3);
    signalColorDesc.textContent = diag.chromatic_profile.description;
  }

  // --------------------------------------------------------------------------
  // State Transitions & Helpers
  // --------------------------------------------------------------------------
  function showLoadingState() {
    resultsEmpty.classList.add("hidden");
    resultsActive.classList.add("hidden");
    resultsError.classList.add("hidden");
    resultsLoading.classList.remove("hidden");
    btnAnalyze.disabled = true;
    resultStatusTag.textContent = "Analyzing...";
  }

  function showError(title, detail) {
    resultsLoading.classList.add("hidden");
    resultsActive.classList.add("hidden");
    resultsEmpty.classList.add("hidden");
    resultsError.classList.remove("hidden");
    errorTitle.textContent = title;
    errorDetail.textContent = detail;
    btnAnalyze.disabled = !currentFile;
    resultStatusTag.textContent = "Error";
    resultStatusTag.className = "status-tag idle";
  }

  function resetResultsView() {
    resultsActive.classList.add("hidden");
    resultsLoading.classList.add("hidden");
    resultsError.classList.add("hidden");
    resultsEmpty.classList.remove("hidden");
    resultStatusTag.textContent = "Ready to Analyze";
    resultStatusTag.className = "status-tag idle";
  }

  btnReset.addEventListener("click", () => {
    currentFile = null;
    if (currentPreviewUrl) {
      URL.revokeObjectURL(currentPreviewUrl);
      currentPreviewUrl = null;
    }
    fileInput.value = "";
    previewImage.src = "";
    previewContainer.classList.add("hidden");
    dropzonePrompt.classList.remove("hidden");
    fileDetails.classList.add("hidden");
    btnAnalyze.disabled = true;
    btnReset.disabled = true;
    resetResultsView();
  });
});

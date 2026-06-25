// Global state variables for charts and data
let severityChart = null;
let sentencesChart = null;
let typologyMap = null;
let globalTypologyList = [];
let currentPredictions = null;
let currentAuditCountry = null;
let lastQueryResult = null;

// Configure Chart.js global defaults for dark mode legibility
if (window.Chart) {
    Chart.defaults.color = '#f1f5f9';
    Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.08)';
    if (Chart.defaults.plugins && Chart.defaults.plugins.tooltip) {
        Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.96)';
        Chart.defaults.plugins.tooltip.titleColor = '#ffffff';
        Chart.defaults.plugins.tooltip.bodyColor = '#f1f5f9';
        Chart.defaults.plugins.tooltip.borderColor = 'rgba(255, 255, 255, 0.12)';
        Chart.defaults.plugins.tooltip.borderWidth = 1;
    }
}

// DOM Elements
const auditForm = document.getElementById("audit-form");
const submitBtn = document.getElementById("submit-btn");
const btnText = submitBtn.querySelector(".btn-text");
const spinner = submitBtn.querySelector(".spinner");
const factPatternInput = document.getElementById("fact-pattern");
const priorsInput = document.getElementById("priors");
const priorsVal = document.getElementById("priors-val");
const suggestionTags = document.querySelectorAll(".suggestion-tag");
const resultsPanel = document.getElementById("results-panel");
const panelPlaceholder = resultsPanel.querySelector(".panel-placeholder");
const panelContent = resultsPanel.querySelector(".panel-content");
const actualSentenceInput = document.getElementById("actual-sentence");
const selectedAuditCountry = document.getElementById("selected-audit-country");

// Colors for dynamic clustering
const CLUSTER_COLORS = [
    "#06b6d4", // Cyan
    "#10b981", // Emerald
    "#f43f5e", // Rose
    "#a855f7", // Purple
    "#fb923c", // Orange
    "#facc15"  // Yellow
];

// Initialize on page load
document.addEventListener("DOMContentLoaded", () => {
    // 0. Setup theme toggle
    initTheme();

    // 1. Setup range slider display
    priorsInput.addEventListener("input", (e) => {
        priorsVal.textContent = e.target.value;
    });

    // 2. Setup suggestion tags
    suggestionTags.forEach(tag => {
        tag.addEventListener("click", () => {
            factPatternInput.value = tag.getAttribute("data-text");
        });
    });

    // 3. Load typology map data
    loadTypologyMap();

    // 4. Setup form submit
    auditForm.addEventListener("submit", handleAuditSubmit);

    // 5. Setup selected country change listener
    selectedAuditCountry.addEventListener("change", () => {
        currentAuditCountry = selectedAuditCountry.value;
        if (lastQueryResult) {
            const actualVal = actualSentenceInput.value.trim();
            const currentPayload = {
                priors: parseInt(priorsInput.value),
                plea_guilty: document.getElementById("plea-guilty").checked,
                mitigating_circumstances: document.getElementById("mitigating").checked,
                juvenile: document.getElementById("juvenile").checked,
                court_region: document.getElementById("court-region").checked
            };
            updateDriftGauges(lastQueryResult, actualVal !== "" ? parseFloat(actualVal) : null);
            updateMLInsights(lastQueryResult, currentPayload);

            // Update country details side panel
            const countryMeta = globalTypologyList.find(d => d.country === currentAuditCountry);
            if (countryMeta) {
                showCountryDetails(countryMeta);
            }
        }
    });

    // 6. Setup actual sentence real-time feedback listener
    actualSentenceInput.addEventListener("input", () => {
        if (lastQueryResult) {
            const actualVal = actualSentenceInput.value.trim();
            updateDriftGauges(lastQueryResult, actualVal !== "" ? parseFloat(actualVal) : null);
        }
    });
});

// Load the 2D Typology coordinates from the backend
async function loadTypologyMap() {
    try {
        const response = await fetch("/api/typology");
        if (!response.ok) throw new Error("Failed to load typology map");
        
        globalTypologyList = await response.json();
        generateDynamicLegend(globalTypologyList);
        renderTypologyMap(globalTypologyList);
    } catch (error) {
        console.error("Error loading typology:", error);
        document.getElementById("typology-details-panel").innerHTML = `
            <div class="empty-detail-state" style="color: var(--color-danger)">
                <p>Failed to connect to the backend ML server. Please check that the server is running and models are trained.</p>
            </div>
        `;
    }
}

// Dynamically generate the typology legend based on optimal K
function generateDynamicLegend(data) {
    const legendContainer = document.getElementById("typology-legend-container");
    legendContainer.innerHTML = "";
    
    // Find unique clusters
    const uniqueClusters = {};
    data.forEach(item => {
        uniqueClusters[item.cluster_id] = item.cluster_name;
    });
    
    // Sort cluster keys
    const sortedIds = Object.keys(uniqueClusters).map(Number).sort((a, b) => a - b);
    
    sortedIds.forEach(id => {
        const name = uniqueClusters[id].split(" (")[0]; // shorter name for legend
        const color = CLUSTER_COLORS[id % CLUSTER_COLORS.length];
        
        const itemSpan = document.createElement("span");
        itemSpan.className = "legend-item";
        itemSpan.innerHTML = `<span class="legend-dot" style="background-color: ${color}"></span> ${name}`;
        legendContainer.appendChild(itemSpan);
    });
}

// Render the 2D Scatter Plot (PCA Typology Map) with dynamic datasets
function renderTypologyMap(data) {
    const ctx = document.getElementById("typology-map").getContext("2d");
    
    // Extract unique cluster IDs present in data
    const uniqueClusterIds = [...new Set(data.map(d => d.cluster_id))].sort((a, b) => a - b);
    
    // Map each cluster to a dataset
    const datasets = uniqueClusterIds.map(clusterId => {
        const clusterData = data.filter(d => d.cluster_id === clusterId);
        const color = CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
        const label = clusterData[0].cluster_name.split(" (")[0];
        
        return {
            label: label,
            data: clusterData.map(d => ({ x: d.x, y: d.y, r: 8, country: d.country, meta: d })),
            backgroundColor: color,
            borderColor: "rgba(255,255,255,0.3)",
            borderWidth: 1,
            hoverRadius: 11
        };
    });

    if (typologyMap) typologyMap.destroy();

    typologyMap = new Chart(ctx, {
        type: 'bubble',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const raw = context.raw;
                            return `${raw.country} (${raw.meta.legal_family})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: document.documentElement.getAttribute("data-theme") === "light" ? "rgba(0, 0, 0, 0.05)" : "rgba(255, 255, 255, 0.03)" },
                    ticks: { color: document.documentElement.getAttribute("data-theme") === "light" ? "#64748b" : "#cbd5e1", font: { size: 9 } },
                    title: { display: true, text: "PCA Component 1 (Punitive Scaling / Slope)", color: document.documentElement.getAttribute("data-theme") === "light" ? "#64748b" : "#cbd5e1" }
                },
                y: {
                    grid: { color: document.documentElement.getAttribute("data-theme") === "light" ? "rgba(0, 0, 0, 0.05)" : "rgba(255, 255, 255, 0.03)" },
                    ticks: { color: document.documentElement.getAttribute("data-theme") === "light" ? "#64748b" : "#cbd5e1", font: { size: 9 } },
                    title: { display: true, text: "PCA Component 2 (Mitigation Variance)", color: document.documentElement.getAttribute("data-theme") === "light" ? "#64748b" : "#cbd5e1" }
                }
            },
            onClick: (event, activeElements) => {
                if (activeElements.length > 0) {
                    const activeElement = activeElements[0];
                    const datasetIndex = activeElement.datasetIndex;
                    const index = activeElement.index;
                    const clickedPoint = typologyMap.data.datasets[datasetIndex].data[index];
                    showCountryDetails(clickedPoint.meta);

                    // Sync dropdown and global audit state
                    selectedAuditCountry.value = clickedPoint.meta.country;
                    currentAuditCountry = clickedPoint.meta.country;

                    // Update gauges and ML trace instantly if query has already run
                    if (lastQueryResult) {
                        const actualVal = actualSentenceInput.value.trim();
                        const currentPayload = {
                            priors: parseInt(priorsInput.value),
                            plea_guilty: document.getElementById("plea-guilty").checked,
                            mitigating_circumstances: document.getElementById("mitigating").checked,
                            juvenile: document.getElementById("juvenile").checked,
                            court_region: document.getElementById("court-region").checked
                        };
                        updateDriftGauges(lastQueryResult, actualVal !== "" ? parseFloat(actualVal) : null);
                        updateMLInsights(lastQueryResult, currentPayload);
                    }
                }
            }
        }
    });
}

// Show detailed information for a selected country in the side-panel
function showCountryDetails(countryMeta) {
    const detailsPanel = document.getElementById("typology-details-panel");
    const emptyState = detailsPanel.querySelector(".empty-detail-state");
    const content = detailsPanel.querySelector(".detail-content");

    emptyState.classList.add("hidden");
    content.classList.remove("hidden");

    // Fill textual fields
    document.getElementById("detail-country-name").textContent = countryMeta.country;
    document.getElementById("detail-legal-family").textContent = countryMeta.legal_family;
    document.getElementById("detail-region").textContent = countryMeta.region;
    
    // Cluster info
    const clusterPill = document.getElementById("detail-cluster-pill");
    clusterPill.textContent = countryMeta.cluster_name;
    clusterPill.className = "cluster-pill"; // reset
    
    const id = countryMeta.cluster_id;
    // Map cluster classes dynamically
    if (id === 0) clusterPill.classList.add("cluster-pill-rehab");
    else if (id === globalTypologyList.reduce((max, d) => Math.max(max, d.cluster_id), 0)) clusterPill.classList.add("cluster-pill-punitive");
    else clusterPill.classList.add("cluster-pill-prop");
    
    document.getElementById("detail-cluster-desc").textContent = countryMeta.cluster_description;
    
    // Country description
    const desc = getCountryDescription(countryMeta.country);
    document.getElementById("detail-country-desc").textContent = desc;

    // Show predictions if they exist
    const predBox = document.getElementById("detail-current-pred-box");
    if (currentPredictions && currentPredictions[countryMeta.country]) {
        predBox.classList.remove("hidden");
        const pred = currentPredictions[countryMeta.country];
        document.getElementById("detail-pred-sentence").textContent = `${pred.sentence} months`;
        
        const driftVal = pred.cross_drift;
        const sign = driftVal >= 0 ? "+" : "";
        document.getElementById("detail-pred-drift").textContent = `${sign}${driftVal}%`;
        document.getElementById("detail-pred-drift").className = `pred-value ${driftVal >= 0 ? "color-violence" : "color-safety"}`;
    } else {
        predBox.classList.add("hidden");
    }
}

// Static fallback mapping for country legal details descriptions
function getCountryDescription(country) {
    const details = {
        "United States": "Steep federal sentencing guidelines, mandatory minimums, strong focus on individual criminal history, and high sentencing severity. Metro courts are slightly more lenient (-15%).",
        "United Kingdom": "Highly structured guidelines established by the Sentencing Council, balancing offense seriousness against mitigation. Metro courts are slightly more lenient (-8%).",
        "Canada": "Statutory principles require sentences proportional to gravity, focusing heavily on restorative mitigation. Metro courts are slightly more lenient (-6%).",
        "Australia": "Guided by court judgments and regional codes, prioritizing deterrence, rehabilitation, and community protection. Metro courts are slightly more lenient (-7%).",
        "Singapore": "Strict penal codes aiming for high deterrence, imposing severe penalties for public safety risks, drug trafficking, and violent offenses. City-state layout results in zero regional bias.",
        "India": "Broad judicial discretion. Sentences in metropolitan hubs are slightly more severe (+20%) due to city-specific crime control laws.",
        "Germany": "Structured around the principle of guilt; focuses heavily on rehabilitation and social reintegration, with short/medium prison terms.",
        "France": "Involves statutory minimums for multiple repeat offenders (recidivists), combined with judicial options for alternative rehabilitation measures.",
        "Italy": "Codified civil system with formulaic discounts, such as a mandatory 33% sentence reduction for fast-track plea bargains (patteggiamento).",
        "Spain": "Civil penal code with strict classifications of crimes, with a moderate focus on social rehabilitation and restorative sentencing options.",
        "Switzerland": "A highly predictable system featuring low overall incarceration rates and prioritizing fines, community service, and rehabilitative therapy.",
        "Brazil": "Broad sentencing ranges within a unified penal code. Metropolitan hubs are slightly more severe (+10%). Constitutitional cap is 30 years.",
        "Norway": "Focuses almost exclusively on restorative rehabilitation, offering flat response curves, short prison terms, and a maximum standard cap of 21 years.",
        "Sweden": "Centers on 'penal value' calculations, keeping custodial sentences low and integrating substantial probation and community options.",
        "Denmark": "Nordic model focused on social rehabilitation, offering flat scaling, high plea/mitigation impact, and a strong preference for non-custodial options.",
        "Japan": "A focus on 'precision justice' and confession-based rehabilitation; showing deep remorse and settling with victims grants large discounts.",
        "South Korea": "A system incorporating German civil structures, where depositing restitution funds (gong-tak) and remorse lead to mitigation.",
        "Saudi Arabia": "Sentences scale sharply based on Sharia principles (Qisas/Tazir), heavily penalizing physical violence and public moral violations.",
        "Iran": "Codified Islamic penal codes emphasizing retribution, featuring high severity weights and conservative discounts for legal mitigation.",
        "South Africa": "A hybrid framework merging common law and civil principles; uses statutory minimum terms for heavy crimes like aggravated robbery."
    };
    return details[country] || "Unified legal structure with distinct national guidelines.";
}

// Handle Form Submission and query Flask API
async function handleAuditSubmit(e) {
    e.preventDefault();

    const factPattern = factPatternInput.value.trim();
    if (factPattern.length < 10) return;

    // Show loading state
    submitBtn.disabled = true;
    btnText.textContent = "Extracting Features & Auditing...";
    spinner.classList.remove("hidden");

    // Gather inputs
    const payload = {
        fact_pattern: factPattern,
        priors: parseInt(priorsInput.value),
        plea_guilty: document.getElementById("plea-guilty").checked,
        mitigating_circumstances: document.getElementById("mitigating").checked,
        juvenile: document.getElementById("juvenile").checked,
        court_region: document.getElementById("court-region").checked // Metropolitan
    };

    const actualVal = actualSentenceInput.value.trim();
    const auditCountry = selectedAuditCountry.value;

    if (auditCountry !== "") {
        currentAuditCountry = auditCountry;
    } else {
        currentAuditCountry = null;
    }

    if (actualVal !== "") {
        payload.actual_sentence = parseFloat(actualVal);
    } else {
        payload.actual_sentence = null;
    }

    try {
        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const err = await response.json();
            throw new Error(err.error || "Analysis failed");
        }

        const result = await response.json();
        
        // Save to global state
        currentPredictions = result.predictions;
        lastQueryResult = result;

        // Display results container
        panelPlaceholder.classList.add("hidden");
        panelContent.classList.remove("hidden");
        resultsPanel.classList.remove("placeholder-state");

        // 1. Update Severity scores & progress bars
        updateSeverityScores(result.severity_scores);

        // 2. Update Gauges
        updateDriftGauges(result, payload.actual_sentence);

        // 3. Render bar chart
        renderSentencesChart(result.predictions);

        // 4. Update the side detail panel if a country was selected
        if (currentAuditCountry) {
            const countryMeta = globalTypologyList.find(d => d.country === currentAuditCountry);
            if (countryMeta) {
                showCountryDetails(countryMeta);
            }
        }

        // 5. Update ML Pipeline Trace
        updateMLInsights(result, payload);

    } catch (error) {
        alert(error.message);
    } finally {
        submitBtn.disabled = false;
        btnText.textContent = "Run Multilateral Audit";
        spinner.classList.add("hidden");
    }
}

// Update the progress bars and numeric scores under the radar chart
function updateSeverityScores(scores) {
    const dims = ["violence", "financial_loss", "victim_vulnerability", "premeditation", "public_safety_risk"];
    
    dims.forEach(dim => {
        const val = scores[dim];
        const fill = document.getElementById(`bar-${dim === "financial_loss" ? "financial" : dim === "public_safety_risk" ? "safety" : dim}`);
        if (fill) fill.style.width = `${val * 100}%`;
        
        const text = document.getElementById(`score-${dim === "financial_loss" ? "financial" : dim === "public_safety_risk" ? "safety" : dim}`);
        if (text) text.textContent = val.toFixed(2);
    });

    renderSeverityRadar(scores);
}

// Render the 3-model ML Pipeline insights displays
function updateMLInsights(result, payload) {
    if (!result) return;
    
    // 1. Model 1: NLP Transformer Embedding
    const vectorDisplay = document.getElementById("ml-vector-display");
    if (vectorDisplay && result.embedding_snippet) {
        const formattedSnippet = result.embedding_snippet.map(x => x.toFixed(5)).join(", ");
        vectorDisplay.textContent = `[\n  ${formattedSnippet},\n  ...\n]`;
    }

    // 2. Model 2: GBDT Classifier Features
    const featuresDisplay = document.getElementById("ml-features-display");
    if (featuresDisplay) {
        const sev = result.severity_scores;
        const priors = parseInt(payload.priors);
        const plea = payload.plea_guilty ? 1 : 0;
        const mit = payload.mitigating_circumstances ? 1 : 0;
        const juv = payload.juvenile ? 1 : 0;
        const reg = payload.court_region ? 1 : 0;
        
        featuresDisplay.textContent = `[\n` +
            `  violence: ${sev.violence.toFixed(3)},\n` +
            `  property_loss: ${sev.financial_loss.toFixed(3)},\n` +
            `  vulnerability: ${sev.victim_vulnerability.toFixed(3)},\n` +
            `  premeditation: ${sev.premeditation.toFixed(3)},\n` +
            `  safety_risk: ${sev.public_safety_risk.toFixed(3)},\n` +
            `  priors: ${priors},\n` +
            `  plea_guilty: ${plea},\n` +
            `  mitigation: ${mit},\n` +
            `  juvenile: ${juv},\n` +
            `  court_region: ${reg}\n` +
            `]`;
    }

    // 3. Model 3: Typology PCA Coordinates
    const clusterDisplay = document.getElementById("ml-cluster-display");
    if (clusterDisplay) {
        const activeCountry = currentAuditCountry || "United States";
        const countryMeta = globalTypologyList.find(d => d.country === activeCountry);
        if (countryMeta) {
            clusterDisplay.textContent = 
                `Country: ${activeCountry}\n` +
                `PCA X (Slope): ${countryMeta.x.toFixed(4)}\n` +
                `PCA Y (Mitigation): ${countryMeta.y.toFixed(4)}\n` +
                `Cluster: ${countryMeta.cluster_name}\n` +
                `Family: ${countryMeta.legal_family}`;
        } else {
            clusterDisplay.textContent = "Select country in Case Parameters to audit.";
        }
    }
}

// Render the 5-axis Radar chart of NLP Severity Scores
function renderSeverityRadar(scores) {
    const ctx = document.getElementById("severity-chart").getContext("2d");

    const data = [
        scores.violence,
        scores.financial_loss,
        scores.victim_vulnerability,
        scores.premeditation,
        scores.public_safety_risk
    ];

    if (severityChart) severityChart.destroy();

    severityChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Violence', 'Property Loss', 'Vulnerability', 'Premeditation', 'Safety Risk'],
            datasets: [{
                label: 'Extracted Severity',
                data: data,
                backgroundColor: 'rgba(99, 102, 241, 0.25)',
                borderColor: '#6366f1',
                pointBackgroundColor: '#6366f1',
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#6366f1',
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                r: {
                    angleLines: { color: document.documentElement.getAttribute("data-theme") === "light" ? "rgba(0, 0, 0, 0.08)" : "rgba(255, 255, 255, 0.06)" },
                    grid: { color: document.documentElement.getAttribute("data-theme") === "light" ? "rgba(0, 0, 0, 0.08)" : "rgba(255, 255, 255, 0.06)" },
                    pointLabels: { color: document.documentElement.getAttribute("data-theme") === "light" ? "#334155" : "#f1f5f9", font: { size: 9 } },
                    ticks: { display: false, stepSize: 0.2 },
                    min: 0,
                    max: 1.0
                }
            }
        }
    });
}

// Update the Drift Gauges using empirical quantiles and corrected math
function updateDriftGauges(result, actualSentence) {
    const referenceMedian = result.reference_median_months;
    document.getElementById("global-median-val").textContent = `${referenceMedian.toFixed(1)} months`;

    const withinVal = document.getElementById("within-drift-val");
    const withinStatus = document.getElementById("within-drift-status");
    const crossVal = document.getElementById("cross-drift-val");
    const crossStatus = document.getElementById("cross-drift-status");

    // 1. Within-Jurisdiction Audit (Quantile-based)
    if (actualSentence !== null && currentAuditCountry) {
        const countryPred = result.predictions[currentAuditCountry];
        const diff = countryPred.within_drift_diff; // diff in months
        const status = countryPred.within_drift_status;
        const q90 = countryPred.threshold_90;
        const q95 = countryPred.threshold_95;

        withinVal.textContent = `${diff >= 0 ? "+" : ""}${diff.toFixed(1)}m`;
        withinStatus.textContent = status;
        
        // Add specific thresholds to description block
        const parentBox = document.getElementById("within-drift-box");
        const descP = parentBox.querySelector(".gauge-desc");
        descP.innerHTML = `Empirical boundaries for ${currentAuditCountry}:<br>Moderate boundary: ${q90} months<br>Critical boundary: ${q95} months`;
        
        withinStatus.className = "gauge-label";
        if (status === "Consistent") withinStatus.classList.add("status-consistent");
        else if (status === "Moderate Drift") withinStatus.classList.add("status-moderate");
        else withinStatus.classList.add("status-critical");
    } else {
        withinVal.textContent = "-";
        withinStatus.textContent = "No Sentence Input";
        withinStatus.className = "gauge-label status-neutral";
        
        const parentBox = document.getElementById("within-drift-box");
        const descP = parentBox.querySelector(".gauge-desc");
        descP.textContent = "Checks if the actual sentence falls within the country's empirical 90% or 95% model residuals (quantile boundaries).";
    }

    // 2. Cross-Jurisdiction Audit
    const activeCountry = currentAuditCountry || "United States";
    const pred = result.predictions[activeCountry];
    const dev = pred.cross_drift; // % deviation from reference median

    crossVal.textContent = `${dev >= 0 ? "+" : ""}${dev.toFixed(1)}%`;
    crossStatus.textContent = `${activeCountry} vs Median`;
    
    crossStatus.className = "gauge-label";
    if (Math.abs(dev) <= 20) crossStatus.classList.add("status-consistent");
    else if (Math.abs(dev) <= 50) crossStatus.classList.add("status-moderate");
    else crossStatus.classList.add("status-critical");
}

// Render the 20-country horizontal bar chart
function renderSentencesChart(predictions) {
    const ctx = document.getElementById("sentences-chart").getContext("2d");

    const sorted = Object.keys(predictions)
        .map(country => ({ country, ...predictions[country] }))
        .sort((a, b) => b.sentence - a.sentence);

    const labels = sorted.map(item => item.country);
    const data = sorted.map(item => item.sentence);

    const barColors = sorted.map(item => {
        const family = item.legal_family;
        if (family === "Common Law") return "#6366f1"; // Indigo
        if (family === "Civil Law") return "#10b981"; // Emerald
        if (family === "Nordic Civil Law") return "#06b6d4"; // Cyan
        if (family === "Islamic Law") return "#f59e0b"; // Amber
        return "#ec4899"; // Pink for Mixed
    });

    if (sentencesChart) sentencesChart.destroy();

    sentencesChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: barColors,
                borderWidth: 0,
                borderRadius: 4,
                barThickness: 12
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (context) => {
                            const item = sorted[context.dataIndex];
                            return `Expected: ${context.formattedValue} months (${item.legal_family})`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: document.documentElement.getAttribute("data-theme") === "light" ? "rgba(0, 0, 0, 0.06)" : "rgba(255, 255, 255, 0.04)" },
                    ticks: { color: document.documentElement.getAttribute("data-theme") === "light" ? "#334155" : "#f1f5f9" },
                    title: { display: true, text: "Sentencing Length (Months)", color: document.documentElement.getAttribute("data-theme") === "light" ? "#64748b" : "#cbd5e1", font: { size: 10 } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: document.documentElement.getAttribute("data-theme") === "light" ? "#0f172a" : "#ffffff", font: { size: 10, weight: 500 } }
                }
            }
        }
    });
}

// Theme management helpers
function initTheme() {
    const themeToggle = document.getElementById("theme-toggle");
    
    // Check saved theme or user preferences
    const savedTheme = localStorage.getItem("theme");
    const systemPrefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    const initialTheme = savedTheme || (systemPrefersDark ? "dark" : "light");
    
    setTheme(initialTheme);
    
    themeToggle.addEventListener("click", () => {
        const currentTheme = document.documentElement.getAttribute("data-theme") || "dark";
        const nextTheme = currentTheme === "light" ? "dark" : "light";
        setTheme(nextTheme);
    });
}

function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("theme", theme);
    
    const themeToggle = document.getElementById("theme-toggle");
    const themeIcon = themeToggle.querySelector(".theme-toggle-icon");
    if (themeIcon) {
        themeIcon.textContent = theme === "light" ? "🌙" : "☀️";
    }
    
    // Update Chart.js global defaults
    if (window.Chart) {
        Chart.defaults.color = theme === 'light' ? '#334155' : '#f1f5f9';
        Chart.defaults.borderColor = theme === 'light' ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255, 255, 255, 0.08)';
        
        if (Chart.defaults.plugins && Chart.defaults.plugins.tooltip) {
            Chart.defaults.plugins.tooltip.backgroundColor = theme === 'light' ? 'rgba(255, 255, 255, 0.98)' : 'rgba(15, 23, 42, 0.96)';
            Chart.defaults.plugins.tooltip.titleColor = theme === 'light' ? '#0f172a' : '#ffffff';
            Chart.defaults.plugins.tooltip.bodyColor = theme === 'light' ? '#334155' : '#f1f5f9';
            Chart.defaults.plugins.tooltip.borderColor = theme === 'light' ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.12)';
        }
    }
    
    // Update active chart instances
    updateChartsTheme(theme);
}

function updateChartsTheme(theme) {
    const isLight = theme === 'light';
    
    if (severityChart) {
        severityChart.options.scales.r.angleLines.color = isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.06)';
        severityChart.options.scales.r.grid.color = isLight ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.06)';
        severityChart.options.scales.r.pointLabels.color = isLight ? '#334155' : '#f1f5f9';
        severityChart.update();
    }
    
    if (sentencesChart) {
        sentencesChart.options.scales.x.grid.color = isLight ? 'rgba(0, 0, 0, 0.06)' : 'rgba(255, 255, 255, 0.04)';
        sentencesChart.options.scales.x.ticks.color = isLight ? '#334155' : '#f1f5f9';
        sentencesChart.options.scales.x.title.color = isLight ? '#64748b' : '#cbd5e1';
        sentencesChart.options.scales.y.ticks.color = isLight ? '#0f172a' : '#ffffff';
        sentencesChart.update();
    }
    
    if (typologyMap) {
        typologyMap.options.scales.x.grid.color = isLight ? 'rgba(0, 0, 0, 0.05)' : 'rgba(255, 255, 255, 0.03)';
        typologyMap.options.scales.x.ticks.color = isLight ? '#64748b' : '#cbd5e1';
        typologyMap.options.scales.x.title.color = isLight ? '#64748b' : '#cbd5e1';
        typologyMap.options.scales.y.grid.color = isLight ? 'rgba(0, 0, 0, 0.05)' : 'rgba(255, 255, 255, 0.03)';
        typologyMap.options.scales.y.ticks.color = isLight ? '#64748b' : '#cbd5e1';
        typologyMap.options.scales.y.title.color = isLight ? '#64748b' : '#cbd5e1';
        typologyMap.update();
    }
}

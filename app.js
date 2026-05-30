// Aegis Churn Simulator Application Core Logic

// Active state object representing the current simulated customer features
const currentCustomer = {
    SeniorCitizen: 0,
    Partner: "No",
    Tenure: 12,
    MonthlyCharges: 75.0,
    SupportTickets: 3,
    Contract: "Month-to-month",
    InternetService: "Fiber optic",
    TechSupport: "No",
    PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check"
};

// Currently selected model: 'lr' (Logistic Regression) or 'dt' (Decision Tree)
let activeModel = 'lr';

// Customer Archetypes list for quick loading
const customerArchetypes = [
    {
        id: "sarah",
        name: "Sarah Jenkins",
        description: "Senior Citizen At-Risk Subscriber",
        risk: "High",
        data: {
            SeniorCitizen: 1,
            Partner: "No",
            Tenure: 3,
            MonthlyCharges: 95.0,
            SupportTickets: 4,
            Contract: "Month-to-month",
            InternetService: "Fiber optic",
            TechSupport: "No",
            PaperlessBilling: "Yes",
            PaymentMethod: "Electronic check"
        }
    },
    {
        id: "david",
        name: "David Chen",
        description: "Loyal 2-Year Committed Partner",
        risk: "Low",
        data: {
            SeniorCitizen: 0,
            Partner: "Yes",
            Tenure: 58,
            MonthlyCharges: 24.50,
            SupportTickets: 0,
            Contract: "Two year",
            InternetService: "No",
            TechSupport: "No",
            PaperlessBilling: "No",
            PaymentMethod: "Credit card"
        }
    },
    {
        id: "elena",
        name: "Elena Rostova",
        description: "High-Value Single Fiber Risk",
        risk: "Medium",
        data: {
            SeniorCitizen: 0,
            Partner: "No",
            Tenure: 14,
            MonthlyCharges: 108.0,
            SupportTickets: 2,
            Contract: "Month-to-month",
            InternetService: "Fiber optic",
            TechSupport: "No",
            PaperlessBilling: "Yes",
            PaymentMethod: "Electronic check"
        }
    },
    {
        id: "marcus",
        name: "Marcus Thompson",
        description: "Stable DSL Subscriber",
        risk: "Low",
        data: {
            SeniorCitizen: 0,
            Partner: "Yes",
            Tenure: 26,
            MonthlyCharges: 55.0,
            SupportTickets: 1,
            Contract: "One year",
            InternetService: "DSL",
            TechSupport: "Yes",
            PaperlessBilling: "No",
            PaymentMethod: "Bank transfer"
        }
    }
];

// Document Elements
document.addEventListener("DOMContentLoaded", () => {
    // 1. Verify that the Scikit-Learn Model Data is successfully imported
    if (typeof SCIKIT_LEARN_MODEL_DATA === 'undefined') {
        console.error("Scikit-Learn model data is missing. Please make sure train_model.py finished successfully.");
        return;
    }

    // 2. Load the Baseline KPIs into the Dashboard
    loadKPIs(SCIKIT_LEARN_MODEL_DATA);

    // 3. Render Customer Directory
    renderArchetypes(customerArchetypes);

    // 4. Initialize Interactive Range Sliders, Buttons, and Selects
    bindUIInputs();

    // 5. Build and render visual charts
    initCharts(SCIKIT_LEARN_MODEL_DATA);

    // 6. Run initial calculation based on defaults
    calculateChurnRisk();
});

// Load baseline KPIs derived from Scikit-Learn model statistics
function loadKPIs(modelData) {
    document.getElementById("kpi-total").innerText = modelData.averages.total_samples.toLocaleString();
    document.getElementById("kpi-churn-rate").innerText = `${(modelData.averages.churn_rate * 100).toFixed(1)}%`;
    document.getElementById("kpi-tenure").innerText = `${modelData.averages.tenure} mo`;
    
    // Dual accuracy stats
    document.getElementById("kpi-lr-acc").innerText = `${(modelData.lr_model.metrics.accuracy * 100).toFixed(1)}%`;
    document.getElementById("kpi-dt-acc").innerText = `${(modelData.dt_model.metrics.accuracy * 100).toFixed(1)}%`;
}

// Render dynamic customer profiles in the directory listing
function renderArchetypes(archetypes) {
    const container = document.getElementById("archetypes-container");
    if (!container) return;

    container.innerHTML = ""; // Clear loader
    
    archetypes.forEach(arch => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = `archetype-btn ${arch.id === 'sarah' ? 'active' : ''}`;
        btn.id = `arch-${arch.id}`;
        btn.onclick = () => loadArchetype(arch);
        
        btn.innerHTML = `
            <div class="archetype-header">
                <span class="archetype-name">${arch.name}</span>
                <span class="risk-tag ${arch.risk.toLowerCase()}">${arch.risk} Risk</span>
            </div>
            <div class="archetype-details">${arch.description}</div>
        `;
        container.appendChild(btn);
    });

    // Default load Sarah Jenkins
    if (archetypes.length > 0) {
        loadArchetype(archetypes[0], false);
    }
}

// Load chosen profile values directly into the interactive simulator
function loadArchetype(archetype, triggerRecalc = true) {
    // 1. Highlight clicked archetype card
    document.querySelectorAll(".archetype-btn").forEach(el => el.classList.remove("active"));
    const activeBtn = document.getElementById(`arch-${archetype.id}`);
    if (activeBtn) activeBtn.classList.add("active");

    // 2. Set object state
    Object.assign(currentCustomer, archetype.data);

    // 3. Update HTML Input values to match
    document.getElementById("tenure-input").value = currentCustomer.Tenure;
    document.getElementById("charges-input").value = currentCustomer.MonthlyCharges;
    document.getElementById("tickets-input").value = currentCustomer.SupportTickets;

    document.getElementById("tenure-value").innerText = `${currentCustomer.Tenure} months`;
    document.getElementById("charges-value").innerText = `$${currentCustomer.MonthlyCharges.toFixed(2)}`;
    document.getElementById("tickets-value").innerText = `${currentCustomer.SupportTickets} ticket${currentCustomer.SupportTickets === 1 ? '' : 's'}`;
    
    // Manage ticket badge coloring
    const tBadge = document.getElementById("tickets-value");
    if (currentCustomer.SupportTickets >= 4) {
        tBadge.className = "slider-badge alert-badge";
    } else {
        tBadge.className = "slider-badge";
    }

    // Set segment and toggles (Demographics)
    setToggleActive("senior-control", currentCustomer.SeniorCitizen === 1 ? "Yes" : "No");
    setToggleActive("partner-control", currentCustomer.Partner);

    // Services
    setSegmentActive("contract-control", currentCustomer.Contract);
    setSegmentActive("internet-control", currentCustomer.InternetService);
    setToggleActive("techsupport-control", currentCustomer.TechSupport);
    setToggleActive("paperless-control", currentCustomer.PaperlessBilling);

    document.getElementById("payment-input").value = currentCustomer.PaymentMethod;

    // Reset intervention button styles
    resetPlaybookInterventions();

    // 4. Calculate Risk
    if (triggerRecalc) {
        calculateChurnRisk();
    }
}

// Setup input listeners on all HTML interactive form controls
function bindUIInputs() {
    // Model selector segmented control
    const modelSelector = document.getElementById("model-selector");
    modelSelector.addEventListener("click", (e) => {
        const btn = e.target.closest("button");
        if (!btn) return;
        
        modelSelector.querySelectorAll("button").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        
        activeModel = btn.dataset.val;
        
        // Dynamic UI switches
        const mathBoxTitle = document.getElementById("math-box-title");
        const lrView = document.getElementById("lr-diagnostic-view");
        const dtView = document.getElementById("dt-diagnostic-view");
        
        if (activeModel === 'lr') {
            mathBoxTitle.innerText = "Underlying Logistic Regression Formula:";
            lrView.classList.remove("hidden");
            dtView.classList.add("hidden");
        } else {
            mathBoxTitle.innerText = "Underlying Decision Tree Split Path:";
            lrView.classList.add("hidden");
            dtView.classList.remove("hidden");
        }
        
        // Dynamically update drivers importance chart
        updateImportanceChartTitle();
        renderImportanceChart(SCIKIT_LEARN_MODEL_DATA);
        
        // Recalc
        calculateChurnRisk();
    });

    // Demographics Toggles
    bindToggleButtons("senior-control", (val) => {
        currentCustomer.SeniorCitizen = val === "Yes" ? 1 : 0;
        calculateChurnRisk();
    });
    
    bindToggleButtons("partner-control", (val) => {
        currentCustomer.Partner = val;
        calculateChurnRisk();
    });

    // Sliders
    const tenureInput = document.getElementById("tenure-input");
    tenureInput.addEventListener("input", (e) => {
        currentCustomer.Tenure = parseInt(e.target.value);
        document.getElementById("tenure-value").innerText = `${currentCustomer.Tenure} months`;
        calculateChurnRisk();
    });

    const chargesInput = document.getElementById("charges-input");
    chargesInput.addEventListener("input", (e) => {
        currentCustomer.MonthlyCharges = parseFloat(e.target.value);
        document.getElementById("charges-value").innerText = `$${currentCustomer.MonthlyCharges.toFixed(2)}`;
        calculateChurnRisk();
    });

    const ticketsInput = document.getElementById("tickets-input");
    ticketsInput.addEventListener("input", (e) => {
        currentCustomer.SupportTickets = parseInt(e.target.value);
        const tVal = document.getElementById("tickets-value");
        tVal.innerText = `${currentCustomer.SupportTickets} ticket${currentCustomer.SupportTickets === 1 ? '' : 's'}`;
        
        if (currentCustomer.SupportTickets >= 4) {
            tVal.className = "slider-badge alert-badge";
        } else {
            tVal.className = "slider-badge";
        }
        calculateChurnRisk();
    });

    // Contract segmented buttons
    bindSegmentButtons("contract-control", (val) => {
        currentCustomer.Contract = val;
        calculateChurnRisk();
    });

    // Internet segmented buttons
    bindSegmentButtons("internet-control", (val) => {
        currentCustomer.InternetService = val;
        if (val === "No") {
            currentCustomer.TechSupport = "No";
            setToggleActive("techsupport-control", "No");
        }
        calculateChurnRisk();
    });

    // Tech Support toggle buttons
    bindToggleButtons("techsupport-control", (val) => {
        if (val === "Yes" && currentCustomer.InternetService === "No") {
            currentCustomer.InternetService = "DSL";
            setSegmentActive("internet-control", "DSL");
        }
        currentCustomer.TechSupport = val;
        calculateChurnRisk();
    });

    // Paperless Billing toggle buttons
    bindToggleButtons("paperless-control", (val) => {
        currentCustomer.PaperlessBilling = val;
        calculateChurnRisk();
    });

    // Payment Method select box
    const paymentSelect = document.getElementById("payment-input");
    paymentSelect.addEventListener("change", (e) => {
        currentCustomer.PaymentMethod = e.target.value;
        calculateChurnRisk();
    });
}

// Dynamically updates drivers title label
function updateImportanceChartTitle() {
    const title = document.getElementById("drivers-chart-title");
    if (!title) return;
    
    if (activeModel === 'lr') {
        title.innerText = "Customer Driver Impact (LR Coefficients)";
    } else {
        title.innerText = "Customer Driver Impact (DT Importance)";
    }
}

// Compute the mathematical predictions based on the actively selected model engine
function calculateChurnRisk() {
    if (activeModel === 'lr') {
        calculateLogisticRegressionRisk();
    } else {
        calculateDecisionTreeRisk();
    }
}

// ---------------- MODEL 1: LOGISTIC REGRESSION ----------------
function calculateLogisticRegressionRisk() {
    const model = SCIKIT_LEARN_MODEL_DATA.lr_model;
    const coef = model.coefficients;
    const intercept = model.intercept;

    // 1. One-hot encode/map UI features to binary input variables
    const Tenure = currentCustomer.Tenure;
    const MonthlyCharges = currentCustomer.MonthlyCharges;
    const SupportTickets = currentCustomer.SupportTickets;
    const SeniorCitizen = currentCustomer.SeniorCitizen;
    const Partner_Yes = (currentCustomer.Partner === "Yes") ? 1 : 0;

    const Contract_MonthToMonth = (currentCustomer.Contract === "Month-to-month") ? 1 : 0;
    const Contract_TwoYear = (currentCustomer.Contract === "Two year") ? 1 : 0;
    const Internet_FiberOptic = (currentCustomer.InternetService === "Fiber optic") ? 1 : 0;
    const TechSupport_No = (currentCustomer.TechSupport === "No") ? 1 : 0;
    const Payment_ElectronicCheck = (currentCustomer.PaymentMethod === "Electronic check") ? 1 : 0;
    const PaperlessBilling_Yes = (currentCustomer.PaperlessBilling === "Yes") ? 1 : 0;

    // 2. Sum weights + Intercept
    let z = intercept;
    z += coef.Tenure * Tenure;
    z += coef.MonthlyCharges * MonthlyCharges;
    z += coef.SupportTickets * SupportTickets;
    z += coef.SeniorCitizen * SeniorCitizen;
    z += coef.Partner_Yes * Partner_Yes;
    
    z += coef.Contract_MonthToMonth * Contract_MonthToMonth;
    z += coef.Contract_TwoYear * Contract_TwoYear;
    z += coef.Internet_FiberOptic * Internet_FiberOptic;
    z += coef.TechSupport_No * TechSupport_No;
    z += coef.Payment_ElectronicCheck * Payment_ElectronicCheck;
    z += coef.PaperlessBilling_Yes * PaperlessBilling_Yes;

    // 3. Sigmoid P = 1 / (1 + e^-z)
    const probability = 1 / (1 + Math.exp(-z));

    // 4. Update Gauge & Diagnostics
    updateGauge(probability);
    updateLRDiagnostics(z, probability);
}

function updateLRDiagnostics(z, prob) {
    const exprBox = document.getElementById("math-expanded");
    const explanationBox = document.getElementById("diagnostic-explanation");
    
    if (!exprBox || !explanationBox) return;

    const coef = SCIKIT_LEARN_MODEL_DATA.lr_model.coefficients;
    const intercept = SCIKIT_LEARN_MODEL_DATA.lr_model.intercept;

    exprBox.innerHTML = `z = ${intercept.toFixed(2)}` + 
        ` + (${coef.Tenure.toFixed(3)} &times; ${currentCustomer.Tenure})` +
        ` + (${coef.MonthlyCharges.toFixed(4)} &times; ${currentCustomer.MonthlyCharges.toFixed(0)})` +
        ` + (${coef.SupportTickets.toFixed(3)} &times; ${currentCustomer.SupportTickets})` +
        ` + (${coef.SeniorCitizen.toFixed(3)} &times; ${currentCustomer.SeniorCitizen})` +
        ` + (${coef.Partner_Yes.toFixed(3)} &times; ${(currentCustomer.Partner === 'Yes' ? 1 : 0)})` +
        ` + (${coef.Contract_MonthToMonth.toFixed(3)} &times; ${(currentCustomer.Contract === 'Month-to-month' ? 1 : 0)})` +
        ` + (${coef.Contract_TwoYear.toFixed(3)} &times; ${(currentCustomer.Contract === 'Two year' ? 1 : 0)})` +
        ` = <strong>${z.toFixed(3)}</strong>`;

    explanationBox.innerHTML = getBusinessDiagnosis(prob);
}

// ---------------- MODEL 2: DECISION TREE ----------------
function calculateDecisionTreeRisk() {
    const root = SCIKIT_LEARN_MODEL_DATA.dt_model.decision_tree;
    
    // Evaluate DT recursively
    const path = [];
    const evaluation = evaluateDecisionTree(root, currentCustomer, path);
    
    const probability = evaluation.probability;
    
    // Update Gauge & Diagnostics
    updateGauge(probability);
    updateDTDiagnostics(path, probability);
}

// JS Recursive Decision Tree Traversal crawling structure
function evaluateDecisionTree(node, customer, path) {
    if (node.type === "leaf") {
        return {
            probability: node.probability,
            samples: node.samples
        };
    }
    
    const featureName = node.feature;
    const threshold = node.threshold;
    
    // Resolve UI state to actual encoder values
    const featureVal = getFeatureValForDT(featureName, customer);
    
    let stepDesc = "";
    let nextNode = null;
    
    if (featureVal <= threshold) {
        stepDesc = `${formatFeatureLabel(featureName)} is ${formatFeatureValDisplay(featureName, featureVal)} (≤ ${threshold})`;
        nextNode = node.left;
    } else {
        stepDesc = `${formatFeatureLabel(featureName)} is ${formatFeatureValDisplay(featureName, featureVal)} (> ${threshold})`;
        nextNode = node.right;
    }
    
    path.push({
        desc: stepDesc,
        feature: featureName,
        value: featureVal,
        threshold: threshold
    });
    
    return evaluateDecisionTree(nextNode, customer, path);
}

// Dynamic display of Decision split rule pathways
function updateDTDiagnostics(path, prob) {
    const container = document.getElementById("tree-path-container");
    const explanationBox = document.getElementById("diagnostic-explanation");
    
    if (!container || !explanationBox) return;
    
    container.innerHTML = ""; // Clear baseline
    
    path.forEach((step, idx) => {
        const stepEl = document.createElement("div");
        stepEl.className = "rule-step active";
        
        stepEl.innerHTML = `
            <span class="rule-step-desc">${idx + 1}. ${step.desc}</span>
            <span class="rule-step-val">Matched</span>
        `;
        container.appendChild(stepEl);
    });
    
    // Add leaf node prediction step
    const leafEl = document.createElement("div");
    leafEl.className = "rule-step terminal active";
    leafEl.innerHTML = `
        <span class="rule-step-desc">➔ Reached Leaf Node: Predict Churn Risk</span>
        <span class="rule-step-val" style="color: var(--accent-rose); border-color: rgba(244, 63, 94, 0.25); background: rgba(244, 63, 94, 0.1);">${(prob * 100).toFixed(1)}%</span>
    `;
    container.appendChild(leafEl);
    
    explanationBox.innerHTML = getBusinessDiagnosis(prob);
}

// Helper: Map standard feature indices to values for Decision Tree
function getFeatureValForDT(featureName, customer) {
    switch (featureName) {
        case "Tenure":
            return customer.Tenure;
        case "MonthlyCharges":
            return customer.MonthlyCharges;
        case "SupportTickets":
            return customer.SupportTickets;
        case "SeniorCitizen":
            return customer.SeniorCitizen;
        case "Partner_Yes":
            return customer.Partner === "Yes" ? 1 : 0;
        case "Contract_MonthToMonth":
            return customer.Contract === "Month-to-month" ? 1 : 0;
        case "Contract_TwoYear":
            return customer.Contract === "Two year" ? 1 : 0;
        case "Internet_FiberOptic":
            return customer.InternetService === "Fiber optic" ? 1 : 0;
        case "TechSupport_No":
            return customer.TechSupport === "No" ? 1 : 0;
        case "Payment_ElectronicCheck":
            return customer.PaymentMethod === "Electronic check" ? 1 : 0;
        case "PaperlessBilling_Yes":
            return customer.PaperlessBilling === "Yes" ? 1 : 0;
        default:
            return 0;
    }
}

// Formats categorical values correctly for diagnostic display
function formatFeatureValDisplay(feature, val) {
    if (feature === "Tenure") return `${val} months`;
    if (feature === "MonthlyCharges") return `$${val.toFixed(2)}`;
    if (feature === "SupportTickets") return `${val} ticket${val === 1 ? '' : 's'}`;
    
    // Categoricals
    if (val === 1) return "Yes/Active";
    return "No/Inactive";
}

function formatFeatureLabel(feature) {
    const mapping = {
        'Tenure': 'Tenure',
        'MonthlyCharges': 'Monthly Charges',
        'SupportTickets': 'Support Tickets',
        'SeniorCitizen': 'Senior Citizen',
        'Partner_Yes': 'Has Partner',
        'Contract_MonthToMonth': 'Contract Month-to-Month',
        'Contract_TwoYear': 'Contract 2-Year',
        'Internet_FiberOptic': 'Fiber Optic Service',
        'TechSupport_No': 'Tech Support Disabled',
        'Payment_ElectronicCheck': 'Pay with Electronic Check',
        'PaperlessBilling_Yes': 'Paperless Invoicing'
    };
    return mapping[feature] || feature;
}

// Returns dynamic explanatory business diagnoses
function getBusinessDiagnosis(prob) {
    let diagnosis = "";
    if (prob >= 0.70) {
        diagnosis = `This customer exhibits highly elevated risk levels. Key drivers are the <strong>${currentCustomer.Contract} Contract</strong>, <strong>${currentCustomer.SupportTickets} Support Grievance(s)</strong>, and a lack of <strong>Tech Support</strong>. Operational intervention is strongly advised.`;
    } else if (prob >= 0.40) {
        diagnosis = `Customer falls in the moderate danger zone. A combination of price pressure (<strong>$${currentCustomer.MonthlyCharges.toFixed(2)}/mo</strong>) and transactional contracting increases probability. Offering commitment loyalty rewards could stabilize this account.`;
    } else if (prob >= 0.15) {
        diagnosis = `The account shows low volatility. While minor drivers exist, tenure stability (<strong>${currentCustomer.Tenure} months</strong>) balances out potential risk variables. Maintain standard relationship management.`;
    } else {
        diagnosis = `Excellent account health. Multi-year commitments and zero support friction render this customer highly stable. Model computes negligible probability of near-term churn.`;
    }
    return diagnosis;
}

// ---------------- DYNAMIC METER UPDATES ----------------
function updateGauge(prob) {
    const displayVal = document.getElementById("probability-display");
    const statusVal = document.getElementById("status-display");
    const fillRing = document.getElementById("gauge-fill");
    
    if (!displayVal || !statusVal || !fillRing) return;

    displayVal.innerText = `${(prob * 100).toFixed(0)}%`;

    const maxOffset = 534;
    const offsetVal = maxOffset - (prob * maxOffset);
    fillRing.style.strokeDashoffset = offsetVal;

    let statusClass = "status-safe";
    let statusText = "SAFE";
    let strokeColor = "rgb(16, 185, 129)";

    if (prob >= 0.70) {
        statusClass = "status-high";
        statusText = "CRITICAL RISK";
        strokeColor = "rgb(244, 63, 94)";
    } else if (prob >= 0.40) {
        statusClass = "status-medium";
        statusText = "MODERATE RISK";
        strokeColor = "rgb(245, 158, 11)";
    } else if (prob >= 0.15) {
        statusClass = "status-low";
        statusText = "STABLE RISK";
        strokeColor = "rgb(99, 102, 241)";
    }

    statusVal.className = `probability-status ${statusClass}`;
    statusVal.innerText = statusText;
    fillRing.style.stroke = strokeColor;
    fillRing.style.filter = `drop-shadow(0 0 6px ${strokeColor})`;
}

// ---------------- RETENTION PLAYBOOK INTERVENTIONS ----------------
function applyIntervention(type) {
    const discountBtn = document.getElementById("strategy-discount");
    const contractBtn = document.getElementById("strategy-contract");
    const techBtn = document.getElementById("strategy-tech");
    const ticketBtn = document.getElementById("strategy-tickets");

    if (type === 'discount') {
        const chargesSlider = document.getElementById("charges-input");
        const newVal = Math.max(15, currentCustomer.MonthlyCharges * 0.85);
        
        chargesSlider.value = newVal;
        currentCustomer.MonthlyCharges = newVal;
        document.getElementById("charges-value").innerText = `$${newVal.toFixed(2)}`;
        
        discountBtn.classList.add("active");
        discountBtn.style.pointerEvents = "none";
        discountBtn.querySelector(".playbook-action").innerText = "Applied";
    }
    
    else if (type === 'contract') {
        if (currentCustomer.Contract === "Month-to-month") {
            currentCustomer.Contract = "One year";
            setSegmentActive("contract-control", "One year");
            
            contractBtn.classList.add("active");
            contractBtn.style.pointerEvents = "none";
            contractBtn.querySelector(".playbook-action").innerText = "Applied";
        }
    }
    
    else if (type === 'support') {
        if (currentCustomer.TechSupport === "No") {
            currentCustomer.TechSupport = "Yes";
            setToggleActive("techsupport-control", "Yes");
            
            if (currentCustomer.InternetService === "No") {
                currentCustomer.InternetService = "DSL";
                setSegmentActive("internet-control", "DSL");
            }

            techBtn.classList.add("active");
            techBtn.style.pointerEvents = "none";
            techBtn.querySelector(".playbook-action").innerText = "Applied";
        }
    }
    
    else if (type === 'resolve') {
        if (currentCustomer.SupportTickets > 0) {
            const ticketSlider = document.getElementById("tickets-input");
            ticketSlider.value = 0;
            currentCustomer.SupportTickets = 0;
            
            const badge = document.getElementById("tickets-value");
            badge.innerText = "0 tickets";
            badge.className = "slider-badge";

            ticketBtn.classList.add("active");
            ticketBtn.style.pointerEvents = "none";
            ticketBtn.querySelector(".playbook-action").innerText = "Applied";
        }
    }

    calculateChurnRisk();
}

function resetPlaybookInterventions() {
    const playbooks = ["strategy-discount", "strategy-contract", "strategy-tech", "strategy-tickets"];
    playbooks.forEach(id => {
        const item = document.getElementById(id);
        if (item) {
            item.classList.remove("active");
            item.style.pointerEvents = "auto";
            item.querySelector(".playbook-action").innerText = "Apply";
        }
    });
}

// ---------------- UI HELPERS ----------------
function bindSegmentButtons(containerId, callback) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.addEventListener("click", (e) => {
        const btn = e.target.closest("button");
        if (!btn) return;

        container.querySelectorAll("button").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        
        callback(btn.dataset.val);
    });
}

function setSegmentActive(containerId, value) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.querySelectorAll("button").forEach(b => {
        if (b.dataset.val === value) {
            b.classList.add("active");
        } else {
            b.classList.remove("active");
        }
    });
}

function bindToggleButtons(containerId, callback) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.addEventListener("click", (e) => {
        const btn = e.target.closest("button");
        if (!btn) return;

        container.querySelectorAll("button").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        
        callback(btn.dataset.val);
    });
}

function setToggleActive(containerId, value) {
    const container = document.getElementById(containerId);
    if (!container) return;

    container.querySelectorAll("button").forEach(b => {
        if (b.dataset.val === value) {
            b.classList.add("active");
        } else {
            b.classList.remove("active");
        }
    });
}

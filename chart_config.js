// Chart.js Configuration for Aegis Churn Dashboard

let importanceChart = null;
let segmentsChart = null;

// Initialize charts when the DOM is fully loaded and model data is available
function initCharts(modelData) {
    if (typeof Chart === 'undefined') {
        console.error("Chart.js is not loaded.");
        return;
    }

    // Set global default styles for dark mode aesthetics
    Chart.defaults.color = '#9ca3af'; // var(--text-secondary)
    Chart.defaults.font.family = "'Inter', sans-serif";
    Chart.defaults.font.size = 11;
    Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(15, 23, 42, 0.9)';
    Chart.defaults.plugins.tooltip.titleFont = { family: "'Outfit', sans-serif", weight: '600' };
    Chart.defaults.plugins.tooltip.borderColor = 'rgba(255, 255, 255, 0.1)';
    Chart.defaults.plugins.tooltip.borderWidth = 1;

    renderImportanceChart(modelData);
    renderSegmentsChart(modelData);
}

// 1. Horizontal Bar Chart for Feature Drivers (Coefficients / Importances)
function renderImportanceChart(modelData) {
    const ctx = document.getElementById('importance-chart');
    if (!ctx) return;

    let coefs = {};
    let isTree = false;

    // Load active model variables (activeModel is globally defined in app.js)
    if (typeof activeModel !== 'undefined' && activeModel === 'dt') {
        coefs = modelData.dt_model.importances;
        isTree = true;
    } else {
        coefs = modelData.lr_model.coefficients;
    }
    
    // Sort features by absolute value for a cleaner horizontal listing
    const sortedFeatures = Object.keys(coefs)
        .map(key => ({
            name: formatFeatureName(key),
            rawKey: key,
            value: coefs[key]
        }))
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

    const labels = sortedFeatures.map(item => item.name);
    const data = sortedFeatures.map(item => item.value);
    
    let backgroundColors = [];
    let borderColors = [];

    if (isTree) {
        // Decision Tree values are strictly positive importances (0 to 1). Color them indigo/purple.
        backgroundColors = data.map(() => 'rgba(168, 85, 247, 0.65)'); // Purple
        borderColors = data.map(() => 'rgb(168, 85, 247)');
    } else {
        // Logistic Regression: positive coefficient (increases churn) -> Rose, negative -> Cyan
        backgroundColors = data.map(val => 
            val > 0 ? 'rgba(244, 63, 94, 0.65)' : 'rgba(6, 182, 212, 0.65)'
        );
        borderColors = data.map(val => 
            val > 0 ? 'rgb(244, 63, 94)' : 'rgb(6, 182, 212)'
        );
    }

    if (importanceChart) {
        importanceChart.destroy();
    }

    importanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: isTree ? 'Feature Gini Importance' : 'Log-Odds Coefficient Impact',
                data: data,
                backgroundColor: backgroundColors,
                borderColor: borderColors,
                borderWidth: 1.5,
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
                        label: function(context) {
                            const val = context.raw;
                            if (isTree) {
                                return ` Information Gain Importance: ${val.toFixed(4)}`;
                            } else {
                                const desc = val > 0 ? "Increases churn risk" : "Decreases churn risk";
                                return ` Coefficient: ${val > 0 ? '+' : ''}${val.toFixed(4)} (${desc})`;
                            }
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)', tickColor: 'transparent' },
                    border: { dash: [4, 4] },
                    title: { 
                        display: true, 
                        text: isTree ? 'Information Value (0.0 to 1.0)' : 'Model Weight (Positive = Risk | Negative = Retention)', 
                        color: '#6b7280' 
                    }
                },
                y: {
                    grid: { display: false }
                }
            }
        }
    });
}

// 2. Vertical Bar Chart for Baseline Segment Churn Risks
function renderSegmentsChart(modelData) {
    const ctx = document.getElementById('segments-chart');
    if (!ctx) return;

    const segments = modelData.segments.churn_by_contract;
    const labels = Object.keys(segments).map(k => k === "Month-to-month" ? "Monthly" : k);
    const data = Object.values(segments).map(v => v * 100); // convert decimal to percentage

    if (segmentsChart) {
        segmentsChart.destroy();
    }

    segmentsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Avg Churn Rate',
                data: data,
                backgroundColor: [
                    'rgba(244, 63, 94, 0.55)', // Monthly -> Rose
                    'rgba(245, 158, 11, 0.55)', // 1 Yr -> Gold
                    'rgba(16, 185, 129, 0.55)'  // 2 Yr -> Emerald
                ],
                borderColor: [
                    'rgb(244, 63, 94)',
                    'rgb(245, 158, 11)',
                    'rgb(16, 185, 129)'
                ],
                borderWidth: 1.5,
                borderRadius: 6,
                barThickness: 32
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return ` Churn Rate: ${context.raw.toFixed(1)}%`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false }
                },
                y: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)', tickColor: 'transparent' },
                    border: { dash: [4, 4] },
                    ticks: {
                        callback: function(value) { return value + '%'; }
                    },
                    max: Math.ceil(Math.max(...data) / 10) * 10 + 10 // Round up to nearest 10% + cushion
                }
            }
        }
    });
}

// Utility function to format pythonic dummy variable names to clean business terms
function formatFeatureName(key) {
    const mapping = {
        'Tenure': 'Tenure (Months)',
        'MonthlyCharges': 'Monthly charges ($)',
        'SupportTickets': 'Support Tickets',
        'SeniorCitizen': 'Senior Citizen',
        'Partner_Yes': 'Has Partner',
        'Contract_MonthToMonth': 'Contract: Monthly',
        'Contract_TwoYear': 'Contract: 2-Year',
        'Internet_FiberOptic': 'Internet: Fiber Optic',
        'TechSupport_No': 'No Tech Support',
        'Payment_ElectronicCheck': 'Pay: Electronic Check',
        'PaperlessBilling_Yes': 'Billing: Paperless'
    };
    return mapping[key] || key;
}

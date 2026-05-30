# Aegis Churn Analytics & Multi-Model Simulator

An advanced customer churn prediction platform that bridges data science and software engineering. It combines a backend machine learning pipeline trained in **Python (Scikit-Learn)** with a premium, real-time interactive **Explainable AI (XAI) Web Dashboard** running natively in the browser.

---

## 🚀 Key Features

*   **Dual ML Engine Support**: Trains and fits two separate machine learning architectures:
    1.  **Logistic Regression** (Linear Classifier: 78.5% Accuracy | 0.8420 ROC-AUC)
    2.  **Decision Tree Classifier** (Hierarchical Rules: 77.5% Accuracy | 0.8135 ROC-AUC)
*   **Zero-Latency Client-Side Simulator**: Translates fitted Scikit-Learn weights and recursive tree matrices into native JavaScript, allowing predictions to re-evaluate in `< 1ms` as sliders are moved.
*   **Explainable AI (XAI) Diagnostics**:
    *   *Logistic Regression*: Visualizes linear log-odds formulas: $z = \beta_0 + \sum \beta_i X_i$.
    *   *Decision Tree*: Recursively crawls the serialized tree structure and renders a glowing stepper path representing the *exact rules* the customer satisfied.
*   **Proactive Retention Playbook**: Clickable operational cards ( loyalty discounts, contract commitments, support interventions) that instantly update user inputs and simulate risk reductions.
*   **Dynamic Visualizations**: Render interactive **Chart.js** graphs showing feature weights/gini importances and segment risks which automatically update based on the active model selector.
*   **Automated Testing Suite**: Full pipeline verification unit tests using **pytest** covering data synthesis, pre-processing matrices, and model fits.

---

## 🛠️ Technology Stack

*   **Backend Pipeline**: Python 3.14, Scikit-Learn, Pandas, NumPy
*   **Testing Suite**: Pytest, Unittest
*   **Frontend Dashboard**: HTML5, Vanilla CSS3 (Glassmorphism), ES6+ JavaScript, Chart.js (via CDN)
*   **Deployment**: Static hosting via GitHub Pages

---

## 📂 Repository Structure

```
├── train_model.py          # Data generation, dual model fit, and JSON parameter exporter
├── test_pipeline.py         # Pytest suite verifying backend configurations
├── customer_churn_data.csv # Exported database containing 1,000 synthetic customer records
├── model_data.js           # Shared ML parameter bridge (Intercept, weights, tree JSON)
├── index.html              # Premium dashboard web interface
├── styles.css              # Custom glassmorphic styles, custom ranges, and step indicators
├── app.js                  # Math simulator, recursive splits evaluator, and UI bindings
├── chart_config.js         # Reactive horizontal and vertical Chart.js plots
└── README.md               # Documentation guide
```

---

## 🔧 Installation & Usage Guide

### Prerequisites
*   Git
*   Python 3.x with Scikit-Learn, Pandas, and NumPy installed

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Run the Unit Tests
Verify pipeline and serialization integrity using `pytest`:
```bash
python -m pytest test_pipeline.py
```

### 3. Retrain the Models & Export Parameters
```bash
python train_model.py
```
This generates `customer_churn_data.csv` and updates `model_data.js` with new weights.

### 4. Run the Dashboard Web App
Simply open `index.html` in any web browser, or launch a quick local server:
```bash
python -m http.server 8000
```
Then visit `http://localhost:8000` in your web browser.

---

## 🌐 Deploying to GitHub Pages (Get a Hosted Link)

To host your static dashboard online for free using **GitHub Pages**:

1.  Push your code to your GitHub repository (see the setup guide below).
2.  On GitHub, navigate to your **Repository Settings**.
3.  Scroll down to the **Pages** section in the left sidebar.
4.  Under **Build and deployment**, set the Source to **Deploy from a branch**.
5.  Set the branch to **main** (or your active branch) and directory to **`/ (root)`**.
6.  Click **Save**.
7.  Within 1-2 minutes, your website will be live at:
    `https://<YOUR_USERNAME>.github.io/<YOUR_REPO_NAME>/`

---

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

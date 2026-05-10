const modelSelect = document.getElementById("model-select");
const metricsContainer = document.getElementById("metrics-container");
const comparisonTable = document.getElementById("comparison-table");
const predictionForm = document.getElementById("prediction-form");
const predictionResult = document.getElementById("prediction-result");
const featuresInput = document.getElementById("features-json");

let models = [];
let metricsByModel = {};

const metricLabels = {
  accuracy: "Accuracy",
  f1_score: "F1 Score",
  precision: "Precision",
  recall: "Recall",
  roc_auc: "ROC AUC",
};

const defaultFeatures = {
  wind_speed_knts: 0,
  visibility_m: 0,
  temperature_deg: 0,
};

function formatMetric(value) {
  if (typeof value !== "number") {
    return "-";
  }
  return value.toFixed(4);
}

function getModelLabel(modelName) {
  const model = models.find((item) => item.name === modelName);
  return model ? model.label : modelName;
}

function renderMetrics(modelName) {
  const metrics = metricsByModel[modelName];
  if (!modelName) {
    metricsContainer.innerHTML = '<p class="muted">Model seçin.</p>';
    return;
  }

  if (!metrics || Object.keys(metrics).length === 0) {
    metricsContainer.innerHTML = '<p class="muted">Bu model için metrik bulunamadı.</p>';
    return;
  }

  metricsContainer.innerHTML = Object.entries(metricLabels)
    .map(([key, label]) => {
      return `
        <div class="metric-box">
          <span>${label}</span>
          <strong>${formatMetric(metrics[key])}</strong>
        </div>
      `;
    })
    .join("");
}

function renderComparisonTable() {
  if (models.length === 0) {
    comparisonTable.innerHTML = '<p class="muted">Yüklü model bulunamadı.</p>';
    return;
  }

  const headers = Object.values(metricLabels).map((label) => `<th>${label}</th>`).join("");
  const rows = models
    .map((model) => {
      const metrics = metricsByModel[model.name] || {};
      const values = Object.keys(metricLabels)
        .map((key) => `<td>${formatMetric(metrics[key])}</td>`)
        .join("");
      return `<tr><th scope="row">${model.label}</th>${values}</tr>`;
    })
    .join("");

  comparisonTable.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            ${headers}
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

async function loadModels() {
  const response = await fetch("/api/models");
  if (!response.ok) {
    throw new Error("Modeller yüklenemedi.");
  }

  const data = await response.json();
  models = data.models || [];
  metricsByModel = data.metrics || {};

  modelSelect.innerHTML = models
    .map((model) => `<option value="${model.name}">${model.label}</option>`)
    .join("");

  if (models.length > 0) {
    modelSelect.value = models[0].name;
    renderMetrics(models[0].name);
  } else {
    renderMetrics("");
  }
  renderComparisonTable();
}

window.updateMetrics = function updateMetrics() {
  renderMetrics(modelSelect.value);
};

predictionForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  predictionResult.innerHTML = "";

  try {
    const payload = {
      model_name: modelSelect.value,
      data: JSON.parse(featuresInput.value),
    };

    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Tahmin başarısız.");
    }

    const goAroundProbability = data.probability ? formatMetric(data.probability.goaround) : "-";
    predictionResult.innerHTML = `
      <div class="result-box">
        <p><strong>Model:</strong> ${getModelLabel(data.model)}</p>
        <p><strong>Tahmin:</strong> ${data.prediction === 1 ? "Go-around" : "No go-around"}</p>
        <p><strong>Go-around olasılığı:</strong> ${goAroundProbability}</p>
      </div>
    `;
  } catch (error) {
    predictionResult.innerHTML = `<div class="error-box">${String(error.message || error)}</div>`;
  }
});

featuresInput.value = JSON.stringify(defaultFeatures, null, 2);
loadModels().catch((error) => {
  metricsContainer.innerHTML = `<p class="error-text">${String(error.message || error)}</p>`;
  comparisonTable.innerHTML = "";
});

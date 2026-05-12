"use strict";

const SAMPLE = {
  airport: "EDDF", runway: "25L", wtc: "M", typecode: "A320",
  icaoaircrafttype: "L2J", airport_country: "DE", airport_region: "EU",
  operator_country: "DE", operator_region: "EU", has_intersection: "0",
  glide_slope_angle: 3.0, rwy_length: 4000, month: 5, day_of_week: 2, hour_utc: 14,
  wind_speed_knts: 12, wind_dir_deg: 270, wind_gust_knts: 18,
  visibility_m: 8000, temperature_deg: 15, press_sea_level_p: 1013,
  press_p: 1010, weather_intensity: "", weather_precipitation: "",
  weather_desc: "", weather_obscuration: "", weather_other: "",
};

const DERIVED_ORDER = [
  "runway_heading_deg",
  "headwind_knts",
  "tailwind_knts",
  "crosswind_knts",
  "gust_spread_knts",
  "low_visibility_flag",
  "very_low_visibility_flag",
  "strong_wind_flag",
  "strong_crosswind_flag",
  "tailwind_flag",
  "adverse_weather_flag",
  "airport_risk_train",
  "runway_risk_train",
  "typecode_risk_train",
];

let latestDerived = null;
let deriveTimer = null;
let featureImportance = {};

document.getElementById("btn-sample").addEventListener("click", () => {
  for (const [key, val] of Object.entries(SAMPLE)) {
    const el = document.getElementById(key);
    if (el) el.value = val;
  }
  updateDerivedFeatures();
});

document.getElementById("btn-clear").addEventListener("click", () => {
  document.querySelectorAll(".field input, .field select").forEach(el => el.value = "");
  document.getElementById("result-panel").style.display = "none";
  document.getElementById("error-box").style.display = "none";
  updateDerivedFeatures();
});

function collectPayload() {
  const numFields  = ["glide_slope_angle","rwy_length",
                       "month","day_of_week","hour_utc","wind_speed_knts","wind_dir_deg",
                       "wind_gust_knts","visibility_m","temperature_deg","press_sea_level_p","press_p"];
  const strFields  = ["airport","runway","typecode","icaoaircrafttype","wtc","has_intersection",
                       "airport_country","airport_region","operator_country","operator_region",
                       "weather_intensity","weather_precipitation","weather_desc","weather_obscuration","weather_other"];

  const payload = {};
  for (const f of numFields) {
    const el = document.getElementById(f);
    if (el && el.value !== "") payload[f] = parseFloat(el.value);
  }
  for (const f of strFields) {
    const el = document.getElementById(f);
    if (el && el.value !== "") payload[f] = el.value;
  }
  return payload;
}

function formatDerivedValue(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  if (Math.abs(value) < 10 && value !== 0) return Number(value).toFixed(4);
  return Number(value).toFixed(2);
}

function renderDerivedFeatures(data) {
  const grid = document.getElementById("derived-grid");
  const values = data?.values || {};
  const metadata = data?.metadata || {};
  const orderedDerived = [...DERIVED_ORDER].sort((a, b) => {
    const aScore = Math.max(...(metadata[a]?.depends_on || []).map((dep) => featureImportance[dep]?.normalized_importance || 0));
    const bScore = Math.max(...(metadata[b]?.depends_on || []).map((dep) => featureImportance[dep]?.normalized_importance || 0));
    return bScore - aScore;
  });
  grid.innerHTML = orderedDerived.map((name) => {
    const meta = metadata[name] || { label: name, depends_on: [] };
    const dependencies = meta.depends_on.map((item) => `<code>${item}</code>`).join(", ");
    const linkedImportance = Math.max(
      0,
      ...(meta.depends_on || []).map((dep) => featureImportance[dep]?.normalized_importance || 0)
    );
    return `
      <div class="derived-item">
        <div class="derived-label">${meta.label}</div>
        <div class="derived-value">${formatDerivedValue(values[name])}</div>
        <div class="importance-meter"><span style="width:${Math.round(linkedImportance * 100)}%"></span></div>
        <div class="derived-deps">from ${dependencies || "model schema"}</div>
      </div>
    `;
  }).join("");
}

function formatImportance(value) {
  if (!value || value <= 0) return "low impact";
  return `${(value * 100).toFixed(0)}% relative impact`;
}

function applyFeatureImportance(data) {
  featureImportance = Object.fromEntries((data.features || []).map((item) => [item.feature, item]));
  const grid = document.getElementById("ranked-feature-grid");
  const fields = Array.from(grid.querySelectorAll(".field"));

  fields.forEach((field) => {
    const control = field.querySelector("input, select");
    if (!control) return;
    const info = featureImportance[control.id];
    field.dataset.importance = info ? String(info.normalized_importance) : "0";
    field.dataset.rank = info ? String(info.rank) : "999";

    const oldBadge = field.querySelector(".importance-badge");
    if (oldBadge) oldBadge.remove();

    const badge = document.createElement("div");
    badge.className = "importance-badge";
    if (info) {
      badge.innerHTML = `
        <span>#${info.rank}</span>
        <strong>${formatImportance(info.normalized_importance)}</strong>
        <div class="importance-meter"><span style="width:${Math.round(info.normalized_importance * 100)}%"></span></div>
      `;
    } else {
      badge.innerHTML = `
        <span>not used</span>
        <strong>not in final model</strong>
        <div class="importance-meter"><span style="width:0%"></span></div>
      `;
    }
    field.appendChild(badge);
  });

  fields
    .sort((a, b) => Number(a.dataset.rank) - Number(b.dataset.rank))
    .forEach((field) => grid.appendChild(field));

  const summary = document.getElementById("importance-summary");
  summary.innerHTML = `
    <strong>Best model:</strong> ${data.model_name} &middot;
    <strong>Feature set:</strong> ${data.feature_set} &middot;
    <strong>Method:</strong> permutation importance on ${data.metric}
  `;
}

function renderSensitivity(data) {
  const panel = document.getElementById("sensitivity-panel");
  const items = (data.items || []).slice(0, 8);
  if (items.length === 0) {
    panel.innerHTML = "";
    return;
  }

  panel.innerHTML = `
    <h3>Local What-If Sensitivity</h3>
    <div class="sensitivity-list">
      ${items.map((item) => {
        const info = featureImportance[item.feature];
        const rank = info ? `#${info.rank}` : "unranked";
        const plus = item.plus_delta >= 0 ? `+${item.plus_delta.toFixed(4)}` : item.plus_delta.toFixed(4);
        const minus = item.minus_delta >= 0 ? `+${item.minus_delta.toFixed(4)}` : item.minus_delta.toFixed(4);
        return `
          <div class="sensitivity-item">
            <strong>${rank} ${item.feature}</strong>
            <span>+${item.step}: ${plus}</span>
            <span>-${item.step}: ${minus}</span>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

async function updateSensitivity(payload) {
  const panel = document.getElementById("sensitivity-panel");
  panel.innerHTML = "";
  try {
    const resp = await fetch("/sensitivity", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || "Sensitivity unavailable.");
    renderSensitivity(data);
  } catch (err) {
    panel.innerHTML = `<div class="derived-error">Sensitivity unavailable: ${err.message}</div>`;
  }
}

async function loadFeatureImportance() {
  const summary = document.getElementById("importance-summary");
  try {
    const resp = await fetch("/feature-importance");
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || "Feature importance unavailable.");
    applyFeatureImportance(data);
    if (latestDerived) renderDerivedFeatures(latestDerived);
  } catch (err) {
    summary.textContent = `Feature importance unavailable: ${err.message}`;
  }
}

async function updateDerivedFeatures() {
  try {
    const resp = await fetch("/derived-features", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(collectPayload()),
    });
    const data = await resp.json();
    if (!resp.ok || data.error) throw new Error(data.error || "Derived feature calculation failed.");
    latestDerived = data;
    renderDerivedFeatures(data);
  } catch (err) {
    document.getElementById("derived-grid").innerHTML =
      `<div class="derived-error">Derived features unavailable: ${err.message}</div>`;
  }
}

function scheduleDerivedUpdate() {
  clearTimeout(deriveTimer);
  deriveTimer = setTimeout(updateDerivedFeatures, 250);
}

document.querySelectorAll(".field input, .field select").forEach((el) => {
  el.addEventListener("input", scheduleDerivedUpdate);
  el.addEventListener("change", scheduleDerivedUpdate);
});

document.getElementById("btn-predict").addEventListener("click", async () => {
  const resultPanel = document.getElementById("result-panel");
  const errorBox    = document.getElementById("error-box");
  const spinner     = document.getElementById("spinner");

  resultPanel.style.display = "none";
  errorBox.style.display    = "none";
  spinner.style.display     = "block";

  const payload = collectPayload();

  try {
    const resp = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await resp.json();
    spinner.style.display = "none";

    if (!resp.ok || data.error) {
      errorBox.textContent = data.error || "Prediction failed.";
      errorBox.style.display = "block";
      return;
    }
    if (data.derived_features) {
      latestDerived = data.derived_features;
      renderDerivedFeatures(data.derived_features);
    }

    const isGA = data.predicted_class === 1;
    const pGA  = (data.probability_go_around * 100).toFixed(1);
    const pNL  = (data.probability_normal_landing * 100).toFixed(1);

    document.getElementById("result-label").textContent = data.predicted_label;
    document.getElementById("result-label").className   = "result-label " + (isGA ? "label-ga" : "label-nl");
    document.getElementById("prob-ga-text").textContent  = `Go-Around: ${pGA}%`;
    document.getElementById("prob-nl-text").textContent  = `Normal Landing: ${pNL}%`;
    document.getElementById("bar-ga").style.width        = pGA + "%";
    document.getElementById("bar-nl").style.width        = pNL + "%";
    document.getElementById("result-meta").textContent   =
      `Threshold: ${data.threshold.toFixed(2)}  |  Probability: ${pGA}%`;
    updateSensitivity(payload);
    resultPanel.style.display = "block";
  } catch (err) {
    spinner.style.display = "none";
    errorBox.textContent = "Network error: " + err.message;
    errorBox.style.display = "block";
  }
});

loadFeatureImportance().finally(updateDerivedFeatures);

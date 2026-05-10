"use strict";

const SAMPLE = {
  airport: "EDDF", runway: "25L", wtc: "M", typecode: "A320",
  icaoaircrafttype: "L2J", airport_country: "DE", airport_region: "EU",
  operator_country: "DE", operator_region: "EU", has_intersection: "0",
  n_approaches: 1, n_rwy_approached: 1, glide_slope_angle: 3.0,
  rwy_length: 4000, month: 5, day_of_week: 2, hour_utc: 14,
  wind_speed_knts: 12, wind_dir_deg: 270, wind_gust_knts: 18,
  visibility_m: 8000, temperature_deg: 15, press_sea_level_p: 1013,
  press_p: 1010, weather_intensity: "", weather_precipitation: "",
  weather_desc: "", weather_obscuration: "", weather_other: "",
};

document.getElementById("btn-sample").addEventListener("click", () => {
  for (const [key, val] of Object.entries(SAMPLE)) {
    const el = document.getElementById(key);
    if (el) el.value = val;
  }
});

document.getElementById("btn-clear").addEventListener("click", () => {
  document.querySelectorAll(".field input, .field select").forEach(el => el.value = "");
  document.getElementById("result-panel").style.display = "none";
  document.getElementById("error-box").style.display = "none";
});

document.getElementById("btn-predict").addEventListener("click", async () => {
  const resultPanel = document.getElementById("result-panel");
  const errorBox    = document.getElementById("error-box");
  const spinner     = document.getElementById("spinner");

  resultPanel.style.display = "none";
  errorBox.style.display    = "none";
  spinner.style.display     = "block";

  const numFields  = ["n_approaches","n_rwy_approached","glide_slope_angle","rwy_length",
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
    resultPanel.style.display = "block";
  } catch (err) {
    spinner.style.display = "none";
    errorBox.textContent = "Network error: " + err.message;
    errorBox.style.display = "block";
  }
});

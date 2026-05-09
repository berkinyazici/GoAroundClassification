const form = document.getElementById("predict-form");
const output = document.getElementById("output");
const result = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  try {
    const payload = {
      features: JSON.parse(document.getElementById("features").value),
      model_name: document.getElementById("model_name").value,
    };

    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }

    result.textContent = JSON.stringify(data, null, 2);
    output.classList.remove("hidden");
  } catch (error) {
    result.textContent = String(error);
    output.classList.remove("hidden");
  }
});

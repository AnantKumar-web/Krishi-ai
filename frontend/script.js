```javascript
const form = document.getElementById("predictionForm");

form.addEventListener("submit", async function(e) {

    e.preventDefault();

    const data = {
        crop_type: document.getElementById("crop_type").value,
        soil_type: parseInt(document.getElementById("soil_type").value),
        soil_moisture: parseFloat(document.getElementById("soil_moisture").value),
        temperature: parseFloat(document.getElementById("temperature").value),
        rainfall: parseFloat(document.getElementById("rainfall").value),
        crop_stage: parseInt(document.getElementById("crop_stage").value),
        fertilizer: parseFloat(document.getElementById("fertilizer").value),
        humidity: parseFloat(document.getElementById("humidity").value)
    };

    console.log("Sending Data:", data);

    const resultBox = document.getElementById("result");

    resultBox.innerHTML = "Predicting...";

    try {

        const response = await fetch("https://krishi-ai-backend-hva2.onrender.com/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        console.log("Backend Response:", result);

        resultBox.innerHTML = `
        <h3>Prediction Result</h3>
        <p><b>Disease Probability (%):</b> ${result.disease_probability_percent}</p>
        <p><b>Predicted Yield (tons/acre):</b> ${result.predicted_yield_tons_per_acre}</p>
        <p><b>KRI:</b> ${result.KRI}</p>
        <p><b>Risk Level:</b> ${result.risk_level}</p>
        <p><b>Decision:</b> ${result.decision}</p>
        `;

    } catch (error) {

        console.error("Error:", error);

        resultBox.innerHTML = `
        <p style="color:red;">Error connecting to backend.</p>
        `;

    }

});
```

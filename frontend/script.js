document.getElementById("predictionForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const data = {
        soil_type: 1,
        soil_moisture: parseFloat(document.getElementById("soil_moisture").value),
        temperature: parseFloat(document.getElementById("temperature").value),
        rainfall: parseFloat(document.getElementById("rainfall").value),
        crop_stage: 0,
        fertilizer: parseFloat(document.getElementById("fertilizer").value),
        humidity: parseFloat(document.getElementById("humidity").value)
    };

    const response = await fetch("https://krishi-ai-backend-hva2.onrender.com/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    document.getElementById("result").innerHTML = `
        <h3>Prediction Result</h3>
        <p><b>Disease Score:</b> ${result.disease_score}</p>
        <p><b>Yield:</b> ${result.predicted_yield}</p>
        <p><b>KRI:</b> ${result.KRI}</p>
        <p><b>Decision:</b> ${result.decision}</p>
    `;
});
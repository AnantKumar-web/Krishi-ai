document.getElementById("predictionForm").addEventListener("submit", async function(e){

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

try{

const response = await fetch("https://krishi-ai-backend-hva2.onrender.com/predict",{
method:"POST",
headers:{
"Content-Type":"application/json"
},
body:JSON.stringify(data)
});

const result = await response.json();

console.log("Backend Response:", result);

resultBox.innerHTML = `
<div class="result-card"><b>Disease Probability:</b> ${result.disease_probability_percent}%</div>
<div class="result-card"><b>Predicted Yield:</b> ${result.predicted_yield_tons_per_acre} tons/acre</div>
<div class="result-card"><b>KRI Score:</b> ${result.KRI}</div>
<div class="result-card"><b>Risk Level:</b> ${result.risk_level}</div>
<div class="result-card"><b>Recommendation:</b> ${result.decision}</div>
`;

}catch(error){

console.error(error);
resultBox.innerHTML = "Error connecting to AI backend.";

}

});

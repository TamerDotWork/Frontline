const ws = new WebSocket(`ws://${window.location.host}/ws`);
const statusContainer = document.getElementById("tag");
const sc = statusContainer.querySelector(".status-text");
const logsDiv = document.getElementById("logs");
const core_div = document.getElementById("core");

ws.onopen = () => {
    console.log("WebSocket connected");
    sc.textContent = "WebSocket connected!";
    statusContainer.classList.add("success");
};

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    // Unified log structure now
    const log = data; // everything is in log_entry now

    // Update total tokens (from usageMetadata)
    const div_total = document.getElementById("total");
    const usage = log.raw_data?.usageMetadata || {};
    div_total.textContent = usage.totalTokenCount || "-";

    // Update prompt/response
    const t_div = document.getElementById("t");
    t_div.innerHTML = `<div>${log.prompt}${log.success ? "<img src='static/success.svg' />" : ""}</div>`;

    const l_div = document.getElementById("l");
    l_div.textContent = log.latency_ms + "ms";

    const sr_div = document.getElementById("sr");
    sr_div.textContent = log.success_percent + "%";

    const int_div = document.getElementById("int");
    int_div.textContent = usage.promptTokenCount || usage.inputTokens || "-";

    const outt_div = document.getElementById("outt");
    outt_div.textContent = usage.candidatesTokenCount || usage.outputTokens || "-";

    const mv_div = document.getElementById("mv");
    mv_div.textContent = log.raw_data?.modelVersion || "-";

    const rid_div = document.getElementById("rid");
    rid_div.textContent = log.raw_data?.responseId || "-";

    const provider_div = document.getElementById("provider");
    provider_div.textContent = log.provider || "-";

 // Example: total_cost from backend
    const totalCost = data.total_cost || 0;

    // Format total cost
    const totalCostDiv = document.getElementById("tc");
    if (totalCost < 1) {
        totalCostDiv.textContent = "Less than $1";
    } else {
        totalCostDiv.textContent = "$" + totalCost.toFixed(2);
    }

    core_div.classList.add("react");
    console.log(log);

    // Update pie chart
    const successRate = log.success_percent || 0;
    const failureRate = 100 - successRate;
    successChart.data.datasets[0].data = [successRate, failureRate];
    successChart.update();

    // Auto-scroll
    logsDiv.scrollTop = logsDiv.scrollHeight;

    // Remove animation class after 3 seconds
    setTimeout(() => core_div.classList.remove("react"), 3000);
};

ws.onclose = () => {
    sc.textContent = "WebSocket disconnected!";
    statusContainer.classList.remove("success", "failure");
};

ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    sc.textContent = "WebSocket error!";
    statusContainer.classList.add("failure");
};

/* Charts setup */
const ctx = document.getElementById("successChart").getContext("2d");
const trackColor = "#edf1f4"; // background track

// Create gradients
const successGradient = ctx.createLinearGradient(0, 0, 0, 180);
successGradient.addColorStop(0, "#5FD49E");
successGradient.addColorStop(1, "#29C67D");

const failGradient = ctx.createLinearGradient(0, 0, 0, 180);
failGradient.addColorStop(0, "#FF9077");
failGradient.addColorStop(1, "#FF6340");

let successChart = new Chart(ctx, {
    type: "doughnut",
    data: {
        datasets: [{
            data: [0, 100],
            backgroundColor: [successGradient, trackColor],
            borderWidth: 0,
            hoverOffset: 0
        }]
    },
    options: {
        responsive: true,
        cutout: "72%",
        rotation: -90,
        circumference: 360,
        plugins: {
            legend: { display: false },
            tooltip: { enabled: false }
        }
    }
});

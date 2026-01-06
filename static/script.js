
    const ws = new WebSocket(`ws://${window.location.host}/ws`);
    const statusContainer = document.getElementById("tag");
    const sc = statusContainer.querySelector(".status-text");

    const logsDiv = document.getElementById("logs");

    ws.onopen = () => {
    console.log("WebSocket connected");
    sc.textContent = "WebSocket connected!";
    statusContainer.classList.add("success");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);

      // Create a new log entry
      const div_total = document.getElementById("total");
      const div = document.createElement("div");
      div.className = "log-entry";

      const prompt = document.createElement("p");
      prompt.innerHTML = `<strong>Prompt:</strong> ${data.prompt}`;
      // div.appendChild(prompt);

      const response = document.createElement("p");
      response.innerHTML = `<strong>Response:</strong> ${data.response}`;
      // div.appendChild(response);

      if (data.meta) {
        const meta = document.createElement("p");
        meta.className = "meta";
        meta.innerHTML = `
          <strong>Provider:</strong> ${data.meta.provider || "-"} |
          <strong>Status:</strong> ${data.meta.status_code || "-"} |
          <strong>Latency:</strong> ${data.meta.latency_ms || "-"} ms |
          <strong>Request Bytes:</strong> ${data.meta.request_bytes || "-"} |
          <strong>Response Bytes:</strong> ${data.meta.response_bytes || "-"}
        `;
        // div.appendChild(meta);
    
       div_total.textContent =  data.data.usageMetadata.totalTokenCount;
        // console.log(data.data.usageMetadata.totalTokenCount);
      
      }

      // Add to logs container
      // logsDiv.appendChild(div);

      // Auto-scroll
      logsDiv.scrollTop = logsDiv.scrollHeight;
    };

    ws.onclose = () => {
      console.log("WebSocket disconnected");
 
 
     sc.textContent = "WebSocket disconnected!";
     statusContainer.classList.remove("success");
     statusContainer.classList.remove("failure");
    };

    ws.onerror = (err) => {
    console.error("WebSocket error:", err);
    sc.textContent = "WebSocket error!";
    statusContainer.classList.add("failure");
    };
 
 
document.addEventListener("DOMContentLoaded", () => {
    const svgs = document.querySelectorAll('.core div svg');
    const delay = 0.15; // seconds

    svgs.forEach((svg, index) => {
        svg.style.animationDelay = `${index * delay}s`;
        svg.style.animationIterationCount = 'infinite'; // loop forever
    });
});
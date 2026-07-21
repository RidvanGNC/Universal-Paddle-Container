let ENDPOINTS = [];

function buildFieldInput(field) {
    const wrapper = document.createElement("div");
    const label = document.createElement("label");
    label.textContent = field.name;
    wrapper.appendChild(label);

    let input;
    if (field.kind === "checkbox") {
        input = document.createElement("input");
        input.type = "checkbox";
        input.checked = !!field.default;
    } else if (field.kind === "select") {
        input = document.createElement("select");
        for (const option of field.options || []) {
            const opt = document.createElement("option");
            opt.value = option;
            opt.textContent = option;
            if (option === field.default) opt.selected = true;
            input.appendChild(opt);
        }
    } else {
        input = document.createElement("input");
        input.type = "number";
        input.step = "any";
        input.value = field.default ?? "";
    }
    input.dataset.field = field.name;
    input.dataset.kind = field.kind;
    wrapper.appendChild(input);
    return wrapper;
}

function renderExtraFields(endpoint) {
    const container = document.getElementById("extra-fields");
    container.innerHTML = "";
    for (const field of endpoint.extra_fields) {
        container.appendChild(buildFieldInput(field));
    }
}

function currentEndpoint() {
    const select = document.getElementById("capability-select");
    return ENDPOINTS.find((e) => e.capability_name === select.value);
}

async function onRunClick() {
    const endpoint = currentEndpoint();
    const fileInput = document.getElementById("file-input");
    const jsonPanel = document.getElementById("result-json");
    const imagePanel = document.getElementById("result-image");
    const timingPanel = document.getElementById("result-timing");

    if (!endpoint) return;
    if (!fileInput.files.length) {
        alert("Choose a test image first.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);
    for (const input of document.querySelectorAll("#extra-fields [data-field]")) {
        const value = input.dataset.kind === "checkbox" ? input.checked : input.value;
        formData.append(input.dataset.field, value);
    }

    const runBtn = document.getElementById("run-btn");
    runBtn.disabled = true;
    runBtn.textContent = "Running...";
    timingPanel.textContent = "";
    jsonPanel.hidden = true;
    imagePanel.hidden = true;

    try {
        const resp = await fetch(endpoint.path, { method: endpoint.method, body: formData });
        if (endpoint.response_kind === "image") {
            if (!resp.ok) throw new Error(`${resp.status}: ${await resp.text()}`);
            const blob = await resp.blob();
            imagePanel.src = URL.createObjectURL(blob);
            imagePanel.hidden = false;
            timingPanel.textContent = `Processing time: ${resp.headers.get("X-Processing-Time-Ms") || "?"} ms, device: ${resp.headers.get("X-Device-Used") || "?"}`;
        } else {
            const body = await resp.json();
            if (!resp.ok) throw new Error(`${resp.status}: ${body.detail || JSON.stringify(body)}`);
            jsonPanel.textContent = JSON.stringify(body, null, 2);
            jsonPanel.hidden = false;
            if (body.processing_time_ms !== undefined) {
                timingPanel.textContent = `Processing time: ${body.processing_time_ms.toFixed(1)} ms, device: ${body.device_used || "?"}`;
            }
        }
    } catch (err) {
        jsonPanel.textContent = `Error: ${err.message}`;
        jsonPanel.hidden = false;
    } finally {
        runBtn.disabled = false;
        runBtn.textContent = "Run";
    }
}

async function init() {
    ENDPOINTS = await fetchJSON("/ui/playground-config");
    const select = document.getElementById("capability-select");
    for (const endpoint of ENDPOINTS) {
        const opt = document.createElement("option");
        opt.value = endpoint.capability_name;
        opt.textContent = endpoint.label;
        select.appendChild(opt);
    }
    select.addEventListener("change", () => renderExtraFields(currentEndpoint()));
    renderExtraFields(ENDPOINTS[0]);
    document.getElementById("run-btn").addEventListener("click", onRunClick);
}

init();

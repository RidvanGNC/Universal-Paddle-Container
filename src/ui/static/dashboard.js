function isDirField(fieldName) {
    return fieldName.endsWith("_dir_name");
}

async function populateDirectoryDropdown(select, capabilityName) {
    const currentValue = select.dataset.currentValue || "";
    let directories = [];
    try {
        const data = await fetchJSON(`/admin/capabilities/${capabilityName}/available-models`);
        directories = data.directories;
    } catch (err) {
        console.error("Failed to list available models", err);
    }
    if (currentValue && !directories.includes(currentValue)) {
        directories = [currentValue, ...directories];
    }
    select.innerHTML = "";
    const emptyOption = document.createElement("option");
    emptyOption.value = "";
    emptyOption.textContent = "(none)";
    select.appendChild(emptyOption);
    for (const dir of directories) {
        const option = document.createElement("option");
        option.value = dir;
        option.textContent = dir;
        if (dir === currentValue) option.selected = true;
        select.appendChild(option);
    }
}

function upgradeDirInputsToSelects(row, capabilityName) {
    const inputs = row.querySelectorAll(".config-input");
    for (const input of inputs) {
        const fieldName = input.dataset.field;
        if (!isDirField(fieldName)) continue;
        const select = document.createElement("select");
        select.className = "config-input";
        select.dataset.field = fieldName;
        select.dataset.currentValue = input.value;
        input.replaceWith(select);
        populateDirectoryDropdown(select, capabilityName);
    }
}

function collectConfig(row) {
    const config = {};
    for (const input of row.querySelectorAll(".config-input")) {
        config[input.dataset.field] = input.value;
    }
    return config;
}

function updateRowResult(row, result) {
    const badge = row.querySelector(".loaded-cell .badge");
    badge.textContent = result.loaded ? "loaded" : "not loaded";
    badge.className = "badge " + (result.loaded ? "ok" : "off");

    const problemsCell = row.querySelector(".problems-cell");
    problemsCell.innerHTML = "";
    if (result.problems && result.problems.length) {
        const ul = document.createElement("ul");
        for (const problem of result.problems) {
            const li = document.createElement("li");
            li.textContent = problem;
            ul.appendChild(li);
        }
        problemsCell.appendChild(ul);
    }
}

async function onReloadClick(row) {
    const capabilityName = row.dataset.capability;
    const config = collectConfig(row);
    const button = row.querySelector(".reload-btn");
    button.disabled = true;
    button.textContent = "Reloading...";
    try {
        const result = await fetchJSON(`/admin/capabilities/${capabilityName}/reload`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ config }),
        });
        updateRowResult(row, result);
    } catch (err) {
        alert(`Reload failed: ${err.message}`);
    } finally {
        button.disabled = false;
        button.textContent = "Reload";
    }
}

document.querySelectorAll("#capabilities-table tbody tr").forEach((row) => {
    upgradeDirInputsToSelects(row, row.dataset.capability);
    row.querySelector(".reload-btn").addEventListener("click", () => onReloadClick(row));
});

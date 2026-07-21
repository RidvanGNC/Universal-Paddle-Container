async function fetchJSON(url, options) {
    const resp = await fetch(url, options);
    const contentType = resp.headers.get("content-type") || "";
    const body = contentType.includes("application/json") ? await resp.json() : await resp.text();
    if (!resp.ok) {
        const detail = typeof body === "object" && body !== null ? (body.detail || JSON.stringify(body)) : body;
        throw new Error(`${resp.status}: ${detail}`);
    }
    return body;
}

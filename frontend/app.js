let sessionId = null;

const fileInput = document.getElementById("file");
const sheetNameInput = document.getElementById("sheetName");
const activeCellInput = document.getElementById("activeCell");
const queryInput = document.getElementById("query");
const output = document.getElementById("output");
const runButton = document.getElementById("runQuery");
const createButton = document.getElementById("createSession");

function setOutput(value) {
  output.textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

createButton.addEventListener("click", async () => {
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setOutput("Choose an Excel file first.");
    return;
  }
  const form = new FormData();
  form.append("file", file);
  if (sheetNameInput.value) form.append("sheet_name", sheetNameInput.value);
  if (activeCellInput.value) form.append("active_cell", activeCellInput.value);
  const res = await fetch("/excel/session", { method: "POST", body: form });
  const data = await res.json();
  sessionId = data.session_id;
  runButton.disabled = !sessionId;
  setOutput(data);
});

runButton.addEventListener("click", async () => {
  if (!sessionId) {
    setOutput("Create a session first.");
    return;
  }
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("text", queryInput.value);
  const res = await fetch("/excel/query", { method: "POST", body: form });
  const data = await res.json();
  setOutput(data);
});


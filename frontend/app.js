const form = document.querySelector("#question-form");
const questionInput = document.querySelector("#question");
const status = document.querySelector("#status");
const answer = document.querySelector("#answer");
const submitButton = form.querySelector('button[type="submit"]');

document.querySelectorAll(".suggestion").forEach((suggestion) => {
  suggestion.addEventListener("click", () => {
    questionInput.value = suggestion.dataset.question;
    form.requestSubmit();
  });
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  status.classList.remove("status--error");
  const thorough = (document.querySelector('input[name="hamur"]:checked') || {}).value === "itarlegt";
  status.textContent = thorough ? "Hugsa um svarið… (getur tekið allt að mínútu)" : "Leita að svari…";
  answer.replaceChildren();
  submitButton.disabled = true;

  try {
    // Served by app.py: same server. Opened as a file: the default local server.
    const apiUrl = location.protocol.startsWith("http")
      ? "/spyrja_px_api"
      : "http://127.0.0.1:8000/spyrja_px_api";
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        spurning: questionInput.value,
        hamur: (document.querySelector('input[name="hamur"]:checked') || {}).value || "venjulegt"
      })
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const data = await response.json();
    status.textContent = "";
    const heading = document.createElement("h2");
    heading.className = "answer-heading";
    heading.textContent = "Niðurstaða";
    answer.append(heading);

    // Several questions in one prompt come back as a list in "svör".
    const results = data["svör"] || [data];
    results.forEach((result) => renderResult(result, results.length > 1));
  } catch (error) {
    status.classList.add("status--error");
    status.textContent = `Ekki tókst að sækja svar: ${error.message}`;
  } finally {
    submitButton.disabled = false;
  }
});
function renderResult(data, showQuestion) {
  if (showQuestion && data.spurning) {
    const question = document.createElement("h3");
    question.className = "answer-question";
    question.textContent = data.spurning;
    answer.append(question);
  }

  const answerText = document.createElement("p");
  answerText.className = "answer-text";
  // The link to the table is shown below, so drop raw URLs (also "[url](url)") from the text.
  answerText.textContent = data.svar
    .replace(/\s*\[https?:\/\/[^\]]*\](\(https?:\/\/[^)]*\))?/g, "")
    .replace(/[,;]?\s*\(?https?:\/\/\S+?\)?(?=\s|$)/g, "")
    .trim();
  answerText.style.whiteSpace = "pre-line";
  answer.append(answerText);

  if (data.tafla && data.tafla.title) {
    answer.append(renderDetails(data));
  }

  if (data["töflu_slóð"]) {
    const source = document.createElement("a");
    source.className = "source-link";
    source.href = data["töflu_slóð"];
    source.textContent = "Skoða töflu hjá Hagstofunni";
    source.target = "_blank";
    source.rel = "noopener";
    answer.append(source);
  }
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// The table, chosen dimensions, notes and data behind an answer, in a collapsible box.
function renderDetails(data) {
  const details = element("details", "answer-details");
  details.open = true;
  details.append(element("summary", "", `Gögn úr töflu ${data.tafla.id.replace(".px", "")}: ${data.tafla.title}`));

  (data.athugasemdir || []).forEach((note) => details.append(element("p", "data-note", note)));

  const table = data.gagnatafla;
  if (table && table.rows && table.rows.length) {
    if (table.fixed && table.fixed.length) {
      const chips = element("div", "data-chips");
      table.fixed.forEach(([name, value]) => {
        const chip = element("span", "data-chip");
        chip.append(element("span", "data-chip-name", name), element("span", "data-chip-value", value));
        chips.append(chip);
      });
      details.append(chips);
    }

    const wrapper = element("div", "data-table-wrapper");
    const htmlTable = element("table", "data-table");
    const headRow = element("tr");
    table.columns.forEach((name) => headRow.append(element("th", "", name)));
    htmlTable.append(element("thead"));
    htmlTable.tHead.append(headRow);
    const body = element("tbody");
    table.rows.forEach((row) => {
      const tr = element("tr");
      row.forEach((value) => {
        // Numbers are right-aligned.
        const isNumber = /^-?[\d.]+(,\d+)?$/.test(value);
        tr.append(element("td", isNumber ? "number" : "", value));
      });
      body.append(tr);
    });
    // Sums computed by the server (a whole year of a monthly table, an age range).
    (table.summary || []).forEach(([label, value]) => {
      const tr = element("tr", "sum-row");
      const name = element("td", "", `Samtals: ${label}`);
      name.colSpan = Math.max(1, table.columns.length - 1);
      tr.append(name, element("td", "number", value));
      body.append(tr);
    });
    htmlTable.append(body);
    wrapper.append(htmlTable);
    details.append(wrapper);
  }

  if (data["tími"] && data["tími"].total) {
    details.append(element("p", "data-time", `Svarað á ${data["tími"].total.toFixed(1)} s`));
  }
  return details;
}

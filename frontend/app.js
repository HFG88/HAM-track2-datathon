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
  status.textContent = "Leita að svari…";
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
      body: JSON.stringify({ spurning: questionInput.value })
    });

    if (!response.ok) {
      throw new Error(`Server returned ${response.status}`);
    }

    const data = await response.json();
    status.textContent = "";
    const heading = document.createElement("h2");
    heading.className = "answer-heading";
    heading.textContent = "Niðurstaða";
    const answerText = document.createElement("p");
    answerText.className = "answer-text";
    answerText.textContent = data.svar;
    answerText.style.whiteSpace = "pre-line";
    answer.append(heading, answerText);

    if (data.tafla && data.tafla.title) {
      const details = document.createElement("details");
      details.className = "answer-details";
      const summary = document.createElement("summary");
      summary.textContent = `Tafla: ${data.tafla.title} (${data.tafla.id})`;
      details.append(summary);

      const lines = [];
      if (data["víddir"]) lines.push(`Valdar víddir: ${data["víddir"]}`);
      (data.athugasemdir || []).forEach((note) => lines.push(`Athugasemd: ${note}`));
      (data["gögn"] || []).forEach((row) => lines.push(row));
      if (data["tími"]) lines.push(`Tími: ${data["tími"].total ?? "?"} s`);
      const pre = document.createElement("pre");
      pre.textContent = lines.join("\n");
      details.append(pre);
      answer.append(details);
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
  } catch (error) {
    status.classList.add("status--error");
    status.textContent = `Ekki tókst að sækja svar: ${error.message}`;
  } finally {
    submitButton.disabled = false;
  }
});
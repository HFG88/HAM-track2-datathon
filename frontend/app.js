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
    const response = await fetch("http://127.0.0.1:8000/spyrja_px_api", {
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
    answer.append(heading, answerText);

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
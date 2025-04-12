const successIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" class="bi bi-check-lg" viewBox="0 0 16 16">
            <path d="M13.485 1.85a.5.5 0 0 1 1.065.02.75.75 0 0 1-.02 1.065L5.82 12.78a.75.75 0 0 1-1.106.02L1.476 9.346a.75.75 0 1 1 1.05-1.07l2.74 2.742L12.44 2.92a.75.75 0 0 1 1.045-.07z"/>
        </svg>`;
const copyIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" class="bi bi-clipboard" viewBox="0 0 16 16">
                <path d="M10 1.5a.5.5 0 0 1 .5-.5h2a2 2 0 0 1 2 2v12a2 2 0 0 1-2 2h-9a2 2 0 0 1-2-2V3a2 2 0 0 1 2-2h2a.5.5 0 0 1 .5.5V3h3V1.5zM6.5 3V2h3v1h-3zm4 0v1h2a1 1 0 0 0-1-1h-2V3zm-5 0H3a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1V4a1 1 0 0 0-1-1H5.5V3z"/>
            </svg>`;

// Function to change icons after copying
const changeIcon = (button) => {
  button.innerHTML = successIcon;
  setTimeout(() => {
    button.innerHTML = copyIcon; // Reset to copy icon
  }, 2000);
};

// Function to get code text from tables, skipping line numbers
const getCodeFromTable = (codeBlock) => {
  return [...codeBlock.querySelectorAll("tr")]
    .map((row) => row.querySelector("td:last-child")?.innerText ?? "")
    .join("");
};

// Function to get code text from non-table blocks
const getNonTableCode = (codeBlock) => {
  return codeBlock.textContent.trim();
};

document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("pre code").forEach((codeBlock) => {
    const pre = codeBlock.parentNode;
    pre.style.position = "relative"; // Ensure parent `pre` can contain absolute elements

    // Create and append the copy button
    const copyBtn = document.createElement("button");
    copyBtn.className = "clipboard-button";
    copyBtn.innerHTML = copyIcon;
    copyBtn.setAttribute("aria-label", "Copy code to clipboard");
    pre.appendChild(copyBtn);

    // Attach event listener to copy button
    copyBtn.addEventListener("click", async () => {
      const isTable = codeBlock.querySelector("table");
      const codeToCopy = isTable
        ? getCodeFromTable(codeBlock)
        : getNonTableCode(codeBlock);
      try {
        await navigator.clipboard.writeText(codeToCopy);
        changeIcon(copyBtn); // Show success icon
      } catch (error) {
        console.error("Failed to copy text: ", error);
        // No icon change on error - just log to console
      }
    });

    // Add language label if needed
    const langClass = codeBlock.className.match(/language-(\w+)/);
    if (langClass) {
      const lang = langClass[1];
      const label = document.createElement("span");
      label.className = "code-label label-" + lang;
      label.textContent = lang.toUpperCase();
      pre.appendChild(label);
    }

    // Only add scroll handler if necessary for horizontally scrollable code blocks
    if (pre.scrollWidth > pre.clientWidth) {
      pre.addEventListener("scroll", () => {
        copyBtn.style.right = `${5 - pre.scrollLeft}px`;
        const label = pre.querySelector(".code-label");
        if (label) {
          label.style.right = `-${pre.scrollLeft}px`;
        }
      });
    }
  });
});

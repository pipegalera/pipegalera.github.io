document.addEventListener("DOMContentLoaded", function () {
  // SVG icons for copy button states
  const copyIcon = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>`;
  const checkIcon = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>`;

  // Find all code blocks
  const codeBlocks = document.querySelectorAll("pre code");

  codeBlocks.forEach(function (codeBlock) {
    // Skip ASCII art blocks
    if (codeBlock.classList.contains("language-ascii-art")) {
      return;
    }

    const pre = codeBlock.parentNode;

    // Make sure pre has position: relative for proper button positioning
    if (getComputedStyle(pre).position !== "relative") {
      pre.style.position = "relative";
    }

    // Create copy button
    const copyButton = document.createElement("button");
    copyButton.className = "copy-button";
    copyButton.innerHTML = copyIcon;
    copyButton.setAttribute("aria-label", "Copy code");
    copyButton.setAttribute("title", "Copy code");

    // Add button to pre element
    pre.appendChild(copyButton);

    // Add click event
    copyButton.addEventListener("click", function () {
      // Get the text to copy
      let codeToCopy = "";

      // Check if code has a table (line numbers)
      const codeTable = codeBlock.querySelector("table");
      if (codeTable) {
        // Get text from table rows, skipping line numbers
        const rows = codeTable.querySelectorAll("tr");
        codeToCopy = Array.from(rows)
          .map((row) => {
            const codeCells = row.querySelectorAll("td");
            // Skip the first cell (line number) and get text from second cell
            return codeCells.length > 1 ? codeCells[1].textContent : "";
          })
          .join("\n");
      } else {
        // Get text directly from code block
        codeToCopy = codeBlock.textContent;
      }

      // Copy to clipboard
      navigator.clipboard
        .writeText(codeToCopy)
        .then(function () {
          // Success - show check mark
          copyButton.innerHTML = checkIcon;
          copyButton.classList.add("copied");

          // Reset button after 2 seconds
          setTimeout(function () {
            copyButton.innerHTML = copyIcon;
            copyButton.classList.remove("copied");
          }, 2000);
        })
        .catch(function (err) {
          console.error("Failed to copy code: ", err);
        });
    });

    // Handle scrolling
    pre.addEventListener("scroll", function () {
      // Keep the button in the top-right corner when scrolling horizontally
      copyButton.style.right = `${5 - pre.scrollLeft}px`;
    });
  });
});

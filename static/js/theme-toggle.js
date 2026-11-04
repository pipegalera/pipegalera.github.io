// Light/dark theme toggle.
//
// With no choice stored, the site follows the system setting. Clicking the
// toggle switches theme and remembers the choice in localStorage. If the choice
// matches the system setting, it is forgotten, so the site follows the system again.
// The stored choice is applied before render by a small inline script in header.html.
(function () {
    var root = document.documentElement;
    var system = window.matchMedia("(prefers-color-scheme: dark)");
    var button = document.getElementById("theme-toggle");
    var themeColors = { light: "#fafafa", dark: "#151515" };

    function systemTheme() {
        return system.matches ? "dark" : "light";
    }

    function currentTheme() {
        return root.getAttribute("data-theme") || systemTheme();
    }

    function render() {
        var theme = currentTheme();
        var dark = theme === "dark";
        if (button) {
            button.setAttribute("aria-pressed", String(dark));
            button.title = dark ? "Switch to light theme" : "Switch to dark theme";
        }
        // Keep the browser UI color (mobile address bar) in step with the page
        document.querySelectorAll('meta[name="theme-color"]').forEach(function (meta) {
            meta.setAttribute("content", themeColors[theme]);
        });
    }

    function choose(theme) {
        var followSystem = theme === systemTheme();
        try {
            if (followSystem) localStorage.removeItem("theme");
            else localStorage.setItem("theme", theme);
        } catch (e) {}
        if (followSystem) root.removeAttribute("data-theme");
        else root.setAttribute("data-theme", theme);
        render();
    }

    if (button) {
        button.hidden = false;
        button.addEventListener("click", function () {
            choose(currentTheme() === "dark" ? "light" : "dark");
        });
    }

    system.addEventListener("change", render);
    render();
})();

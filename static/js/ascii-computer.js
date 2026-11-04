// Animates the ASCII computer on the home page: the screen types a short
// command and prints a reply, and a key on the keyboard dips with each letter.
// Everything else in the drawing stays still.
//
// The drawing itself lives in content/_index.md as a ```ascii-art block, which
// is what visitors see without JavaScript or with "reduce motion" turned on.
(function () {
    var code = document.querySelector("pre > code.language-ascii-art");
    if (!code) return;

    var pre = code.parentElement;
    pre.setAttribute("role", "img");
    pre.setAttribute("aria-label", "ASCII drawing of an old computer typing on its screen");

    var original = code.textContent;
    var lines = original.split("\n");

    // Find the screen (the two rows above its bottom edge) and the keyboard keys.
    // If the drawing has been changed and these are not found, leave it static.
    var screenEdge = "|_________|";
    var screenBottom = lines.findIndex(function (l) { return l.indexOf(screenEdge) !== -1; });
    var keyRow = lines.findIndex(function (l) { return l.indexOf(":::") !== -1; });
    if (screenBottom < 2 || keyRow < 0) return;

    var screenCol = lines[screenBottom].indexOf(screenEdge) + 1;
    var screenWidth = screenEdge.length - 2;
    var keyCols = [];
    for (var i = 0; i < lines[keyRow].length; i++) {
        if (lines[keyRow][i] === ":") keyCols.push(i);
    }

    // Command and reply pairs. Commands must fit in 6 characters, replies in 8.
    var sessions = [
        ["whoami", "pipe"],
        ["uptime", "24/7"],
        ["ping", "pong"],
    ];

    function put(line, col, text) {
        return line.slice(0, col) + text + line.slice(col + text.length);
    }

    function fit(text) {
        return (" " + text + " ".repeat(screenWidth)).slice(0, screenWidth);
    }

    function draw(top, bottom, key) {
        var frame = lines.slice();
        frame[screenBottom - 2] = put(frame[screenBottom - 2], screenCol, fit(top));
        frame[screenBottom - 1] = put(frame[screenBottom - 1], screenCol, fit(bottom));
        if (key !== null) frame[keyRow] = put(frame[keyRow], keyCols[key], ".");
        code.textContent = frame.join("\n");
    }

    // Each step: [screen top row, screen bottom row, pressed key or null, milliseconds to show]
    function script() {
        var steps = [];
        sessions.forEach(function (session) {
            var command = session[0];
            var reply = session[1];
            for (var blink = 0; blink < 2; blink++) {
                steps.push([">_", "", null, 530]);
                steps.push([">", "", null, 530]);
            }
            for (var n = 1; n <= command.length; n++) {
                var key = Math.floor(Math.random() * keyCols.length);
                steps.push([">" + command.slice(0, n) + "_", "", key, 90 + Math.random() * 140]);
            }
            steps.push([">" + command, "", null, 350]);
            steps.push([">" + command, reply, null, 2200]);
        });
        return steps;
    }

    var reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    var steps = [];
    var index = 0;
    var timer = null;

    function tick() {
        if (index >= steps.length) {
            steps = script();
            index = 0;
        }
        var step = steps[index++];
        draw(step[0], step[1], step[2]);
        timer = setTimeout(tick, step[3]);
    }

    function start() {
        if (timer || reduceMotion.matches || document.hidden) return;
        tick();
    }

    function stop() {
        clearTimeout(timer);
        timer = null;
    }

    // Pause while the tab is hidden; stop and restore the drawing if reduce motion is turned on.
    document.addEventListener("visibilitychange", function () {
        if (document.hidden) stop();
        else start();
    });
    reduceMotion.addEventListener("change", function () {
        if (reduceMotion.matches) {
            stop();
            code.textContent = original;
        } else {
            start();
        }
    });

    start();
})();

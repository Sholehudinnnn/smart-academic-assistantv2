(() => {
    "use strict";

    const field = document.getElementById("particleField");
    const page = document.querySelector(".welcome-page");
    const cta = document.getElementById("welcomeCta");
    const reducedMotion = window.matchMedia(
        "(prefers-reduced-motion: reduce)"
    ).matches;

    if (field && !reducedMotion) {
        const fragment = document.createDocumentFragment();
        const particleCount = window.innerWidth < 700 ? 20 : 34;

        for (let index = 0; index < particleCount; index += 1) {
            const particle = document.createElement("span");
            const size = 2 + Math.random() * 4;

            particle.className = "particle";
            particle.style.left = `${Math.random() * 100}%`;
            particle.style.setProperty("--size", `${size}px`);
            particle.style.setProperty(
                "--duration",
                `${12 + Math.random() * 17}s`
            );
            particle.style.setProperty(
                "--delay",
                `${-Math.random() * 28}s`
            );
            particle.style.setProperty(
                "--drift",
                `${Math.random() * 90 - 45}px`
            );
            particle.style.setProperty(
                "--opacity",
                `${0.12 + Math.random() * 0.2}`
            );
            fragment.appendChild(particle);
        }

        field.appendChild(fragment);
    }

    if (page && !reducedMotion && window.matchMedia("(pointer: fine)").matches) {
        let frameRequested = false;
        let pointerX = 50;
        let pointerY = 42;

        page.addEventListener("pointermove", event => {
            pointerX = (event.clientX / window.innerWidth) * 100;
            pointerY = (event.clientY / window.innerHeight) * 100;

            if (frameRequested) {
                return;
            }

            frameRequested = true;
            window.requestAnimationFrame(() => {
                page.style.setProperty("--spot-x", `${pointerX}%`);
                page.style.setProperty("--spot-y", `${pointerY}%`);
                frameRequested = false;
            });
        });

        page.addEventListener("pointerleave", () => {
            page.style.setProperty("--spot-x", "50%");
            page.style.setProperty("--spot-y", "42%");
        });
    }

    if (page && cta && !reducedMotion) {
        cta.addEventListener("click", event => {
            if (
                event.defaultPrevented ||
                event.button !== 0 ||
                event.metaKey ||
                event.ctrlKey ||
                event.shiftKey ||
                event.altKey
            ) {
                return;
            }

            event.preventDefault();
            const destination = cta.href;
            page.classList.add("is-leaving");
            window.setTimeout(() => {
                window.location.assign(destination);
            }, 220);
        });
    }
})();

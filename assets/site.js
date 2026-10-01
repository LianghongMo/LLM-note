// Highlight the section currently being read in the contents rail:
// the last section heading that has scrolled above 30% of the viewport.
(function () {
  'use strict';
  const links = Array.from(document.querySelectorAll('.toc-rail .toc a[href^="#"]'));
  if (!links.length) return;

  const entries = links
    .map((a) => {
      const section = document.getElementById(decodeURIComponent(a.getAttribute('href').slice(1)));
      const heading = section && section.querySelector('h2, h3');
      return heading ? { a, heading } : null;
    })
    .filter(Boolean);

  let current = null;
  let queued = false;

  function update() {
    queued = false;
    const line = window.innerHeight * 0.3;
    let next = entries[0];
    for (const e of entries) {
      if (e.heading.getBoundingClientRect().top <= line) next = e;
      else break;
    }
    if (next === current) return;
    if (current) current.a.classList.remove('is-current');
    next.a.classList.add('is-current');
    current = next;
  }

  function queue() {
    if (!queued) {
      queued = true;
      requestAnimationFrame(update);
    }
  }

  window.addEventListener('scroll', queue, { passive: true });
  window.addEventListener('resize', queue);
  update();
})();

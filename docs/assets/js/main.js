/* nano4M-Audio — minimal vanilla JS: scrollspy, mobile nav, copy BibTeX */
(function () {
  'use strict';

  /* ---------- mobile navigation ---------- */
  var toggle = document.getElementById('navToggle');
  var sidebar = document.getElementById('sidebar');

  function closeNav() {
    if (!sidebar) return;
    sidebar.classList.remove('open');
    if (toggle) { toggle.setAttribute('aria-expanded', 'false'); toggle.setAttribute('aria-label', 'Open section navigation'); }
  }
  function openNav() {
    if (!sidebar) return;
    sidebar.classList.add('open');
    if (toggle) { toggle.setAttribute('aria-expanded', 'true'); toggle.setAttribute('aria-label', 'Close section navigation'); }
  }

  if (toggle && sidebar) {
    toggle.addEventListener('click', function (e) {
      e.stopPropagation();
      if (sidebar.classList.contains('open')) closeNav(); else openNav();
    });
    // close after picking a section (mobile)
    sidebar.addEventListener('click', function (e) {
      if (e.target.closest('a') && window.matchMedia('(max-width:999px)').matches) closeNav();
    });
    // click outside to close
    document.addEventListener('click', function (e) {
      if (!sidebar.classList.contains('open')) return;
      if (!sidebar.contains(e.target) && e.target !== toggle) closeNav();
    });
    // Escape to close
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeNav();
    });
  }

  /* ---------- scrollspy: highlight the section in view ---------- */
  var links = Array.prototype.slice.call(document.querySelectorAll('.nav-list a'));
  var sections = [];
  links.forEach(function (a) {
    var id = a.getAttribute('href').slice(1);
    var sec = document.getElementById(id);
    if (sec) { sections.push(sec); }
  });

  function setActive(id) {
    links.forEach(function (a) {
      var on = a.getAttribute('href') === '#' + id;
      a.classList.toggle('active', on);
      if (on) a.setAttribute('aria-current', 'true'); else a.removeAttribute('aria-current');
    });
  }

  if ('IntersectionObserver' in window && sections.length) {
    var visible = {};
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) { visible[en.target.id] = en.isIntersecting ? en.intersectionRatio : 0; });
      // pick the most-visible section currently on screen
      var best = null, bestRatio = 0;
      sections.forEach(function (s) {
        var r = visible[s.id] || 0;
        if (r > bestRatio) { bestRatio = r; best = s.id; }
      });
      if (best) setActive(best);
    }, { rootMargin: '-20% 0px -65% 0px', threshold: [0, 0.25, 0.5, 1] });
    sections.forEach(function (s) { io.observe(s); });
    setActive('summary');
  }

  /* ---------- copy BibTeX ---------- */
  var copyBtn = document.getElementById('copyBib');
  var bib = document.getElementById('bibtex');
  if (copyBtn && bib) {
    copyBtn.addEventListener('click', function () {
      var text = bib.innerText.trim();
      var done = function () {
        copyBtn.textContent = 'Copied ✓';
        copyBtn.classList.add('copied');
        setTimeout(function () { copyBtn.textContent = 'Copy'; copyBtn.classList.remove('copied'); }, 1800);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else { fallback(); }
      function fallback() {
        var ta = document.createElement('textarea');
        ta.value = text; ta.style.position = 'fixed'; ta.style.opacity = '0';
        document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); done(); } catch (err) { /* no-op */ }
        document.body.removeChild(ta);
      }
    });
  }
})();

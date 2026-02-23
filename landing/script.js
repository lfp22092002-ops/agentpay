/* ============================================
   AgentPay — Landing Page Scripts
   ============================================ */

// --- Scroll Animations (Intersection Observer) ---
document.addEventListener('DOMContentLoaded', () => {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
      }
    });
  }, {
    threshold: 0.1,
    rootMargin: '0px 0px -40px 0px'
  });

  document.querySelectorAll('.fade-in').forEach((el) => observer.observe(el));

  // --- Mobile Nav Toggle ---
  const toggle = document.querySelector('.nav-toggle');
  const mobileNav = document.querySelector('.nav-mobile');
  if (toggle && mobileNav) {
    toggle.addEventListener('click', () => {
      mobileNav.classList.toggle('open');
    });
    // Close on link click
    mobileNav.querySelectorAll('a').forEach((a) => {
      a.addEventListener('click', () => mobileNav.classList.remove('open'));
    });
  }

  // --- Code Tabs ---
  const tabs = document.querySelectorAll('.code-tab');
  const blocks = document.querySelectorAll('.code-block');
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.tab;
      tabs.forEach((t) => t.classList.remove('active'));
      blocks.forEach((b) => b.classList.remove('active'));
      tab.classList.add('active');
      const block = document.getElementById('tab-' + target);
      if (block) block.classList.add('active');
    });
  });

  // --- Copy to Clipboard ---
  document.querySelectorAll('.copy-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const targetId = btn.dataset.target;
      const block = document.getElementById(targetId);
      if (!block) return;
      const code = block.querySelector('pre code, pre');
      if (!code) return;
      const text = code.textContent;
      navigator.clipboard.writeText(text).then(() => {
        btn.classList.add('copied');
        const span = btn.querySelector('span');
        const prev = span.textContent;
        span.textContent = 'Copied!';
        setTimeout(() => {
          btn.classList.remove('copied');
          span.textContent = prev;
        }, 2000);
      });
    });
  });

  // --- Smooth scroll for anchor links ---
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener('click', (e) => {
      const target = document.querySelector(anchor.getAttribute('href'));
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }
    });
  });
});

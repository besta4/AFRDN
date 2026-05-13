/**
 * nav.js — Floating nav bar with liquid sliding indicator.
 *
 * The indicator pill translates to the active tab button using getBoundingClientRect
 * so it works regardless of nav layout changes.
 */

const TABS = [
  { id: "dashboard", label: "Dashboard", icon: "ChartBar" },
  { id: "intelligence", label: "Intelligence", icon: "Brain" },
  { id: "cosmos", label: "Cosmos", icon: "Graph" },
  { id: "analytics", label: "Analytics", icon: "ChartLine" },
  { id: "audit", label: "Audit", icon: "ClockCounterClockwise" },
];

let activeTab = "dashboard";

export function initNav(onTabChange) {
  const nav = document.getElementById("floating-nav");
  const indicator = document.getElementById("nav-indicator");
  const tabList = document.getElementById("nav-tabs");

  // Build tab buttons
  TABS.forEach(({ id, label, icon }) => {
    const btn = document.createElement("button");
    btn.id = `nav-btn-${id}`;
    btn.dataset.tab = id;
    btn.className =
      "relative z-10 flex items-center gap-1.5 px-3 py-2 rounded-[10px] " +
      "text-xs font-medium transition-colors duration-200 whitespace-nowrap " +
      "text-slate-400 hover:text-slate-200";
    btn.innerHTML = `
      <i class="ph ph-${icon.toLowerCase()} text-base"></i>
      <span class="hidden sm:inline">${label}</span>
    `;
    btn.addEventListener("click", () => setActiveTab(id, onTabChange));
    tabList.appendChild(btn);
  });

  // Show nav
  nav.classList.remove("hidden");

  // Activate default tab
  setActiveTab(activeTab, onTabChange);
}

export function setActiveTab(tabId, onTabChange) {
  activeTab = tabId;

  // Update all tab content visibility
  document.querySelectorAll(".tab-content").forEach((el) => {
    el.classList.toggle("active", el.id === `tab-${tabId}`);
  });

  // Update button colours
  document.querySelectorAll("[data-tab]").forEach((btn) => {
    const active = btn.dataset.tab === tabId;
    btn.classList.toggle("text-white", active);
    btn.classList.toggle("text-slate-400", !active);
  });

  // Slide indicator pill
  _moveIndicator(tabId);

  onTabChange?.(tabId);
}

function _moveIndicator(tabId) {
  const indicator = document.getElementById("nav-indicator");
  const btn = document.getElementById(`nav-btn-${tabId}`);
  const tabList = document.getElementById("nav-tabs");
  if (!btn || !indicator || !tabList) return;

  const navRect = tabList.getBoundingClientRect();
  const btnRect = btn.getBoundingClientRect();

  indicator.style.left = `${btnRect.left - navRect.left}px`;
  indicator.style.width = `${btnRect.width}px`;
}

// Re-position on window resize
window.addEventListener("resize", () => _moveIndicator(activeTab));

import { useEffect, useState } from 'react'

export type Theme = 'dark' | 'light'
const KEY = 'lamplighter-theme'

// The theme to start with: a saved choice, else the OS preference, else dark.
// Exported so main.tsx can apply it synchronously before first paint (no flash).
export function initialTheme(): Theme {
  try {
    const saved = localStorage.getItem(KEY)
    if (saved === 'light' || saved === 'dark') return saved
    if (window.matchMedia?.('(prefers-color-scheme: light)').matches) return 'light'
  } catch {
    /* localStorage/matchMedia unavailable — fall through to the default */
  }
  return 'dark'
}

export function applyTheme(theme: Theme): void {
  document.documentElement.dataset.theme = theme
}

// Theme state + a toggle. The token overrides live in index.css under
// :root[data-theme="light"]; this just sets the attribute and persists the choice.
export function useTheme() {
  const [theme, setTheme] = useState<Theme>(initialTheme)
  useEffect(() => {
    applyTheme(theme)
    try {
      localStorage.setItem(KEY, theme)
    } catch {
      /* persistence is best-effort */
    }
  }, [theme])
  return { theme, toggle: () => setTheme((t) => (t === 'dark' ? 'light' : 'dark')) }
}

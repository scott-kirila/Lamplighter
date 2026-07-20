import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import App from './App'
import { ErrorBoundary } from './components/ErrorBoundary'
import { applyTheme, initialTheme } from './hooks/useTheme'
// The app's OWN typeface (latin, the three weights in use) — bundled woff2,
// no CDN. Before index.css so the font-face rules precede their first use;
// glyph icons (▶ ⊕ ↻ …) fall through to the platform mono stack by design.
import '@fontsource/jetbrains-mono/latin-400.css'
import '@fontsource/jetbrains-mono/latin-600.css'
import '@fontsource/jetbrains-mono/latin-700.css'
import './index.css'

// Apply the theme before first paint so a light-preference user doesn't see a
// dark flash (the CSS default) on load.
applyTheme(initialTheme())

const queryClient = new QueryClient()

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <App />
      </QueryClientProvider>
    </ErrorBoundary>
  </StrictMode>
)

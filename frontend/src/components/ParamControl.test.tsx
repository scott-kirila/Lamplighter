import { describe, expect, it } from 'vitest'
import { createRoot } from 'react-dom/client'
import { flushSync } from 'react-dom'
import { ParamControl } from './ParamControl'
import type { ParamDef } from '../types/graph'

// No testing-library in this repo — render with the real react-dom client into
// jsdom and read the container (the ErrorBoundary.test pattern).
function render(ui: React.ReactNode) {
  const container = document.createElement('div')
  const root = createRoot(container)
  flushSync(() => root.render(ui))
  return container
}

const bias: ParamDef = { name: 'bias', label: 'Bias', type: 'bool', default: true }

describe('bool param display', () => {
  it('shows the DEFAULT when the value is unset — a fresh Linear has bias ON', () => {
    // The param-count card proves the default is live (…+ 64); the checkbox
    // must agree instead of rendering unchecked for an untouched param.
    const el = render(
      <ParamControl param={bias} value={undefined} nodeColor="#888" onChange={() => {}} />
    )
    expect(el.querySelector<HTMLInputElement>('input[type=checkbox]')!.checked).toBe(true)
  })

  it('an explicit value overrides the default', () => {
    const el = render(
      <ParamControl param={bias} value={false} nodeColor="#888" onChange={() => {}} />
    )
    expect(el.querySelector<HTMLInputElement>('input[type=checkbox]')!.checked).toBe(false)
  })
})

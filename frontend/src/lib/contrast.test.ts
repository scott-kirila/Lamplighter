import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { resolve as resolvePath } from 'node:path'

/**
 * The palette's contrast floors, asserted against index.css itself.
 *
 * These are the numbers that made node titles unreadable (activations measured
 * 1.86:1 under white) and the bottom of the text ramp invisible (1.61:1). They
 * are easy to regress by eye — a hex nudged one shade "for balance" costs half
 * a point of contrast and looks fine to the person making the change. So the
 * ratios are computed here from the real token values rather than trusted to a
 * comment.
 *
 * Deliberately not a blanket WCAG AA sweep: eight ramp steps cannot span 11:1
 * to 4.5:1 and stay distinguishable, and this is a local prototyping tool, not
 * a public site. What's pinned is the floor below which text stops being text.
 */

// Read from disk rather than imported: vitest runs with `css: false`, so a
// `?raw` CSS import resolves to an empty string. vitest's cwd is frontend/.
// Comments are stripped first — the ratios written in them ("1.86:1") would
// otherwise parse as declarations and swallow the token that follows.
const css = readFileSync(resolvePath(process.cwd(), 'src/index.css'), 'utf8')
  .replace(/\/\*[\s\S]*?\*\//g, '')

/** Token values from a `:root…{ }` block, resolving one level of `var(--x)`. */
function tokens(selector: string): Record<string, string> {
  const start = css.indexOf(selector)
  if (start === -1) throw new Error(`no ${selector} block in index.css`)
  const open = css.indexOf('{', start)
  const block = css.slice(open + 1, css.indexOf('\n}', open))
  const out: Record<string, string> = {}
  // Captures custom properties AND plain ones like `color-scheme`.
  for (const [, name, value] of block.matchAll(/([\w-]+)\s*:\s*([^;]+);/g)) {
    out[name] = value.trim()
  }
  return out
}

const DARK = tokens(':root {')
const LIGHT = { ...DARK, ...tokens(":root[data-theme='light']") }

/** Follows `var(--x)` chains, and accepts a bare `--x` as shorthand for one. */
function resolve(theme: Record<string, string>, value: string): string {
  const text = value.trim()
  const ref = /^var\((--[\w-]+)\)$/.exec(text)
  if (ref) return resolve(theme, theme[ref[1]])
  if (text.startsWith('--')) return resolve(theme, theme[text])
  return text
}

function luminance(hex: string): number {
  const h = hex.replace('#', '')
  const chan = [0, 2, 4].map((i) => {
    const c = parseInt(h.slice(i, i + 2), 16) / 255
    return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4
  })
  return 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]
}

function ratio(theme: Record<string, string>, fg: string, bg: string): number {
  const a = luminance(resolve(theme, fg))
  const b = luminance(resolve(theme, bg))
  const [hi, lo] = a > b ? [a, b] : [b, a]
  return Math.round(((hi + 0.05) / (lo + 0.05)) * 100) / 100
}

const THEMES: [string, Record<string, string>][] = [['dark', DARK], ['light', LIGHT]]
const CATEGORIES = ['io', 'output', 'layers', 'activations', 'ops', 'default']

describe.each(THEMES)('%s theme contrast', (_name, theme) => {
  // 4.5 is AA for body text. Node titles are ~13px bold — normal text, not large.
  it.each(CATEGORIES)('node title on the %s fill clears AA', (category) => {
    expect(ratio(theme, `var(--node-${category}-fg)`, `var(--node-${category})`)).toBeGreaterThanOrEqual(4.5)
  })

  it('a filled control clears AA behind its white label', () => {
    expect(ratio(theme, '--text-on-accent', '--accent-fill')).toBeGreaterThanOrEqual(4.5)
  })

  it('the data-node header title clears AA on the dataset fill', () => {
    expect(ratio(theme, '--data-dataset-fg', '--accent-2')).toBeGreaterThanOrEqual(4.5)
  })

  // The ramp: ordered, and nothing at the bottom that can't be read at all.
  it('the text ramp descends without inversions', () => {
    const steps = ['--text', '--text-2', '--text-3', '--text-4', '--text-5', '--text-6', '--text-7', '--text-8']
    const ratios = steps.map((s) => ratio(theme, s, '--panel'))
    for (let i = 1; i < ratios.length; i++) {
      expect(ratios[i], `${steps[i]} must be fainter than ${steps[i - 1]}`).toBeLessThan(ratios[i - 1])
    }
  })

  it('--text-4 is the last step usable for body text', () => {
    expect(ratio(theme, '--text-4', '--panel')).toBeGreaterThanOrEqual(4.5)
  })

  it('even the faintest step stays above the vanishing point', () => {
    // 2.8 is not AA; it is the line between "subordinate" and "invisible".
    // The old floor was 1.61 (dark) / 1.74 (light).
    expect(ratio(theme, '--text-8', '--panel')).toBeGreaterThanOrEqual(2.8)
  })

  it('body text on the canvas and panels is comfortable', () => {
    expect(ratio(theme, '--text', '--bg')).toBeGreaterThanOrEqual(7)
    expect(ratio(theme, '--text', '--panel')).toBeGreaterThanOrEqual(7)
  })

  it('error text is legible on the surfaces that carry it', () => {
    for (const bg of ['--bg', '--panel', '--surface']) {
      expect(ratio(theme, '--error-text', bg), `--error-text on ${bg}`).toBeGreaterThanOrEqual(4)
    }
  })
})

describe('theme parity', () => {
  it('declares color-scheme both ways, so native controls follow', () => {
    expect(DARK['color-scheme']).toBe('dark')
    expect(LIGHT['color-scheme']).toBe('light')
  })

  it('every node category has a foreground token in both themes', () => {
    for (const category of CATEGORIES) {
      expect(DARK[`--node-${category}-fg`], `dark --node-${category}-fg`).toBeTruthy()
      expect(LIGHT[`--node-${category}-fg`], `light --node-${category}-fg`).toBeTruthy()
    }
  })
})

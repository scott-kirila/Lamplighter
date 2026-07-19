import type { CSSProperties } from 'react'

// Shared style tokens — the inline-style primitives the panels copy-pasted
// everywhere (monospace + hairline borders + the field/button/eyebrow/chip
// compounds). Reference a token directly (`style={button}`) for the common case,
// or spread-and-override for a variant (`style={{ ...button, width: '100%' }}`).
// Module-level objects are stable references, so the un-overridden cases also
// stop allocating a fresh style literal per render.
//
// Computed styles (colours/sizes derived from data — concern meters, split
// ratios, conditional accents) stay inline at the call site; tokens cover the
// static chrome, not the dynamic bits.

// The app is monospace throughout — the base font for text that isn't a control.
export const mono: CSSProperties = { fontFamily: 'monospace' }

// The 1px hairline used for borders and dividers.
export const border = '1px solid var(--border)'

// A form control (text/number input, select): field fill, hairline, rounded,
// body text. Sites override padding/width as needed.
export const field: CSSProperties = {
  background: 'var(--field)',
  border,
  borderRadius: 4,
  color: 'var(--text)',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '2px 6px',
  boxSizing: 'border-box',
}

// A small ghost button — hairline outline, transparent fill, monospace. Sites
// set width/colour/borderColor for state (e.g. an accent border when active).
export const button: CSSProperties = {
  background: 'none',
  border,
  borderRadius: 4,
  color: 'var(--text-3)',
  cursor: 'pointer',
  fontFamily: 'monospace',
  fontSize: 11,
  padding: '2px 9px',
  lineHeight: 1.4,
}

// A section eyebrow — the uppercase, letter-spaced treatment of a label above a
// panel or group. Deliberately just the invariant pair: sites vary the colour
// (text-4 heading vs text-6/8 sub-label), fontSize, and margin, so spread it and
// add those — `{ ...eyebrow, color: 'var(--text-8)', fontSize: 10 }`.
export const eyebrow: CSSProperties = {
  textTransform: 'uppercase',
  letterSpacing: 1,
}

// An inline code chip — a run name / identifier set apart from surrounding prose.
export const chip: CSSProperties = {
  background: 'var(--field)',
  border: '1px solid var(--border)',
  borderRadius: 3,
  padding: '1px 5px',
  fontFamily: 'monospace',
}

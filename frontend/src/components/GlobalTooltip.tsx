import { useEffect, useLayoutEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

// The app's ONE tooltip: a portaled bubble driven by `data-tip` attributes
// through document-level delegation — no wrapper component (66 wrap sites
// would risk their flex layouts), no clipping (an overflow:hidden panel can't
// cut a portal). Hover shows after a beat; keyboard focus shows immediately;
// any click, scroll, or Escape hides. Replaces the native title= tooltips
// (OS-styled, second-long delay) the app's explanation layer leaned on.

const SHOW_DELAY_MS = 300
const GAP = 8 // anchor ↔ bubble
const MARGIN = 10 // viewport edge clearance

export function GlobalTooltip() {
  const [tip, setTip] = useState<{ text: string; anchor: DOMRect } | null>(null)
  const ref = useRef<HTMLDivElement>(null)
  const timer = useRef<number | null>(null)
  const anchorEl = useRef<Element | null>(null)

  useEffect(() => {
    const clear = () => {
      if (timer.current != null) {
        window.clearTimeout(timer.current)
        timer.current = null
      }
    }
    const hide = () => {
      clear()
      anchorEl.current = null
      setTip(null)
    }
    const showFor = (el: Element, delay: number) => {
      const text = el.getAttribute('data-tip')
      if (!text) return
      clear()
      anchorEl.current = el
      timer.current = window.setTimeout(() => {
        // The anchor may have re-rendered away while we waited.
        if (anchorEl.current !== el || !el.isConnected) return
        setTip({ text, anchor: el.getBoundingClientRect() })
      }, delay)
    }
    const tipped = (t: EventTarget | null): Element | null =>
      t instanceof Element ? t.closest('[data-tip]') : null
    const onOver = (e: MouseEvent) => {
      const el = tipped(e.target)
      if (el && el !== anchorEl.current) showFor(el, SHOW_DELAY_MS)
    }
    const onOut = (e: MouseEvent) => {
      const el = tipped(e.target)
      if (el && el === anchorEl.current && !(e.relatedTarget instanceof Node && el.contains(e.relatedTarget))) {
        hide()
      }
    }
    const onFocus = (e: FocusEvent) => {
      const el = tipped(e.target)
      if (el) showFor(el, 0) // keyboard users shouldn't wait out a hover delay
    }
    const onBlur = () => hide()
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') hide()
    }
    document.addEventListener('mouseover', onOver)
    document.addEventListener('mouseout', onOut)
    document.addEventListener('focusin', onFocus)
    document.addEventListener('focusout', onBlur)
    document.addEventListener('mousedown', hide)
    document.addEventListener('scroll', hide, true) // panel scrolls stale the rect
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mouseover', onOver)
      document.removeEventListener('mouseout', onOut)
      document.removeEventListener('focusin', onFocus)
      document.removeEventListener('focusout', onBlur)
      document.removeEventListener('mousedown', hide)
      document.removeEventListener('scroll', hide, true)
      document.removeEventListener('keydown', onKey)
      clear()
    }
  }, [])

  // Position once the bubble's real size is measurable (pre-paint): centered
  // under the anchor, clamped to the viewport, flipped above when out of room.
  useLayoutEffect(() => {
    const el = ref.current
    if (!el || !tip) return
    const { anchor } = tip
    const w = el.offsetWidth
    const h = el.offsetHeight
    const x = Math.max(MARGIN, Math.min(anchor.left + anchor.width / 2 - w / 2, window.innerWidth - MARGIN - w))
    let y = anchor.bottom + GAP
    if (y + h > window.innerHeight - MARGIN) y = anchor.top - GAP - h
    el.style.left = `${x}px`
    el.style.top = `${y}px`
    el.style.opacity = '1'
  }, [tip])

  if (!tip) return null
  return createPortal(
    <div
      ref={ref}
      role="tooltip"
      style={{
        position: 'fixed', left: 0, top: 0, opacity: 0, zIndex: 1100,
        maxWidth: 320, padding: '5px 9px', pointerEvents: 'none',
        background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 5,
        color: 'var(--text-2)', fontSize: 11, lineHeight: 1.5,
        boxShadow: 'var(--shadow-md)', whiteSpace: 'pre-line',
        animation: 'lamplighter-fade 120ms ease',
      }}
    >
      {tip.text}
    </div>,
    document.body
  )
}

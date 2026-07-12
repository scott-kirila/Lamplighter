// Split a stored shape string ("1, 3, 28, 28") into its dimension tokens.
export function parseDims(value: string): string[] {
  return value
    .split(',')
    .map((t) => t.trim())
    .filter((t) => t !== '')
}

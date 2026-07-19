import { useCallback, useState } from 'react'
import { DeleteModelModal } from '../components/DeleteModelModal'

// Owns the pending model-delete state and renders the shared DeleteModelModal,
// so each trigger site (overview canvas, sidebar, inspector) just calls
// requestDelete(id, name) and drops {modal} into its tree. requestDelete is
// stable, so it's safe in a callback/effect dep list.
export function useModelDeleteConfirm(deleteModel: (id: string) => void) {
  const [pending, setPending] = useState<{ id: string; name: string } | null>(null)
  const requestDelete = useCallback((id: string, name: string) => setPending({ id, name }), [])
  const modal = pending ? (
    <DeleteModelModal
      name={pending.name}
      onCancel={() => setPending(null)}
      onConfirm={() => {
        deleteModel(pending.id)
        setPending(null)
      }}
    />
  ) : null
  return { requestDelete, modal }
}

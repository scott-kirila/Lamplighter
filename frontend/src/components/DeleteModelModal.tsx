import { ConfirmModal } from './ConfirmModal'
import { chip } from '../styles/ui'

// Deleting a model drops it and all its layers — one styled confirm for every
// trigger (the overview Delete key, the overview sidebar ✕, the model
// inspector), replacing the native window.confirm each used to call. Drive it
// with useModelDeleteConfirm, which owns the pending state.
export function DeleteModelModal({
  name,
  onCancel,
  onConfirm,
}: {
  name: string
  onCancel: () => void
  onConfirm: () => void
}) {
  return (
    <ConfirmModal
      onCancel={onCancel}
      actions={[{ label: 'delete model', primary: true, onClick: onConfirm }]}
    >
      Delete <code style={chip}>{name}</code>? This removes the model and all its layers from
      the project.
    </ConfirmModal>
  )
}

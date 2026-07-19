import { ConfirmModal } from './ConfirmModal'
import { chip } from '../styles/ui'

// The shared guard before any action that replaces the kernel's live model when
// its weights aren't saved: starting a new run, or restoring/resuming another.
// `target` (the run being loaded) distinguishes restore/resume from a fresh run;
// `onConfirm(true)` saves the live weights first, `onConfirm(false)` discards
// them. A thin wrapper over ConfirmModal so both callers look identical.
export function DiscardWeightsModal({
  kernelRunName,
  target,
  onCancel,
  onConfirm,
}: {
  kernelRunName: string | null
  target?: string
  onCancel: () => void
  onConfirm: (save: boolean) => void
}) {
  const verb = target ? 'continue' : 'run'
  return (
    <ConfirmModal
      onCancel={onCancel}
      actions={[
        { label: `discard & ${verb}`, onClick: () => onConfirm(false) },
        { label: `save weights & ${verb}`, onClick: () => onConfirm(true), primary: true },
      ]}
    >
      <code style={chip}>{kernelRunName}</code> is the live model and its weights aren't saved.{' '}
      {target ? (
        <>
          Loading <code style={chip}>{target}</code> will discard them.
        </>
      ) : (
        <>Starting a new run will discard them.</>
      )}
    </ConfirmModal>
  )
}

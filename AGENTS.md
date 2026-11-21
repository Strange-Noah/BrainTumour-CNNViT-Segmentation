# Novo Nordisk repository governance

## Branch ownership

- Main is the shared project baseline: common infrastructure,
  dataset and evaluation contracts, shared deployment, and
  explicitly accepted team-level deliverables.
- All personal or non-team-shared development belongs on that
  contributor's own long-lived logical workstream branch.
- The repository owner follows the same rule as other contributors.
- Continue an existing workstream on its existing branch.
- Branches represent independent development lines, not individual
  tasks, bug fixes, documents, experiment versions or agent sessions.
- Do not create additional permanent branches without an explicit
  owner instruction.
- Temporary branches are permitted for necessary isolation and PR
  review. Once integrated, verify preservation and delete them.
- Never leave a task branch behind merely as a version record.
- Use commits, run directories, configuration hashes, immutable
  tags and experiment reports instead.
- Never modify, merge, rename, rebase or delete teammate-owned
  branches without their explicit permission.
- Preserve summer-archive unchanged.

## Promotion and synchronization

Do not automatically merge entire personal workstreams into main.
Promote only explicitly accepted shared changes or milestone results.

Synchronize reviewed shared changes into workstreams selectively.
Inspect any merge that contains worktree deletions; never allow
main cleanup commits to erase workstream-specific implementation.

Before deleting a file or branch, verify that its complete work
and required source history remain accessible elsewhere in the repo.

Do not force-push or rewrite published history.

## Data and research

Use the frozen patient split and canonical dataset.
Preserve original labels and physical image geometry.

Keep medical images, patient-level outputs, credentials and large
checkpoints in authorized private storage. GitHub contains code,
schemas, aggregate results and provenance.

Record immutable code, data, configuration and artifact identities.

## Execution and security

Use explicitly approved workstream sources for credential-bearing jobs.
Never grant arbitrary branch code access to shared secrets.
Do not bypass environment restrictions or identity checks.

Do not start GPU training or paid resources without explicit authorization.

For authorized GPU runs, observe approximately the first ten minutes
of actual computation, report measured health, then leave healthy
training unattended. Preserve automatic checkpointing and independent
result recovery.

Do not create redundant benchmarks or new infrastructure projects
when narrowly scoped changes suffice.

Repository organization work does not authorize model training.

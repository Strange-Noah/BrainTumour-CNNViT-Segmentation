# Preservation decisions

The minimal read-only audit PR #37 advanced main from 56fc2a7 to 68bd4f1. The latter is the migration base; the former remains the immutable recovered-v1 source. The continuation branch is no longer directly ahead of the new base: five changed blobs were preserved in the exact archive and selectively restored, not merged wholesale. No audit-module content was dropped.

Original/different source histories and all configuration/data/checkpoint identities are retained. Private active metadata is stale and remains unchanged; provider sessions terminal, 963-file collection verified, no active collector. The old automatic collector stays disabled until trusted disabled-by-default main routing is accepted.

The archived original SVG contains inherited trailing whitespace. Byte-for-byte preservation takes precedence over rewriting it. `.gitattributes` disables whitespace lint only for exact historical archive snapshots and that original figure; Git blob equality remains mandatory. New code and current documentation retain normal whitespace checks.

Personal preservation verification: 23 divergent blobs exact; 33 CPU tests completed (32 passed, 1 skipped); syntax compilation passed; default repository renderer unchanged. No GPU submissions.

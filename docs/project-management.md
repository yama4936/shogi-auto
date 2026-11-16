# Project management

## Sources of truth
Git stores reviewed source, experiment plans, tests and small provenance manifests.
Models, arrays, game records and raw event/transport logs stay in external artifact storage.
Never store passwords, keys, host addresses, SSH configuration, personal machine paths or notebook outputs here.
Private/public repository visibility does not relax these exclusions.

The publication branch starts from GitHub main and imports reviewed offline-crossplay source.
The complete original local history is preserved separately, not rewritten or pushed.
Generated payload notebooks are intentionally excluded; the original CPU/CUDA smoke notebook remains.
The legacy src/ tree remains untouched. research/reference/src is the separately frozen
current research reference source; its per-file checksums are in SOURCES.json.
It is NOT a replacement for src/, a complete live deployment, or proof of stronger play.
Keep the two imports separate using an explicit PYTHONPATH.

## Reproducible experiments
After committing source in a clean checkout:
~~~
python scripts/experiment_manifest.py record --id exp-example --output /external/artifacts/exp-example/manifest.json --artifact config=/external/config.json --artifact dataset=/external/data.bin --artifact model=/external/model.json
python scripts/experiment_manifest.py verify /external/artifacts/exp-example/manifest.json --artifact config=/external/config.json --artifact dataset=/external/data.bin --artifact model=/external/model.json
~~~
Only logical labels, sizes and SHA256 values are recorded; local artifact paths are not.
Every manifest records the exact code commit and refuses a dirty tree.
Use unique experiment and attempt IDs. Existing manifests cannot be overwritten.
Record environment/dependency metadata, seed, split, resource/time limits and evaluation
contract in the hashed config. Add results as a separately named artifact/manifest,
never rewrite an initial receipt. A manifest binds bytes; it does not prove scientific
validity, historical execution or complete dependencies. Use sealed, immutable artifacts; metadata checks detect ordinary concurrent edits but are not protection against a hostile writer or a transactional filesystem snapshot.

## Migration status (2026-11-16)
Historical work used dated folders, snapshots, hash seals, an experiment registry and progress reports.
Ninety-one experiments were registered at the initial audit. Historical records remain external
and immutable; do not backdate new manifests or pretend old jobs ran from a new commit.
The live Goal and Floodgate remain on their original paths and are NOT switched by this change.
Historical raw logs, tournament records, models, internal handoff/review notes, machine configuration and credentials are excluded.
Legacy source comments may refer to internal research reports that are intentionally not published.
The legacy GitHub main was at dca8d91; this branch imports later local source without importing payload-notebook history.

Future jobs must adopt this checkout only at a verified safe boundary after source/runtime review.
Until the controller is explicitly handed over, new live Goal experiments still use its old registry.
Do not merge, deploy, restart, cut off games or promote models as part of repository organization.

## Publication
Audit the complete outgoing history, not only the latest diff, before pushing.
Notebook outputs and embedded data need explicit source/permission review.
Current production wrappers, external dependencies, datasets and model availability must be
reviewed separately; this repository snapshot alone cannot recreate the entire live system.

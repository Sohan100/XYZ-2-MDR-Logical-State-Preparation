# Consensus driver

The driver behind review rounds 5 to 8 of [`docs/xyz2_usefulness.md`](../../docs/xyz2_usefulness.md). Each round, ten reviewers vote on every claim and on the overall answer; disputed claims are revised by one agent and checked by two verifiers; reviewer omissions are triaged; the loop stops when every item has at least 80% agreement or after round 8.

- `consensus.py` keeps the state in `.consensus/state.json` (ignored by git) and writes one workflow script per stage from the templates `tpl_fix.js`, `tpl_review.js` and `tpl_write.js`.
- It reads the result of every finished agent from the workflow journals by label, so a stage cut off by a usage limit reruns only the agents that did not finish.

Cycle: `python3 consensus.py fix` → run `.consensus/run_fix.js` → `merge-fix` → `review` → run `run_review.js` → `merge-review` → (repeat) → `write` → run `run_write.js`. `python3 consensus.py status` prints the progress.

The journal directory (`WF`) and the old run id at the top of `consensus.py` are specific to the Claude Code session that ran it. The final state and every round's output are in `docs/data/campaign/usefulness/`.

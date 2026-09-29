## What this changes

<!-- One or two sentences: what is different after this merges, and why. -->

## How it was checked

<!-- Paste real output, not a summary. For anything that touches data, include a run at
real scale: row counts, memory, timings or the printed metric. A green test suite is
plumbing evidence, not proof of correctness (see CONTRIBUTING.md). -->

## Checklist

- [ ] Nothing synthetic reaches a production path or a metric.
- [ ] No score is written into source, docs or the frontend; metrics are read from `/api/model/status`.
- [ ] Anything added at serve time has a real memory measurement (the box is killed at 512 MB).
- [ ] SHAP explanations still work.
- [ ] New caveats are written in `docs/known-issues.md`.
- [ ] `frontend/src/api/types.ts` still mirrors `backend/app/api/schemas.py`.
- [ ] Changes to `.github/workflows/`, the promotion gate or `valid_date` are called out above, or there are none.

# Demo-day runbook

The live site runs on Render's free tier. It sleeps after 15 minutes without traffic and
takes tens of seconds to wake. Everything below exists so that never happens on stage.

## Keep it awake (once, before the finale)

The repo has no pinger. `render.yaml` says one is needed, and a browser tab does not count.
Set one up on a free external monitor, for example UptimeRobot:

1. Add a new monitor of type **HTTP(s)**.
2. URL: `https://sanket-a0dd.onrender.com/api/health`. That route answers both GET and
   HEAD, so a HEAD-only monitor works.
3. Interval: 5 minutes.

Before relying on it, read Render's current free-tier page on instance hours and check that
one always-awake service fits the monthly allowance. The allowance is Render's to change,
so it is not written down here.

## T-15 minutes

1. Open `https://sanket-a0dd.onrender.com/api/health` once and check that:
   - `status` is `"ok"` and `model_trained` is `true`;
   - `current_run_id` is the run the slides quote;
   - `prebuilt.active` is `true`. If it is `false`, the box is building screens live and
     will be slow. The usual cause is a deploy that did not come from `refresh-data.yml`;
     run that workflow and wait for it to finish.
2. Open the site and click through once: map, a district panel, **Replay**. That loads each
   screen's prebuilt response.
3. Open the recorded demo in another tab as the fallback: https://youtu.be/rY73m6pxdpE

## Demo order

1. **Replay a real bust.** Kerala 2018 or Ockhi 2017 (`backend/app/services/replay_cases.py`).
   It shows what the system would have said before anyone knew the outcome.
2. **The national map.** Step the lead day and explain why the map can step down where
   variables stop being scored (`docs/known-issues.md`).
3. **A district panel.** "What drove this prediction" is SHAP over the served XGBoost model.
4. **About.** The limitations, stated plainly.

Every number spoken comes from the site on the day, never from a slide made earlier.

## If the site will not answer

- **The video** covers the whole flow.
- **A local server from the same bundle.** Follow the README's Quick start, with the backend
  on port 8000 and the frontend on 5173. It needs the data bundle and the model installed
  locally beforehand. Do this the day before, not on stage.
- **Probe once and wait.** Render rate-limits per IP, and repeated reloads make it worse.

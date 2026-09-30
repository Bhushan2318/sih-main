# Security policy

## Reporting a vulnerability

Please report security problems privately, not in a public issue.

Use GitHub's private reporting: the repository's **Security** tab → **Report a
vulnerability**. Include what you found, how to reproduce it, and what an attacker could
do with it. You should hear back within a few days. Please give us reasonable time to fix
the problem before you disclose it.

## What is in scope

- The code in this repository.
- The live site at <https://sanket-a0dd.onrender.com>. It is a single free-tier instance
  with 512 MB of memory, so please do not load-test it or run automated scanners against
  it. A report that explains the issue is enough.

The serving instance is deliberately read-only. Uploads and retraining are refused there
(`SERVING_READ_ONLY`), and requests outside the published catalogue are rejected before
their body is read. Findings that get past those guards are especially welcome.

## Secrets

The repository holds no credentials. The only deploy secret, the Render deploy hook, is
kept in GitHub Actions secrets. If you ever find a credential committed here, report it
privately as above.

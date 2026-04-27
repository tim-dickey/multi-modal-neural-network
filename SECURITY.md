# Security Policy

## Supported Versions

Only the latest commit on `main` receives security fixes. Older states of the
repository are not actively patched.

| Version / State  | Supported |
| ---------------- | --------- |
| Latest `main`    | ✅ Yes    |
| Historical commits | ❌ No   |

## Reporting a Vulnerability

**Do not open a public issue to report a security vulnerability.**

Use [GitHub's private security advisory feature](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing/privately-reporting-a-security-vulnerability)
to send a confidential report directly to the maintainers:

1. Navigate to the **Security** tab of this repository.
2. Select **Report a vulnerability**.
3. Fill in the advisory form with as much detail as possible.

Alternatively, contact the maintainer through the repository profile if the
advisory form is unavailable.

## Response Expectations

| Stage                              | Target time     |
| ---------------------------------- | --------------- |
| Initial acknowledgment             | 5 business days |
| Status update                      | 14 business days |
| Fix or mitigation (if confirmed)   | Best effort, communicated in the advisory |

## Scope

The following are **in scope** for security reports:

- **Model loading vulnerabilities** — e.g., unsafe deserialization of pickle
  files or model checkpoints that could execute arbitrary code.
- **Dependency vulnerabilities** in packages listed in `requirements.txt` or
  `pyproject.toml` that have known CVEs affecting this project's usage.
- **CI/CD injection risks** in `.github/workflows/` files (e.g., unsanitized
  inputs in shell steps).
- **Credential or secret exposure** — any token, key, or credential present in
  repository content or CI logs.

## Out of Scope

The following are **not** in scope:

- Vulnerabilities in third-party libraries or frameworks (report those upstream).
- Findings that require physical access to a maintainer's machine.
- Purely theoretical findings without a concrete reproduction path.
- Issues already publicly known or disclosed elsewhere.

## Disclosure Policy

Confirmed vulnerabilities will be disclosed via a published GitHub Security
Advisory after a fix or mitigation is in place, or after a reasonable disclosure
deadline agreed with the reporter.

## Contributing

Thanks for contributing! A few guidelines to help keep CI and analysis working smoothly.

### Codacy integration
- This repository can run Codacy analysis in CI. To enable Codacy for this repository:
  1. Create a Codacy project for this repository (on codacy.com) and obtain the project token.
 2. In GitHub, go to **Settings → Secrets → Actions** for this repository and add a secret named `CODACY_PROJECT_TOKEN` with the project token value.
 3. Once the secret is present, the GitHub Action `Codacy Analysis` will run automatically on pushes and pull requests.

Notes:
- The workflow installs the Codacy CLI (`@codacy/cli`) and runs `codacy analyze` against the repository. The job is skipped when `CODACY_PROJECT_TOKEN` is not configured.
- If you prefer to run Codacy analysis locally, install the Codacy CLI (`npm install -g @codacy/cli`) and run:

```
codacy analyze --project-token "<YOUR_TOKEN>" --directory .
```

If you experience issues with the CLI on CI, please open an issue or reach out to the maintainers.

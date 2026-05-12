# Deployment

## Local

```bash
npm install
npm run dev
```

## Validation

```bash
npm run lint
npm run typecheck
npm run test
npm run build
```

## Vercel

1. Import the GitHub repo.
2. Select Next.js.
3. Build command: `npm run build`.
4. Add optional env vars:
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL`
   - `ROBOFLOW_API_KEY`
   - `ROBOFLOW_MODEL_ID`
   - `ROBOFLOW_API_URL`
5. Deploy.

Fallback modes work without OpenAI or Roboflow keys.

## GitHub Actions

The repo includes `.github/workflows/ci.yml` for lint, typecheck, tests, and production build on every push and pull request.

Vercel deployment is currently handled by the linked Vercel project. To deploy from GitHub Actions instead, add these repository secrets and a Vercel deploy job:

- `VERCEL_TOKEN`
- `VERCEL_ORG_ID`
- `VERCEL_PROJECT_ID`

The current Vercel link is enough for normal production deploys from `main`.

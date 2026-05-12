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

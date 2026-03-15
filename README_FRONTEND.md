Frontend quickstart

This folder contains a minimal React (Vite) frontend that calls the local backend api endpoints.

Prereqs

- Node.js (16+ recommended) and npm/yarn
- Python virtualenv for backend (your existing .venv)

Install and run frontend

1. from project root -> frontend

```pwsh
cd "C:\Users\navan\Downloads\capstone project\frontend"
npm install
npm run dev
```

The dev server will open on http://localhost:5173 by default. Use the UI to call the backend.

Run backend (recommended in separate terminal)

```pwsh
# from project root
& '.\.venv\Scripts\Activate.ps1'
.
# use the persistent starter
.
Start-Process powershell -ArgumentList '-NoExit','-File','start_backend.ps1'
```

Notes

- The frontend is a lightweight dev scaffold; for production, build with `npm run build` and serve the `dist/` files with a static server.
- The `start_backend.ps1` script will restart the backend automatically if it exits; check `outputs/backend_run.log` for logs.

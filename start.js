const { spawn, exec } = require('child_process');
const http = require('http');

const fs = require('fs');
const path = require('path');
const APP = 'app.py';
const HOST = '127.0.0.1';
const PORT = 8501;
const URL = `http://${HOST}:${PORT}`;
const START_TIMEOUT = 60 * 1000; // 60s
const POLL_INTERVAL = 500; // ms

function startStreamlit() {
  // Require project's venv Python only (no fallback). Use __dirname so this file's
  // location is the authoritative project root when called from the wrapper.
  const venvPyWin = path.join(__dirname, '.venv', 'Scripts', 'python.exe');
  const venvPyNix = path.join(__dirname, '.venv', 'bin', 'python');
  const pythonExe = fs.existsSync(venvPyWin) ? venvPyWin : fs.existsSync(venvPyNix) ? venvPyNix : null;
  if (!pythonExe) {
    console.error('\u274C venv python not found at ' + path.join(__dirname, '.venv'));
    console.error('Create the venv with: python -m venv .venv');
    console.error('Then install streamlit into it: .venv\\Scripts\\python.exe -m pip install streamlit');
    process.exit(1);
  }

  console.log('Using python:', pythonExe);
  // Ensure venv Scripts/bin directory is first on PATH so any subprocesses that call
  // "python" resolve to the venv python instead of a global WindowsApps shim.
  const venvBin = path.dirname(pythonExe);
  const env = Object.assign({}, process.env);
  env.PATH = venvBin + path.delimiter + (process.env.PATH || '');
  env.PYTHONEXECUTABLE = pythonExe;
  env.PYTHON_SYS_EXECUTABLE = pythonExe;
  const child = spawn(pythonExe, ['-m', 'streamlit', 'run', APP], { windowsHide: true, stdio: 'inherit', env });
  child.on('exit', (code) => {
    console.error('streamlit exited with code', code);
    process.exit(code || 0);
  });
  child.on('error', (err) => {
    console.error('Failed to start streamlit process:', err);
    process.exit(1);
  });
  return child;
}

function checkServerReady(timeout = START_TIMEOUT) {
  const deadline = Date.now() + timeout;
  return new Promise((resolve) => {
    (function poll() {
      const req = http.get(URL, (res) => {
        res.resume();
        resolve(true);
      });
      req.on('error', () => {
        if (Date.now() < deadline) setTimeout(poll, POLL_INTERVAL);
        else resolve(false);
      });
      req.setTimeout(2000, () => {
        req.abort();
      });
    })();
  });
}

function openBrowser(url) {
  const platform = process.platform;
  if (platform === 'win32') {
    exec(`start "" "${url}"`);
  } else if (platform === 'darwin') {
    exec(`open "${url}"`);
  } else {
    exec(`xdg-open "${url}"`);
  }
}

async function main() {
  console.log('Starting Streamlit...');
  startStreamlit();
  console.log(`Waiting for server at ${URL}...`);
  const ok = await checkServerReady();
  if (ok) {
    console.log('Server is ready — opening browser');
    openBrowser(URL);
  } else {
    console.error('Server did not become ready within timeout');
    process.exit(1);
  }
}

main().catch((err) => {
  console.error('launcher error', err);
  process.exit(1);
});

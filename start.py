"""Cross-platform launcher: Python 3.11+ and Node 18+ required."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent


def venv_python(root=ROOT, windows=None):
    windows = os.name == "nt" if windows is None else windows
    return root / "backend" / ".venv" / ("Scripts/python.exe" if windows else "bin/python")


def tool(name):
    found = shutil.which(name)
    if found:
        return found
    for base in (Path("/opt/homebrew/bin"), Path("/usr/local/bin")):
        candidate = base / name
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError(f"{name} não encontrado. Instale Node.js LTS (inclui npm) e abra novamente o terminal.")


def run(args, cwd=ROOT, env=None):
    subprocess.run([str(a) for a in args], cwd=cwd, env=env, check=True)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def npm_command(node, npm, windows=None):
    """Run npm's JavaScript entry point directly (no Windows batch quoting)."""
    windows = os.name == "nt" if windows is None else windows
    path = Path(npm)
    resolved = path.resolve()
    if resolved.name == "npm-cli.js":
        return [node, str(resolved)]
    cli = path.parent / "node_modules" / "npm" / "bin" / "npm-cli.js"
    if cli.is_file():
        return [node, str(cli)]
    if windows:
        raise RuntimeError("npm-cli.js não encontrado. Reinstale Node.js LTS com npm.")
    return [npm]


def needs_install(marker, source):
    return not marker.exists() or marker.read_text().strip() != fingerprint(source)


def ensure_setup(force=False):
    node, npm = tool("node"), tool("npm")
    npm_args = npm_command(node, npm)
    env = {**os.environ, "PATH": str(Path(node).parent) + os.pathsep + os.environ.get("PATH", "")}
    version = subprocess.check_output([node, "--version"], text=True).strip()
    if int(version.lstrip("v").split(".")[0]) < 18:
        raise RuntimeError("Node.js 18 ou superior é necessário. Instale Node.js LTS.")
    python = venv_python()
    uv = shutil.which("uv")
    if not python.exists():
        print("Preparando ambiente Python…", flush=True)
        if uv:
            run([uv, "venv", "--python", sys.executable, ROOT / "backend" / ".venv"])
        else:
            run([sys.executable, "-m", "venv", ROOT / "backend" / ".venv"])
    requirements = ROOT / "backend" / "requirements.txt"
    marker = ROOT / "backend" / ".venv" / ".lab-requirements.sha256"
    if force or needs_install(marker, requirements):
        print("Instalando dependências do backend…", flush=True)
        run([uv, "pip", "install", "--python", python, "-r", requirements] if uv else [python, "-m", "pip", "install", "-r", requirements])
        marker.write_text(fingerprint(requirements), encoding="utf-8")
    frontend = ROOT / "frontend"
    lock = frontend / "package-lock.json"
    marker = frontend / "node_modules" / ".lab-lock.sha256"
    if force or needs_install(marker, lock):
        print("Instalando dependências da interface…", flush=True)
        run([*npm_args, "ci"], cwd=frontend, env=env)
        marker.write_text(fingerprint(lock), encoding="utf-8")
    print("Compilando interface…", flush=True)
    run([*npm_args, "run", "build"], cwd=frontend, env=env)
    return python


def mcp_config(port):
    path = ROOT / ".mcp.local.json"
    path.write_text(json.dumps({"mcpServers": {"dl-lab": {
        "type": "stdio", "command": str(venv_python()),
        "args": [str(ROOT / "backend" / "mcp_server" / "server.py")],
        "env": {"DL_LAB_API": f"http://127.0.0.1:{port}", "PYTHONIOENCODING": "utf-8"},
    }}}, indent=2), encoding="utf-8")
    return path


def port_free(port):
    with socket.socket() as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def stop(proc, windows=None):
    windows = os.name == "nt" if windows is None else windows
    if proc.poll() is not None:
        return
    try:
        if windows:
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=8)
    except (OSError, subprocess.TimeoutExpired):
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Laboratório de Deep Learning — Windows e macOS")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--setup-only", action="store_true")
    parser.add_argument("--skip-setup", action="store_true", help="Usa ambiente e interface já preparados")
    parser.add_argument("--reinstall", action="store_true")
    parser.add_argument("--check", action="store_true", help="Verifica pré-requisitos sem instalar")
    args = parser.parse_args(argv)
    if sys.version_info < (3, 11):
        raise RuntimeError("Instale Python 3.11 ou superior e execute este arquivo com ele.")
    if not 1 <= args.port <= 65535:
        parser.error("porta deve estar entre 1 e 65535")
    if args.check:
        print(f"Python: {sys.version.split()[0]}\nNode: {tool('node')}\nnpm: {tool('npm')}\nAmbiente: {venv_python()}")
        return
    if not args.setup_only and not port_free(args.port):
        raise RuntimeError(f"A porta {args.port} já está em uso. Feche a execução anterior ou use --port 8002.")
    python = venv_python() if args.skip_setup else ensure_setup(args.reinstall)
    if not python.exists() or not (ROOT / "frontend" / "dist" / "index.html").exists():
        raise RuntimeError("Ambiente incompleto. Execute novamente sem --skip-setup.")
    mcp_config(args.port)
    if args.setup_only:
        print("Preparação concluída.")
        return
    url = f"http://127.0.0.1:{args.port}"
    env = {**os.environ, "DL_LAB_API": url, "PYTHONIOENCODING": "utf-8", "DL_LAB_SERVE_FRONTEND": "1"}
    kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen([str(python), "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(args.port)], cwd=ROOT / "backend", env=env, **kwargs)
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError("O backend encerrou durante a inicialização. Veja o erro acima.")
            try:
                with urllib.request.urlopen(url + "/api/health", timeout=1) as response:
                    if json.load(response).get("ok"):
                        break
            except (OSError, ValueError):
                time.sleep(.3)
        else:
            raise RuntimeError("O backend não iniciou em 120 segundos.")
        print(f"\nLaboratório disponível em {url}\nMantenha esta janela aberta. Ctrl+C encerra o laboratório.", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        proc.wait()
        if proc.returncode:
            raise RuntimeError(f"O backend encerrou com código {proc.returncode}.")
    finally:
        stop(proc)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nLaboratório encerrado.")
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"\nErro: {exc}", file=sys.stderr)
        sys.exit(1)

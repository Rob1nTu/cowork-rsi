#!/usr/bin/env bash
set -euo pipefail
ROOT=/data/L202500306/L202500306/tlb/cowork-rsi
BASE=/data/L202500306/L202500306/tlb/conda-envs/vllm025
mkdir -p "$ROOT/tools" /tmp/cowork-rsi
cd /tmp/cowork-rsi
if [ ! -x "$ROOT/tools/usr/bin/bwrap" ]; then
  apt-get download bubblewrap
  dpkg-deb --fsys-tarfile bubblewrap_*.deb | tar --no-same-owner -xf - -C "$ROOT/tools"
fi
"$ROOT/tools/usr/bin/bwrap" --unshare-user --unshare-pid --unshare-net --ro-bind /usr /usr --ro-bind /bin /bin --ro-bind /lib /lib --ro-bind /lib64 /lib64 --dir /proc --dev /dev --tmpfs /tmp /bin/bash -c 'echo BWRAP_OK; test ! -e /data'
if [ ! -x /tmp/cowork-agent-env/bin/python ]; then
  "$BASE/bin/python" -m venv --copies /tmp/cowork-agent-env
fi
export HTTPS_PROXY=http://127.0.0.1:17897 HTTP_PROXY=http://127.0.0.1:17897 NO_PROXY=127.0.0.1,localhost
/tmp/cowork-agent-env/bin/python -m pip install --disable-pip-version-check --timeout 30 --retries 5 'litellm==1.75.5.post2' 'openai==1.99.9' 'jinja2==3.1.6' 'pydantic==2.11.7' 'rich==14.1.0' 'platformdirs==4.3.8' 'python-dotenv==1.1.1' 'pyyaml==6.0.2' 'requests==2.32.4' 'pytest==8.3.4' 'openpyxl==3.1.5' 'pandas==2.3.1' 'python-docx==1.2.0' 'python-pptx==1.0.2' 'pypdf==5.9.0' 'pdfplumber==0.11.7' 'pillow==11.3.0' 'numpy==2.3.2' 'beautifulsoup4==4.13.4' 'lxml==6.0.0' 'httpx==0.28.1' 'anyio==4.10.0' 'tokenizers==0.21.4' 'huggingface-hub==0.34.3' 'tiktoken==0.11.0' 'tenacity==9.1.2'
/tmp/cowork-agent-env/bin/python -m pip freeze > "$ROOT/agent-requirements.lock.txt"
echo PROVISION_DONE

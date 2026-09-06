"""Restricted subprocess runner used by visual workflow script nodes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict


_PY_WRAPPER = r'''import json, sys
safe = {"abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
        "enumerate": enumerate, "float": float, "int": int, "len": len,
        "list": list, "max": max, "min": min, "range": range, "round": round,
        "set": set, "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
        "zip": zip, "Exception": Exception, "ValueError": ValueError}
scope = {"__builtins__": safe}
payload = json.loads(sys.stdin.read() or "{}")
code, params = payload.get("code", ""), payload.get("params", {})
exec(code, scope, scope)
main = scope.get("main")
if not callable(main): raise ValueError("必须定义 main(params) 函数")
result = main(params)
if not isinstance(result, dict): raise ValueError("main(params) 必须返回对象")
# The parent/child transport stays ASCII even when ``-I`` ignores
# PYTHONIOENCODING and Windows chooses a legacy console encoding.
print(json.dumps(result, ensure_ascii=True, default=str))
'''

_JS_WRAPPER = r'''const vm=require("vm"), fs=require("fs");
const payload=JSON.parse(fs.readFileSync(0,"utf8")||"{}"), source=payload.code||"", params=payload.params||{};
const sandbox={params, result:null, JSON, Math, Number, String, Boolean, Array, Object};
vm.createContext(sandbox, {codeGeneration:{strings:false,wasm:false}});
new vm.Script(source+"\n;result=main(params);",{filename:"workflow-script.js"}).runInContext(sandbox,{timeout:5000});
if(!sandbox.result || typeof sandbox.result!=="object" || Array.isArray(sandbox.result)) throw new Error("main(params) 必须返回对象");
process.stdout.write(JSON.stringify(sandbox.result));
'''


def _sanitize_unicode(value: Any) -> Any:
    """Replace isolated UTF-16 surrogate code units left by clipboard input."""
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, list):
        return [_sanitize_unicode(item) for item in value]
    if isinstance(value, dict):
        return {str(_sanitize_unicode(key)): _sanitize_unicode(item) for key, item in value.items()}
    return value


def run_workflow_script(language: str, code: str, params: Dict[str, Any], timeout_seconds: float = 5) -> Dict[str, Any]:
    """Execute user transformation code with no inherited secrets or working directory access."""
    language = language.lower().strip()
    code = str(_sanitize_unicode(code))
    params = _sanitize_unicode(params)
    timeout = max(0.2, min(float(timeout_seconds), 30.0))
    if len(code.encode("utf-8")) > 100_000:
        raise ValueError("脚本不能超过 100 KB")
    env = {key: value for key, value in os.environ.items() if key.upper() in {"PATH", "PATHEXT", "SYSTEMROOT", "WINDIR", "COMSPEC", "TEMP", "TMP"}}
    env["PYTHONIOENCODING"] = "utf-8"
    with tempfile.TemporaryDirectory(prefix="workflow-script-") as temp:
        if language == "python":
            command = [sys.executable, "-I", "-S", "-c", _PY_WRAPPER]
            # Keep the stdin protocol ASCII-only.  On Windows, a subprocess
            # launched from a non-UTF-8 console can otherwise corrupt CJK
            # text before the isolated interpreter parses the JSON frame.
            payload = json.dumps({"code": code, "params": params}, ensure_ascii=True, default=str)
        elif language in {"javascript", "js"}:
            node = shutil.which("node")
            if not node:
                raise RuntimeError("服务器未安装 Node.js，无法执行 JavaScript")
            wrapper = Path(temp) / "runner.js"
            wrapper.write_text(_JS_WRAPPER, encoding="utf-8")
            command = [node, "--no-addons", str(wrapper)]
            payload = json.dumps({"code": code, "params": params}, ensure_ascii=True, default=str)
        else:
            raise ValueError("脚本语言仅支持 python 或 javascript")
        try:
            completed = subprocess.run(command, input=payload, text=True, encoding="utf-8", errors="replace", capture_output=True, cwd=temp, env=env, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"脚本执行超过 {timeout:g} 秒") from exc
        stdout, stderr = completed.stdout[-20_000:], completed.stderr[-20_000:]
        if completed.returncode != 0:
            raise RuntimeError((stderr or stdout or "脚本执行失败").strip())
        try:
            result = json.loads(stdout.strip())
        except json.JSONDecodeError as exc:
            raise ValueError("脚本没有返回有效对象") from exc
        if not isinstance(result, dict):
            raise ValueError("main(params) 必须返回对象")
        return result

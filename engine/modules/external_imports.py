"""Conservative import helpers for OpenAPI documents and SKILL.md packages."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import socket
import subprocess
import tempfile
import urllib.parse
import urllib.request
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List


MAX_REMOTE_BYTES = 2_000_000
MAX_SKILL_FILES = 100
MAX_SKILL_BYTES = 5_000_000


def validate_remote_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("仅允许使用 HTTPS URL")
    if os.environ.get("AGENTFORGE_ALLOW_PRIVATE_IMPORTS") != "1":
        for result in socket.getaddrinfo(parsed.hostname, 443):
            address = ipaddress.ip_address(result[4][0])
            if address.is_private or address.is_loopback or address.is_link_local or address.is_reserved:
                raise ValueError("默认禁止访问私网、本机或保留地址")
    return url


def read_remote_document(url: str) -> bytes:
    request = urllib.request.Request(validate_remote_url(url), headers={"User-Agent":"AgentForge-Importer/1.0"})
    with urllib.request.urlopen(request, timeout=12) as response:  # noqa: S310 - validated above
        content = response.read(MAX_REMOTE_BYTES + 1)
    if len(content) > MAX_REMOTE_BYTES:
        raise ValueError("远程文档超过 2MB 限制")
    return content


def discover_mcp_tools(url: str, credential_env: str = "", timeout: float = 8) -> List[Dict[str, Any]]:
    """Discover tools exposed by a remote MCP HTTP endpoint."""
    endpoint = validate_remote_url(url)
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if credential_env and os.environ.get(credential_env):
        headers["Authorization"] = f"Bearer {os.environ[credential_env]}"
    payload = json.dumps({"jsonrpc": "2.0", "id": "agentforge-discovery", "method": "tools/list", "params": {}}).encode("utf-8")
    request = urllib.request.Request(endpoint, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 - URL validated above
        raw = response.read(MAX_REMOTE_BYTES + 1)
    if len(raw) > MAX_REMOTE_BYTES:
        raise ValueError("MCP 响应超过 2MB 限制")
    text = raw.decode("utf-8", errors="replace").strip()
    # Streamable HTTP MCP servers commonly return a JSON-RPC message wrapped
    # in SSE.  Treat both JSON and SSE as first-class MCP responses.
    sse_chunks = [
        line[5:].strip() for line in text.splitlines()
        if line.startswith("data:") and line[5:].strip() not in {"[DONE]", ""}
    ]
    if sse_chunks:
        text = "\n".join(sse_chunks)
    parsed = json.loads(text)
    if parsed.get("error"):
        raise ValueError(f"MCP 工具发现失败：{parsed['error']}")
    discovered = (parsed.get("result") or {}).get("tools") or []
    if not isinstance(discovered, list) or not discovered:
        raise ValueError("MCP 服务未返回可用工具")
    return [item for item in discovered if isinstance(item, dict) and item.get("name")]


def parse_openapi(content: bytes) -> Dict[str, Any]:
    text = content.decode("utf-8")
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
            result = yaml.safe_load(text)
        except Exception as exc:
            raise ValueError("OpenAPI 文档必须是合法 JSON 或 YAML") from exc
    if not isinstance(result, dict) or not isinstance(result.get("paths"), dict):
        raise ValueError("OpenAPI 文档缺少 paths")
    return result


def openapi_operations(document: Dict[str, Any], source_url: str, credential_env: str = "") -> List[Dict[str, Any]]:
    servers = document.get("servers") or []
    base_url = str((servers[0] if servers else {}).get("url") or source_url.rsplit("/",1)[0]).rstrip("/")
    operations = []
    for path, path_item in document["paths"].items():
        if not isinstance(path_item, dict): continue
        for method in ("get","post","put","patch","delete"):
            operation = path_item.get(method)
            if not isinstance(operation, dict): continue
            operation_id = str(operation.get("operationId") or f"{method}_{path.strip('/').replace('/','_')}")
            operations.append({"name":operation_id,"display_name":str(operation.get("summary") or operation_id),"description":str(operation.get("description") or operation.get("summary") or ""),"metadata":{"source":"openapi","adapter":"openapi_http","operation_id":operation_id,"operation_url":f"{base_url}{path}","http_method":method.upper(),"input_schema":operation.get("requestBody") or operation.get("parameters") or {},"credential_env":credential_env,"sync_status":"synced","risk":"read" if method=="get" else "medium"}})
    if not operations: raise ValueError("OpenAPI 文档没有可导入的 operation")
    return operations


def read_skill_zip(content: bytes) -> Dict[str, Any]:
    if len(content) > MAX_SKILL_BYTES: raise ValueError("Skill ZIP 超过 5MB 限制")
    with zipfile.ZipFile(BytesIO(content)) as archive:
        files = [item for item in archive.infolist() if not item.is_dir()]
        if len(files) > MAX_SKILL_FILES: raise ValueError("Skill 文件数量超过限制")
        safe: Dict[str,str] = {}
        for item in files:
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts: raise ValueError("Skill ZIP 包含非法路径")
            if item.file_size > 500_000: raise ValueError("Skill 单文件超过 500KB")
            suffix = path.suffix.lower()
            if suffix not in {".md",".txt",".json",".yaml",".yml"}: continue
            safe[path.as_posix()] = archive.read(item).decode("utf-8")
    skill_path = next((name for name in safe if name == "SKILL.md" or name.endswith("/SKILL.md")), None)
    if not skill_path: raise ValueError("Skill 包根目录缺少 SKILL.md")
    content_text = safe[skill_path]
    name = _skill_name(content_text) or Path(skill_path).parent.name or "Imported Skill"
    return {"name":name,"content":content_text,"references":{k:v for k,v in safe.items() if k!=skill_path},"sha256":hashlib.sha256(content).hexdigest()}


def read_skill_file(content: bytes, filename: str) -> Dict[str, Any]:
    """Read a supported Skill upload without executing any embedded content."""
    if len(content) > MAX_SKILL_BYTES:
        raise ValueError("Skill 文件超过 5MB 限制")
    suffix = Path(filename).suffix.lower()
    if suffix == ".zip":
        return read_skill_zip(content)
    if suffix != ".md":
        raise ValueError("仅支持 .zip 或 .md Skill 文件")
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("Skill Markdown 必须使用 UTF-8 编码") from exc
    if not text.strip() or not any(line.startswith("# ") for line in text.splitlines()):
        raise ValueError("Skill Markdown 必须包含一级标题")
    return {
        "name": _skill_name(text) or Path(filename).stem or "Imported Skill",
        "content": text,
        "references": {},
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def read_skill_git(url: str) -> Dict[str, Any]:
    validate_remote_url(url)
    with tempfile.TemporaryDirectory() as temp:
        completed = subprocess.run(["git","clone","--depth","1","--filter=blob:none",url,temp],capture_output=True,text=True,timeout=30,check=False)
        if completed.returncode != 0: raise ValueError(f"Git 仓库读取失败：{completed.stderr[-300:]}")
        root = Path(temp)
        candidates = list(root.glob("SKILL.md"))
        if not candidates: raise ValueError("Git 仓库根目录缺少 SKILL.md")
        content = candidates[0].read_text(encoding="utf-8")
        revision = subprocess.run(["git","-C",temp,"rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip()
        return {"name":_skill_name(content) or root.name,"content":content,"references":{},"sha256":revision}


def _skill_name(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "): return line[2:].strip()
    return ""

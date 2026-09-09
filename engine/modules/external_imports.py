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

from .tools.mcp_remote import discover_tools


MAX_REMOTE_BYTES = 2_000_000
MAX_SKILL_FILES = 100
MAX_SKILL_BYTES = 5_000_000
MAX_REFERENCE_FILES = 30
MAX_REFERENCE_BYTES = 300_000
MAX_SKILL_SCAN_DEPTH = 5


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


def discover_mcp_tools(url: str, credential_env: str = "", timeout: float = 8, access_token: str = "") -> List[Dict[str, Any]]:
    """Discover tools exposed by a remote MCP HTTP endpoint."""
    endpoint = validate_remote_url(url)
    token = access_token or (os.environ.get(credential_env, "") if credential_env else "")
    return discover_tools(endpoint, token=token, timeout=timeout)


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
    identity = skill_identity(content_text, fallback=Path(skill_path).parent.name or "Imported Skill")
    return {"name":identity["name"],"description":identity["description"],"content":content_text,"references":{k:v for k,v in safe.items() if k!=skill_path},"sha256":hashlib.sha256(content).hexdigest()}


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
    identity = skill_identity(text, fallback=Path(filename).stem or "Imported Skill")
    return {
        "name": identity["name"],
        "description": identity["description"],
        "content": text,
        "references": {},
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def read_skill_git(url: str, subdir: str = "") -> Dict[str, Any]:
    """Clone a Skill repository and read its ``SKILL.md``.

    ``subdir`` selects one Skill inside a monorepo (``skills/foo``).  When it is
    omitted we still accept a repository that ships ``SKILL.md`` at the root, and
    otherwise auto-discover a single Skill deeper in the tree.  Repositories with
    several Skills are rejected with the list of candidates so the caller can ask
    the user which one to import.
    """
    validate_remote_url(url)
    subdir = (subdir or "").strip().strip("/")
    with tempfile.TemporaryDirectory() as temp:
        completed = subprocess.run(["git","clone","--depth","1","--filter=blob:none",url,temp],capture_output=True,text=True,timeout=30,check=False)
        if completed.returncode != 0: raise ValueError(f"Git 仓库读取失败：{completed.stderr[-300:]}")
        root = Path(temp)
        target = root
        if subdir:
            if ".." in Path(subdir).parts or Path(subdir).is_absolute():
                raise ValueError("非法的子目录路径")
            target = root / subdir
            if not target.is_dir(): raise ValueError(f"仓库中不存在子目录：{subdir}")
        candidates = list(target.glob("SKILL.md"))
        if not candidates and not subdir:
            candidates = _discover_skill_files(root)
        if not candidates:
            raise ValueError("Git 仓库根目录缺少 SKILL.md，请填写 Skill 所在的子目录" if not subdir else f"子目录 {subdir} 下缺少 SKILL.md")
        skill_path = candidates[0]
        try:
            content = skill_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("SKILL.md 必须使用 UTF-8 编码") from exc
        revision = subprocess.run(["git","-C",temp,"rev-parse","HEAD"],capture_output=True,text=True,check=False).stdout.strip()
        identity = skill_identity(content, fallback=skill_path.parent.name or root.name)
        relative = skill_path.parent.relative_to(root).as_posix()
        return {"name":identity["name"],"description":identity["description"],"content":content,"references":collect_reference_docs(skill_path.parent),"sha256":revision,"source_path":"" if relative == "." else relative}


def _discover_skill_files(root: Path) -> List[Path]:
    """Find SKILL.md files below the repository root (monorepo layout)."""
    found: List[Path] = []
    for path in root.rglob("SKILL.md"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if len(path.relative_to(root).parts) > MAX_SKILL_SCAN_DEPTH:
            continue
        found.append(path)
    if len(found) > 1:
        options = sorted({path.parent.relative_to(root).as_posix() for path in found})
        preview = "、".join(options[:12])
        raise ValueError(f"该仓库包含 {len(found)} 个 Skill，请在「子目录」中指定要导入的一个，可选：{preview}")
    return found


def _skill_name(content: str) -> str:
    for line in content.splitlines():
        if line.startswith("# "): return line[2:].strip()
    return ""


def parse_skill_frontmatter(content: str) -> Dict[str, Any]:
    """Extract the YAML frontmatter block used by the open Agent Skills format.

    The upstream standard keeps ``name``/``description`` in frontmatter.  We keep
    the parser dependency-free and never raise: malformed blocks degrade to the
    legacy "first H1 title" behaviour instead of failing the import.
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}
    for index in range(1, len(lines)):
        if lines[index].strip() in {"---", "..."}:
            block = "\n".join(lines[1:index]).strip()
            if not block:
                return {}
            try:
                import yaml  # type: ignore

                data = yaml.safe_load(block)
            except Exception:
                return _parse_flat_frontmatter(block)
            return data if isinstance(data, dict) else {}
    return {}


def _parse_flat_frontmatter(block: str) -> Dict[str, Any]:
    """Minimal ``key: value`` fallback used when PyYAML is unavailable."""
    result: Dict[str, Any] = {}
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        result[key.strip()] = value.strip().strip("\"'")
    return result


def skill_identity(content: str, fallback: str = "") -> Dict[str, str]:
    """Resolve name/description from frontmatter first, then markdown body."""
    meta = parse_skill_frontmatter(content)
    name = str(meta.get("name") or "").strip() or _skill_name(content) or fallback
    description = str(meta.get("description") or "").strip()
    if not description:
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith(("#", "---")):
                description = stripped
                break
    return {"name": name, "description": description[:500]}


def collect_reference_docs(skill_dir: Path) -> Dict[str, str]:
    """Collect sibling markdown docs (``references/``, ``scripts/`` docs, ...)."""
    references: Dict[str, str] = {}
    try:
        candidates = sorted(path for path in skill_dir.rglob("*.md") if path.is_file())
    except OSError:
        return references
    for path in candidates:
        if path.name == "SKILL.md" or len(references) >= MAX_REFERENCE_FILES:
            continue
        try:
            if path.stat().st_size > MAX_REFERENCE_BYTES:
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError, ValueError):
            continue
        references[path.relative_to(skill_dir).as_posix()] = text
    return references

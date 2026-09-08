"""Persistent semantic index for published Skills."""
from __future__ import annotations
import hashlib, json, math, sqlite3, threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from ..memory.embedding import BGEM3EmbeddingModel, EmbeddingModel, HashingEmbeddingModel
from ..bge_local import resolve_bge_m3_model_path
from ._types import SkillRecord, SkillStatus

class SkillSemanticIndex:
    """SQLite vector index with a lazily loaded, replaceable embedder."""
    def __init__(self,path:str|Path,*,embedding_model:str="bge-m3",embedder:Optional[EmbeddingModel]=None)->None:
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True);self.embedding_model=embedding_model;self._embedder=embedder;self._load_error="";self._lock=threading.RLock()
        if embedder is None and embedding_model=="bge-m3" and resolve_bge_m3_model_path()=="BAAI/bge-m3":self._load_error="未找到本地 BGE-M3 模型，请配置 BGE_M3_MODEL_PATH 后重建索引"
        self.conn=sqlite3.connect(self.path,check_same_thread=False);self.conn.row_factory=sqlite3.Row
        with self.conn:self.conn.executescript("""CREATE TABLE IF NOT EXISTS skill_vectors(skill_id TEXT PRIMARY KEY,version INTEGER NOT NULL,content_hash TEXT NOT NULL,model TEXT NOT NULL,summary TEXT NOT NULL,embedding_json TEXT NOT NULL,status TEXT NOT NULL,error TEXT NOT NULL DEFAULT '',updated_at TEXT NOT NULL);CREATE INDEX IF NOT EXISTS idx_skill_vectors_status ON skill_vectors(status);""")
    def summary(self,skill:SkillRecord)->str:
        meta=skill.metadata;parts=[f"名称：{skill.name}",f"简介：{skill.description}",f"分类：{meta.get('category','')}","适用场景："+"、".join(str(x) for x in meta.get("scenarios") or []),"关键词："+"、".join(str(x) for x in meta.get("keywords") or []),"标签："+"、".join(skill.tags),skill.content[:2400]]
        return "\n".join(part for part in parts if part.split("：",1)[-1].strip())
    def upsert(self,skill:SkillRecord)->Dict[str,Any]:
        if skill.status!=SkillStatus.PUBLISHED:self.remove(skill.id);return {"skill_id":skill.id,"status":"not_published"}
        summary=self.summary(skill);digest=hashlib.sha256(f"{self.embedding_model}:{summary}".encode()).hexdigest();current=self.get(skill.id)
        if current and current["content_hash"]==digest and current["version"]==skill.version and current["status"]=="ready":return current
        try:vector=self._embedding().embed(summary);status,error="ready",""
        except Exception as exc:vector,status,error=[],"failed",str(exc)[:500];self._load_error=error
        now=datetime.now(timezone.utc).isoformat()
        with self._lock,self.conn:self.conn.execute("INSERT OR REPLACE INTO skill_vectors VALUES(?,?,?,?,?,?,?,?,?)",(skill.id,skill.version,digest,self.embedding_model,summary,json.dumps(vector),status,error,now))
        return self.get(skill.id) or {}
    def rebuild(self,skills:Iterable[SkillRecord])->Dict[str,int]:
        published=[x for x in skills if x.status==SkillStatus.PUBLISHED];ids={x.id for x in published}
        with self._lock,self.conn:
            if ids:self.conn.execute(f"DELETE FROM skill_vectors WHERE skill_id NOT IN ({','.join('?' for _ in ids)})",tuple(ids))
            else:self.conn.execute("DELETE FROM skill_vectors")
        rows=[self.upsert(x) for x in published];return {"total":len(rows),"ready":sum(x.get("status")=="ready" for x in rows),"failed":sum(x.get("status")=="failed" for x in rows)}
    def search(self,query:str,*,allowed_ids:Optional[set[str]]=None,limit:int=20)->List[Dict[str,Any]]:
        q=self._embedding().embed(query);results=[]
        for row in self.conn.execute("SELECT * FROM skill_vectors WHERE status='ready'"):
            if allowed_ids is not None and row["skill_id"] not in allowed_ids:continue
            score=_cosine(q,json.loads(row["embedding_json"]));results.append({"skill_id":row["skill_id"],"semantic_score":round(max(0,min(1,score)),6)})
        return sorted(results,key=lambda x:x["semantic_score"],reverse=True)[:max(1,limit)]
    def get(self,skill_id:str)->Optional[Dict[str,Any]]:
        row=self.conn.execute("SELECT * FROM skill_vectors WHERE skill_id=?",(skill_id,)).fetchone();return dict(row) if row else None
    def remove(self,skill_id:str)->None:
        with self._lock,self.conn:self.conn.execute("DELETE FROM skill_vectors WHERE skill_id=?",(skill_id,))
    def health(self)->Dict[str,Any]:
        counts={r["status"]:r["n"] for r in self.conn.execute("SELECT status,COUNT(*) n FROM skill_vectors GROUP BY status")};return {"model":self.embedding_model,"available":not bool(self._load_error),"load_error":self._load_error,"ready":counts.get("ready",0),"failed":counts.get("failed",0),"total":sum(counts.values())}
    def _embedding(self)->EmbeddingModel:
        if self._embedder is not None:return self._embedder
        if self._load_error:raise RuntimeError(self._load_error)
        if self.embedding_model=="hashing":self._embedder=HashingEmbeddingModel()
        elif self.embedding_model=="bge-m3":self._embedder=BGEM3EmbeddingModel()
        else:raise RuntimeError(f"不支持的 Skill Embedding 模型：{self.embedding_model}")
        return self._embedder

def _cosine(left:List[float],right:List[float])->float:
    if not left or not right or len(left)!=len(right):return 0.0
    denom=math.sqrt(sum(x*x for x in left))*math.sqrt(sum(x*x for x in right));return sum(x*y for x,y in zip(left,right))/denom if denom else 0.0

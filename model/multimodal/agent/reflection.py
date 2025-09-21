# agent/reflection.py
from pydantic import BaseModel, Field
from typing import List, Optional
import json
import time
from pathlib import Path

MEM_FILE = Path("./agent_memory.json")

class Reflection(BaseModel):
    missing: Optional[str] = Field("", description="누락되거나 부족한 부분")
    superfluous: Optional[str] = Field("", description="불필요한 부분")
    suggestion: Optional[str] = Field("", description="다음 시도에 반영할 개선안")

class Trajectory(BaseModel):
    timestamp: float
    command: str
    actor_output: str
    tool_outputs: List[str] = []
    evaluation: Optional[str] = None
    reflection: Optional[Reflection] = None

class Experience(BaseModel):
    reflections: List[Reflection] = []

# 간단한 장기메모리 로드/저장
def load_experience() -> Experience:
    if MEM_FILE.exists():
        try:
            j = json.loads(MEM_FILE.read_text(encoding="utf-8"))
            return Experience(**j)
        except Exception:
            return Experience(reflections=[])
    return Experience(reflections=[])

def save_experience(exp: Experience):
    MEM_FILE.write_text(exp.json(), encoding="utf-8")

def append_reflection(ref: Reflection):
    exp = load_experience()
    exp.reflections.append(ref)
    save_experience(exp)

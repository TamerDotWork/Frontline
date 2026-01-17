# models/project.py
from dataclasses import dataclass

@dataclass
class Project:
    project_id: str
    name: str
    provider: str
    model: str
    base_url: str | None
    created_at: str

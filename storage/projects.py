# storage/projects.py
from db import get_db
from models.project import Project

def create_project(project: Project):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO projects (project_id, name, provider, model, base_url, created_at)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (project.project_id, project.name, project.provider, project.model, project.base_url, project.created_at))
    conn.commit()
    conn.close()

def list_projects() -> list[dict]:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def get_project(project_id: str) -> dict | None:
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def update_project(project_id: str, data: dict):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
    UPDATE projects
    SET name = ?, provider = ?, model = ?, base_url = ?
    WHERE project_id = ?
    """, (data["name"], data["provider"], data["model"], data.get("base_url"), project_id))
    conn.commit()
    conn.close()

def delete_project(project_id: str):
    conn = get_db()
    cur = conn.cursor()
    cur.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
    conn.commit()
    conn.close()

# routes/projects.py
from fastapi import APIRouter, Form
from uuid import uuid4
from datetime import datetime
from fastapi.responses import JSONResponse

from models.project import Project
from storage.projects import create_project, list_projects, get_project, update_project, delete_project

router = APIRouter(prefix="/projects", tags=["Projects"])

@router.post("")
async def create_project_api(
    name: str = Form(...),
    provider: str = Form(...),
    model: str = Form(...),
    base_url: str = Form("")
):
    project = Project(
        project_id=str(uuid4()),
        name=name,
        provider=provider,
        model=model,
        base_url=base_url,
        created_at=datetime.utcnow().isoformat() + "Z"
    )
    create_project(project)
    return project

@router.get("")
async def list_projects_api():
    return list_projects()

@router.get("/{project_id}")
async def get_project_api(project_id: str):
    project = get_project(project_id)
    if not project:
        return JSONResponse({"error": "Project not found"}, status_code=404)
    return project

@router.put("/{project_id}")
async def update_project_api(
    project_id: str,
    name: str = Form(...),
    provider: str = Form(...),
    model: str = Form(...),
    base_url: str = Form("")
):
    if not get_project(project_id):
        return JSONResponse({"error": "Project not found"}, status_code=404)

    update_project(project_id, {
        "name": name,
        "provider": provider,
        "model": model,
        "base_url": base_url
    })
    return {"status": "updated"}

@router.delete("/{project_id}")
async def delete_project_api(project_id: str):
    if not get_project(project_id):
        return JSONResponse({"error": "Project not found"}, status_code=404)

    delete_project(project_id)
    return {"status": "deleted"}

from fastapi import APIRouter
from fastapi import Query
from models.input import InputModel
from typing import Optional

#APIRouter creates path operations for user module
router = APIRouter(
    prefix="/users",
    tags=["User"],
    responses={404: {"description": "Not found"}},
)

@router.post("/add")
async def sendInput(input: InputModel,):
    return {input.res}


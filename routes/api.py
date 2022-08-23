from fastapi import APIRouter
from endpoints import chats

router = APIRouter()
router.include_router(chats.router)

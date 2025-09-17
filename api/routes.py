from fastapi import APIRouter
from fastapi import Depends

from api.dependencies import get_association_service_v1
from api.dependencies import get_association_service_v2
from common import HumanResponse
from memory.services import MemoryServiceInterface

router = APIRouter()


@router.post('/chat', response_model=HumanResponse)
def chat(
        request: HumanResponse,
        service: MemoryServiceInterface = Depends(get_association_service_v2),
):
    return service.chat(request)

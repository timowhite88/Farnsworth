"""
FastAPI server setup to handle routes for quantum entanglement UI integration.
"""

from fastapi import APIRouter, HTTPException
from typing import List
from .quantum_entanglement import generate_entangled_ui_elements, integrate_entanglement

router = APIRouter()

@router.get("/ui/entangle/{theme}")
async def get_entangled_ui(theme: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Endpoint to retrieve entangled UI elements based on the theme.

    :param theme: The theme for which to generate entangled UI elements.
    :return: Dictionary containing list of entangled UI elements.
    """
    try:
        ui_elements = await generate_entangled_ui_elements(theme)
        return {"ui_elements": ui_elements}
    
    except Exception as e:
        logger.error(f"Error retrieving UI elements for theme {theme}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@router.post("/ui/entangle/{theme}")
async def integrate_theme(theme: str) -> Dict[str, str]:
    """
    Endpoint to integrate quantum entanglement concepts into the UI based on the theme.

    :param theme: The theme to apply quantum entanglement concepts.
    :return: Dictionary with confirmation message.
    """
    try:
        await integrate_entanglement(theme)
        return {"message": "Quantum entanglement integrated successfully"}
    
    except Exception as e:
        logger.error(f"Error integrating theme {theme}: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
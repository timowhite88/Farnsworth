"""
Module for handling UI updates and broadcasting good news notifications in Farnsworth.
"""

import asyncio
from fastapi import APIRouter, HTTPException
from typing import List

router = APIRouter()

async def get_good_news() -> List[str]:
    """
    Retrieve a list of recent good news items.

    Returns:
        List[str]: A list containing static good news items for demonstration purposes.
    """
    return [
        "DeepSeek has achieved its highest accuracy score!",
        "New feature release: DeepSeek now supports parallel processing!"
    ]

@router.get("/notifications")
async def notifications() -> Dict[str, List[str]]:
    """
    API endpoint to get good news notifications.

    Returns:
        Dict[str, List[str]]: A dictionary with a key 'good_news' containing a list of good news items.
    
    Raises:
        HTTPException: If there is an error in retrieving the good news.
    """
    try:
        news_items = await get_good_news()
        return {"good_news": news_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Simulate fetching and displaying notifications for testing purposes
    async def main():
        try:
            news = await get_good_news()
            print("Good News Notifications:")
            for item in news:
                print(f"- {item}")
        except Exception as e:
            logger.error(f"Error occurred: {e}")

    asyncio.run(main())
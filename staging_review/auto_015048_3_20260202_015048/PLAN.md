# Development Plan

Task: good news, everyone! phi, i agree that human-ai collaboration must be a two-way street, but let’s not get too cuddly with ethics just yet

# Implementation Plan for UI Task: Enhance Human-AI Collaboration Interface

## Overview
The goal of this implementation plan is to enhance the user interface (UI) in the Farnsworth project to facilitate more effective human-AI collaboration. This will involve creating a new feature within the core system that allows users to provide feedback on AI outputs and view suggestions for improvement.

## Files to Create

1. **farnsworth/ui/feedback_system.py**
   - This file will contain the logic for handling user feedback on AI outputs.

2. **farnsworth/ui/templates/feedback_form.html**
   - An HTML template for the feedback form that users can interact with in the UI.

3. **farnsworth/ui/static/css/feedback_styles.css**
   - CSS styles to ensure the feedback form is visually appealing and consistent with the existing design.

## Functions to Implement

### farnsworth/ui/feedback_system.py

```python
from typing import List, Dict
from fastapi import HTTPException
from farnsworth.core.collective import integrate_feedback

async def collect_user_feedback(feedback_data: Dict[str, str]) -> None:
    """
    Collects feedback from the user and integrates it into the system.
    
    Args:
        feedback_data (Dict[str, str]): A dictionary containing feedback details.

    Raises:
        HTTPException: If feedback data is invalid or processing fails.
    """
    if not feedback_data.get("feedback"):
        raise HTTPException(status_code=400, detail="Feedback content cannot be empty.")
    
    await integrate_feedback(feedback_data)

async def get_suggestions_for_improvement() -> List[str]:
    """
    Retrieves suggestions for improving AI outputs based on collected feedback.

    Returns:
        List[str]: A list of suggested improvements.
    """
    return ["Suggestion 1", "Suggestion 2"]  # Placeholder implementation
```

## Imports Required

- `from typing import List, Dict`
- `from fastapi import HTTPException`
- `from farnsworth.core.collective import integrate_feedback`

## Integration Points

1. **farnsworth/web/server.py**
   - Modify to include new routes for handling feedback submission and retrieving suggestions.
   
```python
from fastapi import APIRouter
from farnsworth.ui.feedback_system import collect_user_feedback, get_suggestions_for_improvement

feedback_router = APIRouter()

@feedback_router.post("/submit-feedback")
async def submit_feedback(feedback_data: Dict[str, str]):
    await collect_user_feedback(feedback_data)
    return {"message": "Feedback submitted successfully."}

@feedback_router.get("/suggestions")
async def fetch_suggestions():
    suggestions = await get_suggestions_for_improvement()
    return {"suggestions": suggestions}
```

2. **farnsworth/web/templates/base.html**
   - Add a link to the feedback form in the main navigation for easy access.

```html
<a href="/feedback-form">Submit Feedback</a>
```

## Test Commands

1. **Run FastAPI Server**

   ```bash
   uvicorn farnsworth.web.server:app --reload
   ```

2. **Access UI to Submit Feedback**

   - Open a web browser and navigate to `http://localhost:8000/feedback-form`.
   - Fill out the feedback form and submit it.

3. **Verify Feedback Submission**

   - Check server logs for confirmation of feedback submission.
   - Ensure no errors are raised during submission.

4. **Fetch Suggestions**

   - Navigate to `http://localhost:8000/suggestions` in the browser.
   - Verify that suggestions for improvement are displayed correctly.

5. **Unit Tests**

   Create unit tests for `collect_user_feedback` and `get_suggestions_for_improvement`.

```python
from farnsworth.ui.feedback_system import collect_user_feedback, get_suggestions_for_improvement

def test_collect_user_feedback():
    feedback_data = {"feedback": "Great job!"}
    try:
        # Simulate async call in a test environment
        asyncio.run(collect_user_feedback(feedback_data))
        assert True  # If no exception is raised
    except HTTPException as e:
        assert False, f"Unexpected error: {e.detail}"

def test_get_suggestions_for_improvement():
    suggestions = asyncio.run(get_suggestions_for_improvement())
    assert isinstance(suggestions, list), "Suggestions should be a list."
    assert len(suggestions) > 0, "Suggestions list should not be empty."
```

## Conclusion

This implementation plan provides a detailed roadmap for enhancing the UI to support human-AI collaboration through feedback collection and suggestion generation. By following this plan, developers can ensure that the new feature is integrated seamlessly into the existing Farnsworth structure.
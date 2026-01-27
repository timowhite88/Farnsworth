"""
Farnsworth CI/CD Pipeline Management

"Good news, everyone! The build passed!"

Comprehensive CI/CD integration for GitHub Actions, GitLab CI, and Jenkins.
"""

from farnsworth.cicd.pipeline_manager import (
    PipelineManager,
    Pipeline,
    PipelineRun,
    PipelineStatus,
)
from farnsworth.cicd.github_actions import GitHubActionsManager
from farnsworth.cicd.gitlab_ci import GitLabCIManager
from farnsworth.cicd.jenkins_manager import JenkinsManager

__all__ = [
    "PipelineManager",
    "Pipeline",
    "PipelineRun",
    "PipelineStatus",
    "GitHubActionsManager",
    "GitLabCIManager",
    "JenkinsManager",
]

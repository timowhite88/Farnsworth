"""
Farnsworth GitHub Integration - Full-Featured Implementation.

"The Code must flow! And I shall guide it."

Complete GitHub API integration:
- Issue management (CRUD, labels, assignees)
- Pull request operations (create, review, merge)
- Repository management
- Branch operations
- Commit and file operations
- Workflow/Actions management
- Release management
- Webhook handling
- Organization support
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

try:
    from github import Github, Auth
    from github.GithubException import GithubException
    from github.Repository import Repository
    from github.Issue import Issue
    from github.PullRequest import PullRequest
    GITHUB_AVAILABLE = True
except ImportError:
    GITHUB_AVAILABLE = False
    logger.warning("PyGithub not installed. Run: pip install PyGithub")

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus
from farnsworth.core.nexus import nexus, Signal, SignalType


class IssueState(Enum):
    """Issue state options."""
    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


class PRState(Enum):
    """Pull request state options."""
    OPEN = "open"
    CLOSED = "closed"
    ALL = "all"


class MergeMethod(Enum):
    """PR merge methods."""
    MERGE = "merge"
    SQUASH = "squash"
    REBASE = "rebase"


class ReviewState(Enum):
    """PR review states."""
    APPROVE = "APPROVE"
    REQUEST_CHANGES = "REQUEST_CHANGES"
    COMMENT = "COMMENT"


@dataclass
class GitHubIssue:
    """Structured GitHub issue data."""
    number: int
    title: str
    body: str = ""
    state: IssueState = IssueState.OPEN
    labels: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    milestone: Optional[str] = None
    author: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    comments: int = 0
    html_url: str = ""

    def to_dict(self) -> Dict:
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body[:500] + "..." if len(self.body) > 500 else self.body,
            "state": self.state.value,
            "labels": self.labels,
            "assignees": self.assignees,
            "author": self.author,
            "comments": self.comments,
            "url": self.html_url,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class GitHubPR:
    """Structured GitHub pull request data."""
    number: int
    title: str
    body: str = ""
    state: PRState = PRState.OPEN
    head_branch: str = ""
    base_branch: str = ""
    author: str = ""
    labels: List[str] = field(default_factory=list)
    reviewers: List[str] = field(default_factory=list)
    assignees: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    merged_at: Optional[datetime] = None
    draft: bool = False
    mergeable: Optional[bool] = None
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    html_url: str = ""

    @property
    def is_merged(self) -> bool:
        return self.merged_at is not None

    def to_dict(self) -> Dict:
        return {
            "number": self.number,
            "title": self.title,
            "body": self.body[:500] + "..." if len(self.body) > 500 else self.body,
            "state": self.state.value,
            "head": self.head_branch,
            "base": self.base_branch,
            "author": self.author,
            "labels": self.labels,
            "draft": self.draft,
            "mergeable": self.mergeable,
            "is_merged": self.is_merged,
            "additions": self.additions,
            "deletions": self.deletions,
            "changed_files": self.changed_files,
            "url": self.html_url,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


@dataclass
class GitHubCommit:
    """Structured GitHub commit data."""
    sha: str
    message: str
    author: str = ""
    date: Optional[datetime] = None
    additions: int = 0
    deletions: int = 0
    files_changed: List[str] = field(default_factory=list)
    html_url: str = ""

    def to_dict(self) -> Dict:
        return {
            "sha": self.sha[:7],
            "message": self.message.split('\n')[0],  # First line only
            "author": self.author,
            "date": self.date.isoformat() if self.date else None,
            "additions": self.additions,
            "deletions": self.deletions,
            "files_changed": len(self.files_changed),
            "url": self.html_url
        }


@dataclass
class GitHubRelease:
    """Structured GitHub release data."""
    id: int
    tag_name: str
    name: str
    body: str = ""
    draft: bool = False
    prerelease: bool = False
    created_at: Optional[datetime] = None
    published_at: Optional[datetime] = None
    author: str = ""
    assets: List[Dict] = field(default_factory=list)
    html_url: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "tag": self.tag_name,
            "name": self.name,
            "body": self.body[:500] + "..." if len(self.body) > 500 else self.body,
            "draft": self.draft,
            "prerelease": self.prerelease,
            "author": self.author,
            "assets": len(self.assets),
            "url": self.html_url,
            "published_at": self.published_at.isoformat() if self.published_at else None
        }


@dataclass
class GitHubWorkflowRun:
    """Structured GitHub Actions workflow run data."""
    id: int
    name: str
    status: str  # queued, in_progress, completed
    conclusion: Optional[str] = None  # success, failure, cancelled, etc.
    head_branch: str = ""
    head_sha: str = ""
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    html_url: str = ""

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "conclusion": self.conclusion,
            "branch": self.head_branch,
            "sha": self.head_sha[:7] if self.head_sha else "",
            "url": self.html_url,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class GitHubProvider(ExternalProvider):
    """
    Full-featured GitHub API integration.

    Features:
    - Complete issue management
    - Pull request operations
    - Repository management
    - Branch operations
    - Commit and file operations
    - Actions/workflow management
    - Release management
    - Webhook support
    """

    def __init__(self, token: str, repo: str = None):
        super().__init__(IntegrationConfig(name="github", api_key=token))
        self.repo_name = repo
        self.client: Optional[Github] = None
        self.repo_obj = None
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 60  # seconds

    async def connect(self) -> bool:
        """Connect to GitHub API."""
        if not GITHUB_AVAILABLE:
            logger.error("GitHub: PyGithub package not installed")
            return False

        if not self.config.api_key:
            logger.warning("GitHub: No token provided")
            return False

        loop = asyncio.get_event_loop()

        try:
            auth = Auth.Token(self.config.api_key)
            self.client = Github(auth=auth)

            # Verify access
            await loop.run_in_executor(None, lambda: self.client.get_user().login)

            # Get repo if specified
            if self.repo_name:
                self.repo_obj = await loop.run_in_executor(
                    None, lambda: self.client.get_repo(self.repo_name)
                )
                logger.info(f"GitHub: Connected to {self.repo_name}")
            else:
                logger.info("GitHub: Connected (no default repo)")

            self.status = ConnectionStatus.CONNECTED
            return True
        except Exception as e:
            logger.error(f"GitHub connection failed: {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def disconnect(self):
        """Disconnect from GitHub."""
        if self.client:
            self.client.close()
        self.client = None
        self.repo_obj = None
        self._cache.clear()
        self.status = ConnectionStatus.DISCONNECTED

    async def sync(self):
        """Poll for new issues and PRs."""
        if self.status != ConnectionStatus.CONNECTED or not self.repo_obj:
            return

        loop = asyncio.get_event_loop()
        try:
            # Poll open issues
            issues = await loop.run_in_executor(
                None, lambda: list(self.repo_obj.get_issues(state='open')[:10])
            )

            for issue in issues:
                # Emit as signal
                await nexus.emit(
                    SignalType.EXTERNAL_ALERT,
                    {
                        "source": "github",
                        "type": "issue" if issue.pull_request is None else "pr",
                        "id": issue.number,
                        "title": issue.title,
                        "url": issue.html_url
                    },
                    source="github_provider"
                )

            logger.debug(f"GitHub: Synced {len(issues)} issues/PRs")
        except Exception as e:
            logger.error(f"GitHub sync error: {e}")

    def set_repo(self, repo_name: str):
        """Set the default repository."""
        self.repo_name = repo_name
        if self.client:
            try:
                self.repo_obj = self.client.get_repo(repo_name)
                logger.info(f"GitHub: Switched to {repo_name}")
            except Exception as e:
                logger.error(f"GitHub: Failed to switch repo: {e}")

    # ==================== ISSUES ====================

    async def list_issues(
        self,
        state: IssueState = IssueState.OPEN,
        labels: List[str] = None,
        assignee: str = None,
        since: datetime = None,
        limit: int = 30
    ) -> List[GitHubIssue]:
        """List repository issues."""
        if not self.repo_obj:
            return []

        loop = asyncio.get_event_loop()

        try:
            params = {"state": state.value}
            if labels:
                params["labels"] = labels
            if assignee:
                params["assignee"] = assignee
            if since:
                params["since"] = since

            issues = await loop.run_in_executor(
                None,
                lambda: list(self.repo_obj.get_issues(**params)[:limit])
            )

            # Filter out PRs
            return [self._parse_issue(i) for i in issues if i.pull_request is None]
        except Exception as e:
            logger.error(f"List issues error: {e}")
            return []

    async def get_issue(self, number: int) -> Optional[GitHubIssue]:
        """Get a specific issue."""
        if not self.repo_obj:
            return None

        loop = asyncio.get_event_loop()
        try:
            issue = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_issue(number)
            )
            return self._parse_issue(issue)
        except Exception as e:
            logger.error(f"Get issue error: {e}")
            return None

    async def create_issue(
        self,
        title: str,
        body: str = "",
        labels: List[str] = None,
        assignees: List[str] = None,
        milestone: int = None
    ) -> Optional[GitHubIssue]:
        """Create a new issue."""
        if not self.repo_obj:
            return None

        loop = asyncio.get_event_loop()
        try:
            params = {"title": title, "body": body}
            if labels:
                params["labels"] = labels
            if assignees:
                params["assignees"] = assignees
            if milestone:
                params["milestone"] = self.repo_obj.get_milestone(milestone)

            issue = await loop.run_in_executor(
                None, lambda: self.repo_obj.create_issue(**params)
            )
            logger.info(f"Created issue #{issue.number}: {title}")
            return self._parse_issue(issue)
        except Exception as e:
            logger.error(f"Create issue error: {e}")
            return None

    async def update_issue(
        self,
        number: int,
        title: str = None,
        body: str = None,
        state: str = None,
        labels: List[str] = None,
        assignees: List[str] = None
    ) -> Optional[GitHubIssue]:
        """Update an issue."""
        if not self.repo_obj:
            return None

        loop = asyncio.get_event_loop()
        try:
            issue = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_issue(number)
            )

            params = {}
            if title is not None:
                params["title"] = title
            if body is not None:
                params["body"] = body
            if state is not None:
                params["state"] = state
            if labels is not None:
                params["labels"] = labels
            if assignees is not None:
                params["assignees"] = assignees

            await loop.run_in_executor(None, lambda: issue.edit(**params))
            return self._parse_issue(issue)
        except Exception as e:
            logger.error(f"Update issue error: {e}")
            return None

    async def close_issue(self, number: int, comment: str = None) -> bool:
        """Close an issue."""
        if not self.repo_obj:
            return False

        loop = asyncio.get_event_loop()
        try:
            issue = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_issue(number)
            )

            if comment:
                await loop.run_in_executor(
                    None, lambda: issue.create_comment(comment)
                )

            await loop.run_in_executor(None, lambda: issue.edit(state="closed"))
            logger.info(f"Closed issue #{number}")
            return True
        except Exception as e:
            logger.error(f"Close issue error: {e}")
            return False

    async def add_issue_comment(self, number: int, body: str) -> bool:
        """Add a comment to an issue."""
        if not self.repo_obj:
            return False

        loop = asyncio.get_event_loop()
        try:
            issue = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_issue(number)
            )
            await loop.run_in_executor(None, lambda: issue.create_comment(body))
            return True
        except Exception as e:
            logger.error(f"Add comment error: {e}")
            return False

    # ==================== PULL REQUESTS ====================

    async def list_prs(
        self,
        state: PRState = PRState.OPEN,
        base: str = None,
        head: str = None,
        limit: int = 30
    ) -> List[GitHubPR]:
        """List pull requests."""
        if not self.repo_obj:
            return []

        loop = asyncio.get_event_loop()

        try:
            params = {"state": state.value}
            if base:
                params["base"] = base
            if head:
                params["head"] = head

            prs = await loop.run_in_executor(
                None,
                lambda: list(self.repo_obj.get_pulls(**params)[:limit])
            )

            return [self._parse_pr(pr) for pr in prs]
        except Exception as e:
            logger.error(f"List PRs error: {e}")
            return []

    async def get_pr(self, number: int) -> Optional[GitHubPR]:
        """Get a specific pull request."""
        if not self.repo_obj:
            return None

        loop = asyncio.get_event_loop()
        try:
            pr = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_pull(number)
            )
            return self._parse_pr(pr)
        except Exception as e:
            logger.error(f"Get PR error: {e}")
            return None

    async def create_pr(
        self,
        title: str,
        head: str,
        base: str = "main",
        body: str = "",
        draft: bool = False,
        maintainer_can_modify: bool = True
    ) -> Optional[GitHubPR]:
        """Create a pull request."""
        if not self.repo_obj:
            return None

        loop = asyncio.get_event_loop()
        try:
            pr = await loop.run_in_executor(
                None,
                lambda: self.repo_obj.create_pull(
                    title=title,
                    body=body,
                    head=head,
                    base=base,
                    draft=draft,
                    maintainer_can_modify=maintainer_can_modify
                )
            )
            logger.info(f"Created PR #{pr.number}: {title}")
            return self._parse_pr(pr)
        except Exception as e:
            logger.error(f"Create PR error: {e}")
            return None

    async def merge_pr(
        self,
        number: int,
        commit_title: str = None,
        commit_message: str = None,
        merge_method: MergeMethod = MergeMethod.SQUASH
    ) -> bool:
        """Merge a pull request."""
        if not self.repo_obj:
            return False

        loop = asyncio.get_event_loop()
        try:
            pr = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_pull(number)
            )

            await loop.run_in_executor(
                None,
                lambda: pr.merge(
                    commit_title=commit_title,
                    commit_message=commit_message,
                    merge_method=merge_method.value
                )
            )
            logger.info(f"Merged PR #{number}")
            return True
        except Exception as e:
            logger.error(f"Merge PR error: {e}")
            return False

    async def review_pr(
        self,
        number: int,
        body: str,
        event: ReviewState = ReviewState.COMMENT,
        comments: List[Dict] = None
    ) -> bool:
        """Submit a PR review."""
        if not self.repo_obj:
            return False

        loop = asyncio.get_event_loop()
        try:
            pr = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_pull(number)
            )

            await loop.run_in_executor(
                None,
                lambda: pr.create_review(body=body, event=event.value, comments=comments or [])
            )
            logger.info(f"Reviewed PR #{number}: {event.value}")
            return True
        except Exception as e:
            logger.error(f"Review PR error: {e}")
            return False

    async def request_reviewers(
        self,
        number: int,
        reviewers: List[str] = None,
        team_reviewers: List[str] = None
    ) -> bool:
        """Request reviewers for a PR."""
        if not self.repo_obj:
            return False

        loop = asyncio.get_event_loop()
        try:
            pr = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_pull(number)
            )

            await loop.run_in_executor(
                None,
                lambda: pr.create_review_request(
                    reviewers=reviewers or [],
                    team_reviewers=team_reviewers or []
                )
            )
            return True
        except Exception as e:
            logger.error(f"Request reviewers error: {e}")
            return False

    async def get_pr_files(self, number: int) -> List[Dict]:
        """Get files changed in a PR."""
        if not self.repo_obj:
            return []

        loop = asyncio.get_event_loop()
        try:
            pr = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_pull(number)
            )
            files = await loop.run_in_executor(
                None, lambda: list(pr.get_files())
            )

            return [{
                "filename": f.filename,
                "status": f.status,
                "additions": f.additions,
                "deletions": f.deletions,
                "changes": f.changes,
                "patch": f.patch[:1000] if f.patch else None
            } for f in files]
        except Exception as e:
            logger.error(f"Get PR files error: {e}")
            return []

    # ==================== BRANCHES ====================

    async def list_branches(self, protected_only: bool = False) -> List[Dict]:
        """List repository branches."""
        if not self.repo_obj:
            return []

        loop = asyncio.get_event_loop()
        try:
            branches = await loop.run_in_executor(
                None, lambda: list(self.repo_obj.get_branches())
            )

            result = []
            for b in branches:
                if protected_only and not b.protected:
                    continue
                result.append({
                    "name": b.name,
                    "protected": b.protected,
                    "sha": b.commit.sha[:7]
                })

            return result
        except Exception as e:
            logger.error(f"List branches error: {e}")
            return []

    async def create_branch(self, name: str, from_branch: str = "main") -> bool:
        """Create a new branch."""
        if not self.repo_obj:
            return False

        loop = asyncio.get_event_loop()
        try:
            source = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_branch(from_branch)
            )

            await loop.run_in_executor(
                None,
                lambda: self.repo_obj.create_git_ref(
                    ref=f"refs/heads/{name}",
                    sha=source.commit.sha
                )
            )
            logger.info(f"Created branch: {name}")
            return True
        except Exception as e:
            logger.error(f"Create branch error: {e}")
            return False

    async def delete_branch(self, name: str) -> bool:
        """Delete a branch."""
        if not self.repo_obj:
            return False

        loop = asyncio.get_event_loop()
        try:
            ref = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_git_ref(f"heads/{name}")
            )
            await loop.run_in_executor(None, lambda: ref.delete())
            logger.info(f"Deleted branch: {name}")
            return True
        except Exception as e:
            logger.error(f"Delete branch error: {e}")
            return False

    # ==================== COMMITS ====================

    async def list_commits(
        self,
        branch: str = None,
        since: datetime = None,
        until: datetime = None,
        author: str = None,
        limit: int = 30
    ) -> List[GitHubCommit]:
        """List repository commits."""
        if not self.repo_obj:
            return []

        loop = asyncio.get_event_loop()

        try:
            params = {}
            if branch:
                params["sha"] = branch
            if since:
                params["since"] = since
            if until:
                params["until"] = until
            if author:
                params["author"] = author

            commits = await loop.run_in_executor(
                None,
                lambda: list(self.repo_obj.get_commits(**params)[:limit])
            )

            return [self._parse_commit(c) for c in commits]
        except Exception as e:
            logger.error(f"List commits error: {e}")
            return []

    async def get_commit(self, sha: str) -> Optional[GitHubCommit]:
        """Get a specific commit."""
        if not self.repo_obj:
            return None

        loop = asyncio.get_event_loop()
        try:
            commit = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_commit(sha)
            )
            return self._parse_commit(commit)
        except Exception as e:
            logger.error(f"Get commit error: {e}")
            return None

    # ==================== FILES ====================

    async def get_file_contents(self, path: str, ref: str = None) -> Optional[str]:
        """Get file contents from the repository."""
        if not self.repo_obj:
            return None

        loop = asyncio.get_event_loop()
        try:
            content = await loop.run_in_executor(
                None,
                lambda: self.repo_obj.get_contents(path, ref=ref)
            )
            return content.decoded_content.decode('utf-8')
        except Exception as e:
            logger.error(f"Get file error: {e}")
            return None

    async def create_or_update_file(
        self,
        path: str,
        content: str,
        message: str,
        branch: str = "main",
        sha: str = None
    ) -> bool:
        """Create or update a file in the repository."""
        if not self.repo_obj:
            return False

        loop = asyncio.get_event_loop()
        try:
            # Check if file exists
            existing_sha = sha
            if not existing_sha:
                try:
                    existing = await loop.run_in_executor(
                        None,
                        lambda: self.repo_obj.get_contents(path, ref=branch)
                    )
                    existing_sha = existing.sha
                except:
                    pass

            if existing_sha:
                # Update
                await loop.run_in_executor(
                    None,
                    lambda: self.repo_obj.update_file(
                        path, message, content, existing_sha, branch=branch
                    )
                )
            else:
                # Create
                await loop.run_in_executor(
                    None,
                    lambda: self.repo_obj.create_file(path, message, content, branch=branch)
                )

            logger.info(f"Updated file: {path}")
            return True
        except Exception as e:
            logger.error(f"Create/update file error: {e}")
            return False

    async def delete_file(
        self,
        path: str,
        message: str,
        branch: str = "main"
    ) -> bool:
        """Delete a file from the repository."""
        if not self.repo_obj:
            return False

        loop = asyncio.get_event_loop()
        try:
            content = await loop.run_in_executor(
                None,
                lambda: self.repo_obj.get_contents(path, ref=branch)
            )

            await loop.run_in_executor(
                None,
                lambda: self.repo_obj.delete_file(path, message, content.sha, branch=branch)
            )
            logger.info(f"Deleted file: {path}")
            return True
        except Exception as e:
            logger.error(f"Delete file error: {e}")
            return False

    # ==================== RELEASES ====================

    async def list_releases(self, limit: int = 10) -> List[GitHubRelease]:
        """List repository releases."""
        if not self.repo_obj:
            return []

        loop = asyncio.get_event_loop()
        try:
            releases = await loop.run_in_executor(
                None,
                lambda: list(self.repo_obj.get_releases()[:limit])
            )
            return [self._parse_release(r) for r in releases]
        except Exception as e:
            logger.error(f"List releases error: {e}")
            return []

    async def create_release(
        self,
        tag_name: str,
        name: str = None,
        body: str = "",
        draft: bool = False,
        prerelease: bool = False,
        target_commitish: str = "main"
    ) -> Optional[GitHubRelease]:
        """Create a new release."""
        if not self.repo_obj:
            return None

        loop = asyncio.get_event_loop()
        try:
            release = await loop.run_in_executor(
                None,
                lambda: self.repo_obj.create_git_release(
                    tag=tag_name,
                    name=name or tag_name,
                    message=body,
                    draft=draft,
                    prerelease=prerelease,
                    target_commitish=target_commitish
                )
            )
            logger.info(f"Created release: {tag_name}")
            return self._parse_release(release)
        except Exception as e:
            logger.error(f"Create release error: {e}")
            return None

    # ==================== WORKFLOWS (ACTIONS) ====================

    async def list_workflows(self) -> List[Dict]:
        """List repository workflows."""
        if not self.repo_obj:
            return []

        loop = asyncio.get_event_loop()
        try:
            workflows = await loop.run_in_executor(
                None,
                lambda: list(self.repo_obj.get_workflows())
            )
            return [{
                "id": w.id,
                "name": w.name,
                "path": w.path,
                "state": w.state
            } for w in workflows]
        except Exception as e:
            logger.error(f"List workflows error: {e}")
            return []

    async def list_workflow_runs(
        self,
        workflow_id: int = None,
        branch: str = None,
        status: str = None,
        limit: int = 20
    ) -> List[GitHubWorkflowRun]:
        """List workflow runs."""
        if not self.repo_obj:
            return []

        loop = asyncio.get_event_loop()
        try:
            params = {}
            if branch:
                params["branch"] = branch
            if status:
                params["status"] = status

            if workflow_id:
                workflow = await loop.run_in_executor(
                    None, lambda: self.repo_obj.get_workflow(workflow_id)
                )
                runs = await loop.run_in_executor(
                    None, lambda: list(workflow.get_runs(**params)[:limit])
                )
            else:
                runs = await loop.run_in_executor(
                    None, lambda: list(self.repo_obj.get_workflow_runs(**params)[:limit])
                )

            return [self._parse_workflow_run(r) for r in runs]
        except Exception as e:
            logger.error(f"List workflow runs error: {e}")
            return []

    async def trigger_workflow(
        self,
        workflow_id: int,
        ref: str = "main",
        inputs: Dict = None
    ) -> bool:
        """Trigger a workflow dispatch."""
        if not self.repo_obj:
            return False

        loop = asyncio.get_event_loop()
        try:
            workflow = await loop.run_in_executor(
                None, lambda: self.repo_obj.get_workflow(workflow_id)
            )

            await loop.run_in_executor(
                None,
                lambda: workflow.create_dispatch(ref=ref, inputs=inputs or {})
            )
            logger.info(f"Triggered workflow: {workflow_id}")
            return True
        except Exception as e:
            logger.error(f"Trigger workflow error: {e}")
            return False

    # ==================== UTILITIES ====================

    def _parse_issue(self, issue) -> GitHubIssue:
        """Parse Issue object into GitHubIssue."""
        return GitHubIssue(
            number=issue.number,
            title=issue.title,
            body=issue.body or "",
            state=IssueState(issue.state),
            labels=[l.name for l in issue.labels],
            assignees=[a.login for a in issue.assignees],
            milestone=issue.milestone.title if issue.milestone else None,
            author=issue.user.login if issue.user else "",
            created_at=issue.created_at,
            updated_at=issue.updated_at,
            closed_at=issue.closed_at,
            comments=issue.comments,
            html_url=issue.html_url
        )

    def _parse_pr(self, pr) -> GitHubPR:
        """Parse PullRequest object into GitHubPR."""
        return GitHubPR(
            number=pr.number,
            title=pr.title,
            body=pr.body or "",
            state=PRState(pr.state),
            head_branch=pr.head.ref,
            base_branch=pr.base.ref,
            author=pr.user.login if pr.user else "",
            labels=[l.name for l in pr.labels],
            reviewers=[r.login for r in pr.requested_reviewers],
            assignees=[a.login for a in pr.assignees],
            created_at=pr.created_at,
            updated_at=pr.updated_at,
            merged_at=pr.merged_at,
            draft=pr.draft,
            mergeable=pr.mergeable,
            additions=pr.additions,
            deletions=pr.deletions,
            changed_files=pr.changed_files,
            html_url=pr.html_url
        )

    def _parse_commit(self, commit) -> GitHubCommit:
        """Parse Commit object into GitHubCommit."""
        return GitHubCommit(
            sha=commit.sha,
            message=commit.commit.message,
            author=commit.commit.author.name if commit.commit.author else "",
            date=commit.commit.author.date if commit.commit.author else None,
            additions=commit.stats.additions if commit.stats else 0,
            deletions=commit.stats.deletions if commit.stats else 0,
            files_changed=[f.filename for f in commit.files] if commit.files else [],
            html_url=commit.html_url
        )

    def _parse_release(self, release) -> GitHubRelease:
        """Parse Release object into GitHubRelease."""
        return GitHubRelease(
            id=release.id,
            tag_name=release.tag_name,
            name=release.title or release.tag_name,
            body=release.body or "",
            draft=release.draft,
            prerelease=release.prerelease,
            created_at=release.created_at,
            published_at=release.published_at,
            author=release.author.login if release.author else "",
            assets=[{"name": a.name, "size": a.size, "url": a.browser_download_url} for a in release.get_assets()],
            html_url=release.html_url
        )

    def _parse_workflow_run(self, run) -> GitHubWorkflowRun:
        """Parse WorkflowRun object into GitHubWorkflowRun."""
        return GitHubWorkflowRun(
            id=run.id,
            name=run.name,
            status=run.status,
            conclusion=run.conclusion,
            head_branch=run.head_branch,
            head_sha=run.head_sha,
            created_at=run.created_at,
            updated_at=run.updated_at,
            html_url=run.html_url
        )

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute an action (legacy interface)."""
        if self.status != ConnectionStatus.CONNECTED:
            raise ConnectionError("GitHub not connected")

        action_map = {
            "list_issues": lambda p: self.list_issues(limit=p.get("limit", 10)),
            "create_issue": lambda p: self.create_issue(
                p.get("title"), p.get("body", ""), p.get("labels"), p.get("assignees")
            ),
            "close_issue": lambda p: self.close_issue(p.get("number"), p.get("comment")),
            "list_prs": lambda p: self.list_prs(limit=p.get("limit", 10)),
            "create_pr": lambda p: self.create_pr(
                p.get("title"), p.get("head"), p.get("base", "main"), p.get("body", "")
            ),
            "merge_pr": lambda p: self.merge_pr(p.get("number")),
            "list_branches": lambda p: self.list_branches(),
            "list_commits": lambda p: self.list_commits(limit=p.get("limit", 10)),
            "list_releases": lambda p: self.list_releases(limit=p.get("limit", 5)),
            "get_file": lambda p: self.get_file_contents(p.get("path")),
        }

        if action in action_map:
            return await action_map[action](params)
        else:
            raise ValueError(f"Unknown action: {action}")


# ==================== SKILL INTERFACE ====================

class GitHubSkill:
    """
    Simplified skill interface for agent integration.

    Compatible with the tool router and agent system.
    """

    def __init__(self, token: str = None, repo: str = None):
        import os
        self.provider = GitHubProvider(
            token=token or os.environ.get("GITHUB_TOKEN", ""),
            repo=repo or os.environ.get("GITHUB_REPO", "")
        )
        self._connected = False

    async def connect(self) -> bool:
        """Connect to GitHub."""
        self._connected = await self.provider.connect()
        return self._connected

    async def list_issues(self, limit: int = 10) -> List[Dict]:
        """List open issues."""
        if not self._connected:
            await self.connect()
        issues = await self.provider.list_issues(limit=limit)
        return [i.to_dict() for i in issues]

    async def create_issue(self, title: str, body: str = "") -> Dict:
        """Create an issue."""
        if not self._connected:
            await self.connect()
        issue = await self.provider.create_issue(title, body)
        return issue.to_dict() if issue else {}

    async def list_prs(self, limit: int = 10) -> List[Dict]:
        """List open PRs."""
        if not self._connected:
            await self.connect()
        prs = await self.provider.list_prs(limit=limit)
        return [p.to_dict() for p in prs]

    async def create_pr(
        self,
        title: str,
        head: str,
        base: str = "main",
        body: str = ""
    ) -> Dict:
        """Create a PR."""
        if not self._connected:
            await self.connect()
        pr = await self.provider.create_pr(title, head, base, body)
        return pr.to_dict() if pr else {}

    async def merge_pr(self, number: int) -> bool:
        """Merge a PR."""
        if not self._connected:
            await self.connect()
        return await self.provider.merge_pr(number)

    async def get_file(self, path: str) -> Optional[str]:
        """Get file contents."""
        if not self._connected:
            await self.connect()
        return await self.provider.get_file_contents(path)

    async def list_workflows(self) -> List[Dict]:
        """List Actions workflows."""
        if not self._connected:
            await self.connect()
        return await self.provider.list_workflows()


# Global instance (lazy initialization)
github_skill: Optional[GitHubSkill] = None


def get_github_skill(token: str = None, repo: str = None) -> GitHubSkill:
    """Get or create the GitHub skill instance."""
    global github_skill
    if github_skill is None:
        github_skill = GitHubSkill(token, repo)
    return github_skill

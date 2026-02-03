"""
Farnsworth Google Calendar Integration - Full-Featured Implementation.

"I never miss an appointment. I'm an AI, for goodness sake!"

Complete Google Calendar API integration:
- Event CRUD operations
- Recurring events
- Free/busy lookup
- Calendar management
- Attendee management
- Reminders and notifications
- Event search and filtering
- Multi-calendar support
"""

import asyncio
import os.path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.warning("Google API packages not installed. Run: pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

from .base import ExternalProvider, IntegrationConfig, ConnectionStatus


# OAuth2 scopes for Calendar API
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/calendar.events'
]


class EventStatus(Enum):
    """Event status options."""
    CONFIRMED = "confirmed"
    TENTATIVE = "tentative"
    CANCELLED = "cancelled"


class EventVisibility(Enum):
    """Event visibility options."""
    DEFAULT = "default"
    PUBLIC = "public"
    PRIVATE = "private"
    CONFIDENTIAL = "confidential"


class ReminderMethod(Enum):
    """Reminder notification methods."""
    EMAIL = "email"
    POPUP = "popup"


class RecurrenceFrequency(Enum):
    """Recurrence frequency options."""
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"
    YEARLY = "YEARLY"


@dataclass
class CalendarEvent:
    """Structured calendar event data."""
    id: str
    summary: str
    start: datetime
    end: datetime
    description: str = ""
    location: str = ""
    status: EventStatus = EventStatus.CONFIRMED
    visibility: EventVisibility = EventVisibility.DEFAULT
    attendees: List[str] = field(default_factory=list)
    recurrence: List[str] = field(default_factory=list)
    reminders: List[Dict] = field(default_factory=list)
    html_link: str = ""
    calendar_id: str = "primary"
    organizer: str = ""
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    all_day: bool = False

    @property
    def duration_minutes(self) -> int:
        return int((self.end - self.start).total_seconds() / 60)

    @property
    def is_recurring(self) -> bool:
        return len(self.recurrence) > 0

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "summary": self.summary,
            "description": self.description,
            "location": self.location,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "status": self.status.value,
            "visibility": self.visibility.value,
            "attendees": self.attendees,
            "is_recurring": self.is_recurring,
            "duration_minutes": self.duration_minutes,
            "html_link": self.html_link,
            "all_day": self.all_day
        }


@dataclass
class Calendar:
    """Structured calendar data."""
    id: str
    summary: str
    description: str = ""
    timezone: str = "UTC"
    background_color: str = ""
    foreground_color: str = ""
    selected: bool = True
    primary: bool = False
    access_role: str = "reader"

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "summary": self.summary,
            "description": self.description,
            "timezone": self.timezone,
            "primary": self.primary,
            "access_role": self.access_role
        }


@dataclass
class FreeBusyPeriod:
    """Free/busy time period."""
    start: datetime
    end: datetime

    def to_dict(self) -> Dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat()
        }


class CalendarProvider(ExternalProvider):
    """
    Full-featured Google Calendar integration.

    Features:
    - Complete event CRUD
    - Recurring event support
    - Free/busy lookup
    - Multi-calendar support
    - Attendee management
    - Reminder configuration
    - Event search and filtering
    """

    def __init__(self, creds_path: str = None, token_path: str = None):
        super().__init__(IntegrationConfig(name="google_calendar"))
        self.creds_path = creds_path or "credentials.json"
        self.token_path = token_path or "token.json"
        self.service = None
        self._cache: Dict[str, tuple] = {}
        self._cache_ttl = 60  # seconds
        self._cached_events: List[CalendarEvent] = []
        self._last_sync: Optional[datetime] = None

    async def connect(self) -> bool:
        """Connect to Google Calendar API."""
        if not GOOGLE_AVAILABLE:
            logger.error("Google Calendar: Required packages not installed")
            return False

        loop = asyncio.get_event_loop()
        creds = None

        try:
            # Load existing token
            if os.path.exists(self.token_path):
                creds = await loop.run_in_executor(
                    None,
                    lambda: Credentials.from_authorized_user_file(self.token_path, SCOPES)
                )

            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    await loop.run_in_executor(None, lambda: creds.refresh(Request()))
                elif os.path.exists(self.creds_path):
                    # Interactive auth (won't work in headless mode)
                    logger.warning("Google Calendar: Need interactive auth. Run locally first.")
                    flow = InstalledAppFlow.from_client_secrets_file(self.creds_path, SCOPES)
                    creds = flow.run_local_server(port=0)

                    # Save the token for future runs
                    with open(self.token_path, 'w') as token:
                        token.write(creds.to_json())
                else:
                    logger.error("Google Calendar: No credentials available")
                    return False

            # Build the service
            self.service = await loop.run_in_executor(
                None,
                lambda: build('calendar', 'v3', credentials=creds)
            )

            # Test the connection
            await loop.run_in_executor(
                None,
                lambda: self.service.calendarList().list(maxResults=1).execute()
            )

            logger.info("Google Calendar: Connected successfully")
            self.status = ConnectionStatus.CONNECTED
            return True

        except Exception as e:
            logger.error(f"Google Calendar connection error: {e}")
            self.status = ConnectionStatus.ERROR
            return False

    async def disconnect(self):
        """Disconnect from Calendar API."""
        self.service = None
        self._cache.clear()
        self._cached_events.clear()
        self.status = ConnectionStatus.DISCONNECTED

    async def sync(self):
        """Sync calendar data."""
        if not self.service:
            logger.debug("Calendar sync skipped: not connected")
            return

        try:
            events = await self.get_upcoming_events(limit=50)
            self._cached_events = events
            self._last_sync = datetime.utcnow()
            logger.info(f"Calendar synced: {len(events)} events cached")
        except Exception as e:
            logger.warning(f"Calendar sync failed: {e}")

    # ==================== CALENDARS ====================

    async def list_calendars(self) -> List[Calendar]:
        """List all calendars the user has access to."""
        if not self.service:
            return []

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.service.calendarList().list().execute()
            )

            calendars = []
            for cal in result.get('items', []):
                calendars.append(Calendar(
                    id=cal.get('id', ''),
                    summary=cal.get('summary', ''),
                    description=cal.get('description', ''),
                    timezone=cal.get('timeZone', 'UTC'),
                    background_color=cal.get('backgroundColor', ''),
                    foreground_color=cal.get('foregroundColor', ''),
                    selected=cal.get('selected', True),
                    primary=cal.get('primary', False),
                    access_role=cal.get('accessRole', 'reader')
                ))

            return calendars
        except Exception as e:
            logger.error(f"List calendars error: {e}")
            return []

    async def get_calendar(self, calendar_id: str = "primary") -> Optional[Calendar]:
        """Get a specific calendar."""
        if not self.service:
            return None

        loop = asyncio.get_event_loop()
        try:
            cal = await loop.run_in_executor(
                None,
                lambda: self.service.calendars().get(calendarId=calendar_id).execute()
            )

            return Calendar(
                id=cal.get('id', ''),
                summary=cal.get('summary', ''),
                description=cal.get('description', ''),
                timezone=cal.get('timeZone', 'UTC'),
                primary=calendar_id == 'primary'
            )
        except Exception as e:
            logger.error(f"Get calendar error: {e}")
            return None

    # ==================== EVENTS ====================

    async def get_upcoming_events(
        self,
        limit: int = 10,
        calendar_id: str = "primary",
        time_min: datetime = None,
        time_max: datetime = None,
        query: str = None
    ) -> List[CalendarEvent]:
        """
        Get upcoming events.

        Args:
            limit: Maximum number of events
            calendar_id: Calendar to query
            time_min: Start of time range (default: now)
            time_max: End of time range
            query: Text search query
        """
        if not self.service:
            return []

        loop = asyncio.get_event_loop()

        now = time_min or datetime.utcnow()
        time_min_str = now.isoformat() + 'Z'

        try:
            params = {
                'calendarId': calendar_id,
                'timeMin': time_min_str,
                'maxResults': min(limit, 2500),
                'singleEvents': True,
                'orderBy': 'startTime'
            }

            if time_max:
                params['timeMax'] = time_max.isoformat() + 'Z'
            if query:
                params['q'] = query

            result = await loop.run_in_executor(
                None,
                lambda: self.service.events().list(**params).execute()
            )

            return [self._parse_event(e, calendar_id) for e in result.get('items', [])]
        except Exception as e:
            logger.error(f"Get events error: {e}")
            return []

    async def get_event(self, event_id: str, calendar_id: str = "primary") -> Optional[CalendarEvent]:
        """Get a specific event by ID."""
        if not self.service:
            return None

        loop = asyncio.get_event_loop()
        try:
            event = await loop.run_in_executor(
                None,
                lambda: self.service.events().get(calendarId=calendar_id, eventId=event_id).execute()
            )
            return self._parse_event(event, calendar_id)
        except Exception as e:
            logger.error(f"Get event error: {e}")
            return None

    async def create_event(
        self,
        summary: str,
        start_time: datetime,
        end_time: datetime,
        description: str = "",
        location: str = "",
        attendees: List[str] = None,
        calendar_id: str = "primary",
        timezone: str = "UTC",
        reminders: List[Dict] = None,
        recurrence: str = None,
        all_day: bool = False,
        visibility: EventVisibility = EventVisibility.DEFAULT
    ) -> Optional[CalendarEvent]:
        """
        Create a new calendar event.

        Args:
            summary: Event title
            start_time: Start datetime
            end_time: End datetime
            description: Event description
            location: Event location
            attendees: List of email addresses
            calendar_id: Calendar to create in
            timezone: Timezone for the event
            reminders: List of reminder configs
            recurrence: Recurrence rule (e.g., "RRULE:FREQ=WEEKLY;BYDAY=MO")
            all_day: Whether this is an all-day event
            visibility: Event visibility
        """
        if not self.service:
            return None

        loop = asyncio.get_event_loop()

        # Build event body
        event = {
            'summary': summary,
            'description': description,
            'location': location,
            'visibility': visibility.value
        }

        # Handle all-day vs timed events
        if all_day:
            event['start'] = {'date': start_time.strftime('%Y-%m-%d'), 'timeZone': timezone}
            event['end'] = {'date': end_time.strftime('%Y-%m-%d'), 'timeZone': timezone}
        else:
            event['start'] = {'dateTime': start_time.isoformat(), 'timeZone': timezone}
            event['end'] = {'dateTime': end_time.isoformat(), 'timeZone': timezone}

        # Add attendees
        if attendees:
            event['attendees'] = [{'email': email} for email in attendees]

        # Add reminders
        if reminders:
            event['reminders'] = {
                'useDefault': False,
                'overrides': reminders
            }
        else:
            event['reminders'] = {'useDefault': True}

        # Add recurrence
        if recurrence:
            event['recurrence'] = [recurrence]

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.service.events().insert(
                    calendarId=calendar_id,
                    body=event,
                    sendUpdates='all' if attendees else 'none'
                ).execute()
            )
            logger.info(f"Created event: {summary}")
            return self._parse_event(result, calendar_id)
        except Exception as e:
            logger.error(f"Create event error: {e}")
            return None

    async def update_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
        summary: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        description: str = None,
        location: str = None,
        attendees: List[str] = None,
        timezone: str = None
    ) -> Optional[CalendarEvent]:
        """Update an existing event."""
        if not self.service:
            return None

        loop = asyncio.get_event_loop()

        try:
            # Get the current event
            current = await loop.run_in_executor(
                None,
                lambda: self.service.events().get(calendarId=calendar_id, eventId=event_id).execute()
            )

            # Update fields
            if summary is not None:
                current['summary'] = summary
            if description is not None:
                current['description'] = description
            if location is not None:
                current['location'] = location
            if attendees is not None:
                current['attendees'] = [{'email': email} for email in attendees]
            if start_time is not None:
                tz = timezone or current.get('start', {}).get('timeZone', 'UTC')
                current['start'] = {'dateTime': start_time.isoformat(), 'timeZone': tz}
            if end_time is not None:
                tz = timezone or current.get('end', {}).get('timeZone', 'UTC')
                current['end'] = {'dateTime': end_time.isoformat(), 'timeZone': tz}

            result = await loop.run_in_executor(
                None,
                lambda: self.service.events().update(
                    calendarId=calendar_id,
                    eventId=event_id,
                    body=current,
                    sendUpdates='all' if attendees else 'none'
                ).execute()
            )

            return self._parse_event(result, calendar_id)
        except Exception as e:
            logger.error(f"Update event error: {e}")
            return None

    async def delete_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
        send_updates: bool = True
    ) -> bool:
        """Delete an event."""
        if not self.service:
            return False

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self.service.events().delete(
                    calendarId=calendar_id,
                    eventId=event_id,
                    sendUpdates='all' if send_updates else 'none'
                ).execute()
            )
            logger.info(f"Deleted event: {event_id}")
            return True
        except Exception as e:
            logger.error(f"Delete event error: {e}")
            return False

    async def quick_add(self, text: str, calendar_id: str = "primary") -> Optional[CalendarEvent]:
        """
        Create an event using natural language.

        Example: "Lunch with John tomorrow at noon"
        """
        if not self.service:
            return None

        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.service.events().quickAdd(
                    calendarId=calendar_id,
                    text=text
                ).execute()
            )
            return self._parse_event(result, calendar_id)
        except Exception as e:
            logger.error(f"Quick add error: {e}")
            return None

    # ==================== RECURRING EVENTS ====================

    @staticmethod
    def build_recurrence_rule(
        frequency: RecurrenceFrequency,
        interval: int = 1,
        count: int = None,
        until: datetime = None,
        by_day: List[str] = None,  # ["MO", "TU", "WE", ...]
        by_month_day: List[int] = None,  # [1, 15, ...]
        by_month: List[int] = None  # [1, 6, ...]
    ) -> str:
        """
        Build an RRULE string for recurring events.

        Args:
            frequency: DAILY, WEEKLY, MONTHLY, YEARLY
            interval: How many units between recurrences
            count: Number of occurrences
            until: End date for recurrence
            by_day: Days of week (for WEEKLY)
            by_month_day: Days of month (for MONTHLY)
            by_month: Months (for YEARLY)
        """
        parts = [f"RRULE:FREQ={frequency.value}"]

        if interval > 1:
            parts.append(f"INTERVAL={interval}")
        if count:
            parts.append(f"COUNT={count}")
        if until:
            parts.append(f"UNTIL={until.strftime('%Y%m%dT%H%M%SZ')}")
        if by_day:
            parts.append(f"BYDAY={','.join(by_day)}")
        if by_month_day:
            parts.append(f"BYMONTHDAY={','.join(map(str, by_month_day))}")
        if by_month:
            parts.append(f"BYMONTH={','.join(map(str, by_month))}")

        return ";".join(parts)

    async def get_recurring_instances(
        self,
        event_id: str,
        calendar_id: str = "primary",
        time_min: datetime = None,
        time_max: datetime = None,
        limit: int = 100
    ) -> List[CalendarEvent]:
        """Get instances of a recurring event."""
        if not self.service:
            return []

        loop = asyncio.get_event_loop()

        params = {
            'calendarId': calendar_id,
            'eventId': event_id,
            'maxResults': min(limit, 2500)
        }

        if time_min:
            params['timeMin'] = time_min.isoformat() + 'Z'
        if time_max:
            params['timeMax'] = time_max.isoformat() + 'Z'

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.service.events().instances(**params).execute()
            )
            return [self._parse_event(e, calendar_id) for e in result.get('items', [])]
        except Exception as e:
            logger.error(f"Get recurring instances error: {e}")
            return []

    # ==================== FREE/BUSY ====================

    async def get_free_busy(
        self,
        time_min: datetime,
        time_max: datetime,
        calendar_ids: List[str] = None,
        timezone: str = "UTC"
    ) -> Dict[str, List[FreeBusyPeriod]]:
        """
        Get free/busy information for calendars.

        Args:
            time_min: Start of time range
            time_max: End of time range
            calendar_ids: Calendars to check (default: primary)
            timezone: Timezone for the query
        """
        if not self.service:
            return {}

        if calendar_ids is None:
            calendar_ids = ["primary"]

        loop = asyncio.get_event_loop()

        body = {
            "timeMin": time_min.isoformat() + 'Z',
            "timeMax": time_max.isoformat() + 'Z',
            "timeZone": timezone,
            "items": [{"id": cal_id} for cal_id in calendar_ids]
        }

        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.service.freebusy().query(body=body).execute()
            )

            free_busy = {}
            for cal_id, data in result.get('calendars', {}).items():
                periods = []
                for busy in data.get('busy', []):
                    periods.append(FreeBusyPeriod(
                        start=datetime.fromisoformat(busy['start'].replace('Z', '+00:00')),
                        end=datetime.fromisoformat(busy['end'].replace('Z', '+00:00'))
                    ))
                free_busy[cal_id] = periods

            return free_busy
        except Exception as e:
            logger.error(f"Get free/busy error: {e}")
            return {}

    async def find_free_slots(
        self,
        duration_minutes: int,
        time_min: datetime = None,
        time_max: datetime = None,
        calendar_id: str = "primary",
        working_hours: Tuple[int, int] = (9, 17)
    ) -> List[Tuple[datetime, datetime]]:
        """
        Find free time slots of a given duration.

        Args:
            duration_minutes: Required slot duration
            time_min: Start of search range (default: now)
            time_max: End of search range (default: 7 days from now)
            calendar_id: Calendar to check
            working_hours: Tuple of (start_hour, end_hour) for working day
        """
        if time_min is None:
            time_min = datetime.utcnow()
        if time_max is None:
            time_max = time_min + timedelta(days=7)

        busy_periods = await self.get_free_busy(time_min, time_max, [calendar_id])
        busy = busy_periods.get(calendar_id, [])

        free_slots = []
        current = time_min

        for period in busy:
            # Check if there's a gap before this busy period
            if current < period.start:
                # Respect working hours
                slot_start = current
                slot_end = period.start

                # Check if slot is within working hours
                if (slot_start.hour >= working_hours[0] and
                    slot_end.hour <= working_hours[1]):
                    gap_minutes = (slot_end - slot_start).total_seconds() / 60
                    if gap_minutes >= duration_minutes:
                        free_slots.append((slot_start, slot_end))

            current = max(current, period.end)

        # Check for time after last busy period
        if current < time_max:
            gap_minutes = (time_max - current).total_seconds() / 60
            if gap_minutes >= duration_minutes:
                free_slots.append((current, time_max))

        return free_slots

    # ==================== REMINDERS ====================

    @staticmethod
    def build_reminder(method: ReminderMethod, minutes: int) -> Dict:
        """Build a reminder configuration."""
        return {"method": method.value, "minutes": minutes}

    # ==================== UTILITIES ====================

    def _parse_event(self, event: Dict, calendar_id: str = "primary") -> CalendarEvent:
        """Parse raw event data into CalendarEvent."""
        # Parse start/end times
        start_data = event.get('start', {})
        end_data = event.get('end', {})

        all_day = 'date' in start_data

        if all_day:
            start = datetime.strptime(start_data.get('date', ''), '%Y-%m-%d')
            end = datetime.strptime(end_data.get('date', ''), '%Y-%m-%d')
        else:
            start_str = start_data.get('dateTime', '')
            end_str = end_data.get('dateTime', '')
            start = datetime.fromisoformat(start_str.replace('Z', '+00:00')) if start_str else datetime.now()
            end = datetime.fromisoformat(end_str.replace('Z', '+00:00')) if end_str else datetime.now()

        # Parse attendees
        attendees = [a.get('email', '') for a in event.get('attendees', [])]

        # Parse status
        status_map = {
            'confirmed': EventStatus.CONFIRMED,
            'tentative': EventStatus.TENTATIVE,
            'cancelled': EventStatus.CANCELLED
        }
        status = status_map.get(event.get('status', 'confirmed'), EventStatus.CONFIRMED)

        # Parse visibility
        visibility_map = {
            'default': EventVisibility.DEFAULT,
            'public': EventVisibility.PUBLIC,
            'private': EventVisibility.PRIVATE,
            'confidential': EventVisibility.CONFIDENTIAL
        }
        visibility = visibility_map.get(event.get('visibility', 'default'), EventVisibility.DEFAULT)

        return CalendarEvent(
            id=event.get('id', ''),
            summary=event.get('summary', ''),
            description=event.get('description', ''),
            location=event.get('location', ''),
            start=start,
            end=end,
            status=status,
            visibility=visibility,
            attendees=attendees,
            recurrence=event.get('recurrence', []),
            reminders=event.get('reminders', {}).get('overrides', []),
            html_link=event.get('htmlLink', ''),
            calendar_id=calendar_id,
            organizer=event.get('organizer', {}).get('email', ''),
            all_day=all_day
        )

    async def execute_action(self, action: str, params: Dict[str, Any]) -> Any:
        """Execute an action (legacy interface)."""
        if not self.service:
            raise ConnectionError("Google Calendar not connected")

        action_map = {
            "get_upcoming": lambda p: self.get_upcoming_events(p.get("limit", 5)),
            "create_event": lambda p: self.create_event(
                summary=p.get('summary'),
                start_time=datetime.fromisoformat(p.get('start_time')),
                end_time=datetime.fromisoformat(p.get('end_time')),
                description=p.get('description', ''),
                location=p.get('location', ''),
                attendees=p.get('attendees', []),
                timezone=p.get('timezone', 'UTC')
            ),
            "schedule_event": lambda p: self.create_event(
                summary=p.get('summary'),
                start_time=datetime.fromisoformat(p.get('start_time')),
                end_time=datetime.fromisoformat(p.get('end_time')),
                description=p.get('description', ''),
                location=p.get('location', ''),
                attendees=p.get('attendees', []),
                timezone=p.get('timezone', 'UTC')
            ),
            "delete_event": lambda p: self.delete_event(p.get('event_id')),
            "update_event": lambda p: self.update_event(
                event_id=p.get('event_id'),
                summary=p.get('summary'),
                description=p.get('description'),
                location=p.get('location')
            ),
            "quick_add": lambda p: self.quick_add(p.get('text')),
            "list_calendars": lambda p: self.list_calendars(),
            "get_free_busy": lambda p: self.get_free_busy(
                datetime.fromisoformat(p.get('time_min')),
                datetime.fromisoformat(p.get('time_max'))
            )
        }

        if action in action_map:
            return await action_map[action](params)
        else:
            raise ValueError(f"Unknown action: {action}")


# ==================== SKILL INTERFACE ====================

class CalendarSkill:
    """
    Simplified skill interface for agent integration.

    Compatible with the tool router and agent system.
    """

    def __init__(self, creds_path: str = None, token_path: str = None):
        self.provider = CalendarProvider(creds_path, token_path)
        self._connected = False

    async def connect(self) -> bool:
        """Connect to Google Calendar."""
        self._connected = await self.provider.connect()
        return self._connected

    async def get_upcoming(self, limit: int = 5) -> List[Dict]:
        """Get upcoming events."""
        if not self._connected:
            await self.connect()
        events = await self.provider.get_upcoming_events(limit=limit)
        return [e.to_dict() for e in events]

    async def create_event(
        self,
        summary: str,
        start: str,  # ISO format
        end: str,  # ISO format
        description: str = "",
        location: str = "",
        attendees: List[str] = None
    ) -> Dict:
        """Create a new event."""
        if not self._connected:
            await self.connect()
        event = await self.provider.create_event(
            summary=summary,
            start_time=datetime.fromisoformat(start),
            end_time=datetime.fromisoformat(end),
            description=description,
            location=location,
            attendees=attendees or []
        )
        return event.to_dict() if event else {}

    async def quick_add(self, text: str) -> Dict:
        """Create event from natural language."""
        if not self._connected:
            await self.connect()
        event = await self.provider.quick_add(text)
        return event.to_dict() if event else {}

    async def delete_event(self, event_id: str) -> bool:
        """Delete an event."""
        if not self._connected:
            await self.connect()
        return await self.provider.delete_event(event_id)

    async def find_free_time(self, duration_minutes: int, days: int = 7) -> List[Dict]:
        """Find free time slots."""
        if not self._connected:
            await self.connect()
        slots = await self.provider.find_free_slots(
            duration_minutes=duration_minutes,
            time_max=datetime.utcnow() + timedelta(days=days)
        )
        return [{"start": s.isoformat(), "end": e.isoformat()} for s, e in slots]


# Global instance (lazy initialization)
calendar_skill: Optional[CalendarSkill] = None


def get_calendar_skill(creds_path: str = None, token_path: str = None) -> CalendarSkill:
    """Get or create the Calendar skill instance."""
    global calendar_skill
    if calendar_skill is None:
        calendar_skill = CalendarSkill(creds_path, token_path)
    return calendar_skill

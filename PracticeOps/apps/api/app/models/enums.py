"""Domain enums for PracticeOps.

These Python enums map directly to PostgreSQL ENUM types.
The enum values must match exactly between Python and the database.
"""

import enum


class Role(str, enum.Enum):
    """User roles within a team."""

    MEMBER = "MEMBER"
    SECTION_LEADER = "SECTION_LEADER"
    ADMIN = "ADMIN"


class AssignmentType(str, enum.Enum):
    """Types of practice assignments."""

    SONG_WORK = "SONG_WORK"
    TECHNIQUE = "TECHNIQUE"
    MEMORIZATION = "MEMORIZATION"
    LISTENING = "LISTENING"


class Priority(str, enum.Enum):
    """Priority levels for tickets and assignments."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    BLOCKING = "BLOCKING"


class AssignmentScope(str, enum.Enum):
    """Scope of an assignment - who it applies to."""

    TEAM = "TEAM"
    SECTION = "SECTION"


class TicketCategory(str, enum.Enum):
    """Categories for practice tickets."""

    PITCH = "PITCH"
    RHYTHM = "RHYTHM"
    MEMORY = "MEMORY"
    BLEND = "BLEND"
    TECHNIQUE = "TECHNIQUE"
    OTHER = "OTHER"


class TicketVisibility(str, enum.Enum):
    """Visibility levels for tickets."""

    PRIVATE = "PRIVATE"
    SECTION = "SECTION"
    TEAM = "TEAM"


class TicketStatus(str, enum.Enum):
    """Status progression for tickets."""

    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    RESOLVED = "RESOLVED"
    VERIFIED = "VERIFIED"


class TicketActivityType(str, enum.Enum):
    """Types of activity that can occur on a ticket."""

    CREATED = "CREATED"
    COMMENT = "COMMENT"
    STATUS_CHANGE = "STATUS_CHANGE"
    VERIFIED = "VERIFIED"
    CLAIMED = "CLAIMED"
    REASSIGNED = "REASSIGNED"


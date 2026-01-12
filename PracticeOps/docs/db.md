# PracticeOps Database Schema

## Entity Relationship Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CORE ENTITIES                                   │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐         ┌──────────────┐         ┌───────────────────┐
│    users     │         │    teams     │         │ team_memberships  │
├──────────────┤         ├──────────────┤         ├───────────────────┤
│ id (PK)      │◄───┐    │ id (PK)      │◄───┐    │ id (PK)           │
│ email        │    │    │ name         │    │    │ team_id (FK)──────┼───► teams.id
│ password_hash│    │    │ created_at   │    │    │ user_id (FK)──────┼───► users.id
│ display_name │    │    │ updated_at   │    │    │ role (enum)       │
│ created_at   │    │    └──────────────┘    │    │ section           │
│ updated_at   │    │                        │    │ created_at        │
└──────────────┘    │                        │    │ updated_at        │
                    │                        │    └───────────────────┘
                    │                        │    UQ(team_id, user_id)
                    │                        │
┌───────────────────────────────────────────────────────────────────────────────┐
│                           REHEARSAL CYCLE DOMAIN                               │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────┐         ┌──────────────────┐
│ rehearsal_cycles  │         │   assignments    │
├───────────────────┤         ├──────────────────┤
│ id (PK)           │◄────┐   │ id (PK)          │
│ team_id (FK)──────┼───► │   │ cycle_id (FK)────┼───► rehearsal_cycles.id
│ name              │teams│   │ type (enum)      │
│ date              │.id  │   │ scope (enum)     │
│ created_at        │     │   │ section          │
└───────────────────┘     │   │ title            │
UQ(team_id, date)         │   │ description      │
                          │   │ due_at           │
                          │   │ created_at       │
                          │   │ updated_at       │
                          │   └──────────────────┘
                          │
┌───────────────────────────────────────────────────────────────────────────────┐
│                           PRACTICE LOGGING DOMAIN                              │
└───────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌─────────────────────────┐
│  practice_logs   │         │ practice_log_assignments│
├──────────────────┤         ├─────────────────────────┤
│ id (PK)          │◄────────┤ id (PK)                 │
│ user_id (FK)─────┼───► users.id                      │
│ team_id (FK)─────┼───► teams.id                      │
│ cycle_id (FK)────┼───► rehearsal_cycles.id (nullable)│
│ duration_minutes │         │ practice_log_id (FK)────┼───► practice_logs.id
│ notes            │         │ assignment_id (FK)──────┼───► assignments.id
│ occurred_at      │         └─────────────────────────┘
│ created_at       │
└──────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                              TICKET DOMAIN                                     │
└───────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────┐         ┌────────────────────┐
│       tickets        │         │  ticket_activity   │
├──────────────────────┤         ├────────────────────┤
│ id (PK)              │◄────────┤ id (PK)            │
│ team_id (FK)─────────┼───► teams.id                 │
│ cycle_id (FK)────────┼───► rehearsal_cycles.id      │ ticket_id (FK)──────┼───► tickets.id
│ owner_id (FK)────────┼───► users.id                 │ user_id (FK)────────┼───► users.id
│ created_by (FK)──────┼───► users.id                 │ type (enum)         │
│ claimed_by (FK)──────┼───► users.id (nullable)      │ content             │
│ category (enum)      │         │ old_status (enum)  │
│ priority (enum)      │         │ new_status (enum)  │
│ status (enum)        │         │ created_at         │
│ visibility (enum)    │         └────────────────────┘
│ section              │
│ title                │
│ description          │
│ song_ref             │
│ due_at               │
│ created_at           │
│ updated_at           │
│ resolved_at          │
│ verified_at          │
└──────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                           INVITATION DOMAIN                                    │
└───────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│     invites      │
├──────────────────┤
│ id (PK)          │
│ team_id (FK)─────┼───► teams.id
│ token (UQ)       │
│ role (enum)      │
│ section          │
│ expires_at       │
│ used_at          │
│ created_by (FK)──┼───► users.id
│ created_at       │
└──────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│                         NOTIFICATION DOMAIN                                    │
└───────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────┐
│  notification_preferences   │
├─────────────────────────────┤
│ id (PK)                     │
│ user_id (FK)────────────────┼───► users.id
│ team_id (FK)────────────────┼───► teams.id
│ email_enabled               │
│ deadline_reminder_hours     │
│ created_at                  │
│ updated_at                  │
└─────────────────────────────┘
UQ(team_id, user_id)
```

## Enums

| Enum Name | Values |
|-----------|--------|
| `role_enum` | MEMBER, SECTION_LEADER, ADMIN |
| `assignment_type_enum` | SONG_WORK, TECHNIQUE, MEMORIZATION, LISTENING |
| `assignment_scope_enum` | TEAM, SECTION |
| `priority_enum` | LOW, MEDIUM, BLOCKING |
| `ticket_category_enum` | PITCH, RHYTHM, MEMORY, BLEND, TECHNIQUE, OTHER |
| `ticket_visibility_enum` | PRIVATE, SECTION, TEAM |
| `ticket_status_enum` | OPEN, IN_PROGRESS, BLOCKED, RESOLVED, VERIFIED |
| `ticket_activity_type_enum` | CREATED, COMMENT, STATUS_CHANGE, VERIFIED, CLAIMED, REASSIGNED |

## Composite Unique Constraints

| Table | Columns | Constraint Name | Purpose |
|-------|---------|-----------------|---------|
| `team_memberships` | (team_id, user_id) | `uq_team_memberships_team_user` | Prevent duplicate memberships |
| `rehearsal_cycles` | (team_id, date) | `uq_rehearsal_cycles_team_date` | One cycle per team per date |
| `notification_preferences` | (team_id, user_id) | `uq_notification_preferences_team_user` | One preference set per user per team |

## Index Strategy

### tickets

| Index Name | Columns | Rationale |
|------------|---------|-----------|
| `ix_tickets_cycle_status` | (cycle_id, status) | Filter tickets by cycle and status for dashboard views |
| `ix_tickets_cycle_priority` | (cycle_id, priority) | Sort/filter tickets by priority within a cycle |
| `ix_tickets_team_cycle` | (team_id, cycle_id) | List all tickets for a team's cycle |
| `ix_tickets_visibility_section` | (visibility, section) | Access control queries for section-scoped visibility |

**Query patterns supported:**
- "Show all BLOCKING tickets in current cycle"
- "Show all OPEN tickets for Soprano section"
- "Count tickets by status for team dashboard"

### practice_logs

| Index Name | Columns | Rationale |
|------------|---------|-----------|
| `ix_practice_logs_user_cycle` | (user_id, cycle_id) | User's practice history within a cycle |
| `ix_practice_logs_team_cycle` | (team_id, cycle_id) | Team-wide practice summary for a cycle |
| `ix_practice_logs_cycle_occurred` | (cycle_id, occurred_at DESC) | Chronological practice feed within a cycle |

**Query patterns supported:**
- "Show my practice logs for this week"
- "Calculate team practice minutes for current cycle"
- "Recent practice activity feed"

### assignments

| Index Name | Columns | Rationale |
|------------|---------|-----------|
| `ix_assignments_cycle_scope` | (cycle_id, scope) | Filter team vs section assignments |
| `ix_assignments_cycle_section` | (cycle_id, section) | Section-specific assignment queries |

**Query patterns supported:**
- "Show all TEAM-wide assignments for this cycle"
- "Show assignments for Tenor section this week"

### team_memberships

| Index Name | Columns | Rationale |
|------------|---------|-----------|
| `ix_team_memberships_team_role` | (team_id, role) | List all admins/leaders for a team |
| `ix_team_memberships_team_section` | (team_id, section) | List members by section |

**Query patterns supported:**
- "Who are the section leaders?"
- "List all members in the Alto section"

## Foreign Key Cascade Rules

| Table | FK Column | On Delete |
|-------|-----------|-----------|
| `team_memberships.team_id` | teams.id | CASCADE |
| `team_memberships.user_id` | users.id | CASCADE |
| `rehearsal_cycles.team_id` | teams.id | CASCADE |
| `assignments.cycle_id` | rehearsal_cycles.id | CASCADE |
| `practice_logs.user_id` | users.id | CASCADE |
| `practice_logs.team_id` | teams.id | CASCADE |
| `practice_logs.cycle_id` | rehearsal_cycles.id | SET NULL |
| `invites.team_id` | teams.id | CASCADE |
| `invites.created_by` | users.id | CASCADE |
| `tickets.team_id` | teams.id | CASCADE |
| `tickets.cycle_id` | rehearsal_cycles.id | SET NULL |
| `tickets.owner_id` | users.id | CASCADE |
| `tickets.created_by` | users.id | CASCADE |
| `tickets.claimed_by` | users.id | SET NULL |
| `ticket_activity.ticket_id` | tickets.id | CASCADE |
| `ticket_activity.user_id` | users.id | CASCADE |
| `notification_preferences.user_id` | users.id | CASCADE |
| `notification_preferences.team_id` | teams.id | CASCADE |
| `practice_log_assignments.practice_log_id` | practice_logs.id | CASCADE |
| `practice_log_assignments.assignment_id` | assignments.id | CASCADE |

## Default Values

| Table | Column | Default |
|-------|--------|---------|
| `tickets.priority` | priority_enum | 'LOW' |
| `tickets.status` | ticket_status_enum | 'OPEN' |
| `tickets.visibility` | ticket_visibility_enum | 'PRIVATE' |
| `notification_preferences.email_enabled` | boolean | true |
| `notification_preferences.deadline_reminder_hours` | integer | 24 |

## Migration Commands

```bash
# Run migrations
cd apps/api
alembic upgrade head

# Rollback one migration
alembic downgrade -1

# View migration history
alembic history

# Generate new migration (after model changes)
alembic revision --autogenerate -m "description"
```

## Seed Data

```bash
# Seed initial data (admin user, team, cycle, assignments)
cd apps/api
python -m scripts.seed
```

Creates:
- Admin user: `admin@practiceops.local` (password: `admin123`)
- Team: "Demo A Cappella Group"
- Membership: Admin with Tenor section
- Rehearsal cycle: Next Monday at 7 PM
- Two assignments: Song work (team) and memorization (section)


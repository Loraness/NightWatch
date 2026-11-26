import logging
import sys
from datetime import timedelta

from mcp.server.mcpserver import MCPServer
from sqlalchemy import func, select

from app.database import async_session_maker
from app.models import ServiceLog

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("nightwatch.mcp")

mcp = MCPServer("nightwatch-diagnostics")

_DEPLOYMENT_HISTORY: dict[str, list[str]] = {
    "billing-api": [
        "2026-11-25 02:14 UTC | v2.18.4 | feat(db): tune StatementTimeout and asyncpg pool settings",
        "2026-11-24 18:40 UTC | v2.18.3 | fix(retry): adjust billing webhook backoff",
    ],
    "payment-worker": [
        "2026-11-25 01:50 UTC | v1.9.2 | perf(redis): switch cache batching to pipeline",
        "2026-11-24 14:10 UTC | v1.9.1 | chore(deps): bump httpx to 0.27.2",
    ],
    "auth-service": [
        "2026-11-25 02:05 UTC | v3.4.0 | feat(rate-limit): tighten Gemini fallback quotas",
    ],
}

@mcp.tool()
async def get_logs_by_request_id(request_id: str) -> str:
    """Fetch ordered service logs from the database for a given request_id."""
    clean_id = request_id.strip()
    if not clean_id:
        return "Invalid request_id: must not be empty."

    async with async_session_maker() as session:
        stmt = (
            select(ServiceLog)
            .where(ServiceLog.request_id == clean_id)
            .order_by(ServiceLog.created_at)
            .limit(50)
        )
        logs = (await session.execute(stmt)).scalars().all()

    if not logs:
        return f"No logs found for request_id='{clean_id}'."

    lines = [
        f"[{log.created_at:%Y-%m-%d %H:%M:%S}] {log.level} {log.service_name}: {log.message}"
        for log in logs
    ]
    return "\n".join(lines)


@mcp.tool()
async def get_service_error_stats(service_name: str, minutes: int = 30) -> str:
    """Count ERROR and CRITICAL logs for a service over the last N minutes."""
    clean_service = service_name.strip()
    if not clean_service:
        return "Invalid service_name: must not be empty."
    bounded_minutes = max(1, min(minutes, 1440))
    cutoff = func.now() - timedelta(minutes=bounded_minutes)

    async with async_session_maker() as session:
        stmt = (
            select(ServiceLog.level, func.count(ServiceLog.id))
            .where(
                ServiceLog.service_name == clean_service,
                ServiceLog.level.in_(("ERROR", "CRITICAL")),
                ServiceLog.created_at >= cutoff,
            )
            .group_by(ServiceLog.level)
        )
        rows = (await session.execute(stmt)).all()

    if not rows:
        return (
            f"No ERROR or CRITICAL logs for service='{clean_service}' "
            f"in the last {bounded_minutes} minutes."
        )
    summary = ", ".join(f"{level}={count}" for level, count in rows)
    total = sum(count for _, count in rows)
    return (
        f"Service '{clean_service}' had {total} error log(s) in the last "
        f"{bounded_minutes}m ({summary})."
    )


@mcp.tool()
async def get_recent_deployments(service_name: str) -> str:
    """Return recent deployments and release notes for a given microservice."""
    clean_service = service_name.strip()
    if not clean_service:
        return "Invalid service_name: must not be empty."
    entries = _DEPLOYMENT_HISTORY.get(clean_service)
    if not entries:
        return f"No deployment records found for service='{clean_service}'."
    return "\n".join(entries)


@mcp.resource("nightwatch://services")
def list_monitored_services() -> str:
    """List microservices monitored by NightWatch diagnostics."""
    return "\n".join(sorted(_DEPLOYMENT_HISTORY.keys()))


if __name__ == "__main__":
    mcp.run()

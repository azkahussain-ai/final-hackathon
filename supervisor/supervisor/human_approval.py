"""
supervisor/human_approval.py
------------------------------
STAGE VIII: Human-in-the-Loop core module.

This is the piece both `demo.py` (CLI) and `streamlit_app.py` (web UI)
import. It is deliberately UI-agnostic: it just knows how to take an
AI recommendation (the Stage VII digital-twin report) plus a human's
decision, turn it into a structured, auditable record, and append it
to a JSON log on disk.

Design rules (per the hackathon brief, Stage VIII):
  - The AI recommendation is NEVER the final decision on its own.
  - The human can APPROVE, REJECT, or OVERRIDE (modify) the action.
  - Every decision is logged with who made it, when, and why.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


VALID_DECISIONS = {"APPROVED", "REJECTED", "OVERRIDDEN"}


@dataclass
class ApprovalRecord:
    """One immutable audit-trail entry for a human decision."""

    machine_id: str
    ai_recommendation: str
    rationale: str
    human_decision: str          # APPROVED | REJECTED | OVERRIDDEN
    final_action: str            # what will actually be executed
    reviewer_name: str
    reviewer_comment: str
    timestamp_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    scenario_summary: Optional[Dict[str, Any]] = None
    source_report: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HumanSupervisor:
    """
    Wraps the read/append/validate logic around the JSON audit log.

    Usage:
        supervisor = HumanSupervisor(log_path="outputs/approval_log.json")
        record = supervisor.record_decision(
            report=digital_twin_report,
            decision="APPROVED",
            final_action="PREVENTIVE_MAINTENANCE",
            reviewer_name="A. Khan",
            reviewer_comment="Confidence is high, schedule downtime tonight.",
        )
    """

    def __init__(self, log_path: str = "outputs/approval_log.json"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
        if not os.path.exists(self.log_path):
            with open(self.log_path, "w") as f:
                json.dump([], f)

    # ------------------------------------------------------------------
    def _load_log(self) -> List[Dict[str, Any]]:
        with open(self.log_path, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def _save_log(self, entries: List[Dict[str, Any]]) -> None:
        with open(self.log_path, "w") as f:
            json.dump(entries, f, indent=2, default=str)

    # ------------------------------------------------------------------
    def record_decision(
        self,
        report: Dict[str, Any],
        decision: str,
        final_action: str,
        reviewer_name: str,
        reviewer_comment: str = "",
        source_report: Optional[str] = None,
    ) -> ApprovalRecord:
        """
        Validate + persist one human decision against an AI recommendation.
        Returns the ApprovalRecord that was written.
        """
        decision = decision.upper().strip()
        if decision not in VALID_DECISIONS:
            raise ValueError(
                f"decision must be one of {VALID_DECISIONS}, got {decision!r}"
            )

        record = ApprovalRecord(
            machine_id=report.get("machine_id", "UNKNOWN"),
            ai_recommendation=report.get("recommendation", "UNKNOWN"),
            rationale=report.get("rationale", ""),
            human_decision=decision,
            final_action=final_action,
            reviewer_name=reviewer_name or "unspecified",
            reviewer_comment=reviewer_comment,
            scenario_summary={
                s["scenario"]: {
                    "expected_cost_usd": s.get("expected_cost_usd"),
                    "unplanned_failure_probability": s.get(
                        "unplanned_failure_probability"
                    ),
                }
                for s in report.get("scenarios", [])
            },
            source_report=source_report,
        )

        log = self._load_log()
        log.append(record.to_dict())
        self._save_log(log)
        return record

    # ------------------------------------------------------------------
    def history(self) -> List[Dict[str, Any]]:
        """Return the full audit trail, most recent last."""
        return self._load_log()

    def history_for_machine(self, machine_id: str) -> List[Dict[str, Any]]:
        return [r for r in self._load_log() if r.get("machine_id") == machine_id]

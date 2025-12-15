"""
Alarm and Safety Systems for Implantable Devices.

Provides:
- Hierarchical alarm management (advisory, warning, critical)
- Safety interlocks and fail-safe mechanisms
- Hazard analysis support
- Emergency response protocols
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Any, Optional
from abc import ABC, abstractmethod
import threading


class AlarmPriority(Enum):
    """Alarm priority levels per IEC 60601-1-8."""
    LOW = "low"           # Advisory - informational
    MEDIUM = "medium"     # Warning - attention needed
    HIGH = "high"         # Critical - immediate action required


class AlarmCategory(Enum):
    """Categories of device alarms."""
    PHYSIOLOGICAL = "physiological"     # Patient-related
    TECHNICAL = "technical"             # Device-related
    ENVIRONMENTAL = "environmental"     # External factors
    OPERATOR = "operator"               # User action needed


class AlarmState(Enum):
    """Alarm states."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"
    RESOLVED = "resolved"
    LATCHED = "latched"  # Remains until explicitly reset


@dataclass
class Alarm:
    """Represents a device alarm."""
    alarm_id: str
    code: str
    message: str
    priority: AlarmPriority
    category: AlarmCategory
    device_serial: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    state: AlarmState = AlarmState.ACTIVE
    parameter: Optional[str] = None
    value: Optional[float] = None
    threshold: Optional[float] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None
    resolved_at: Optional[str] = None
    escalation_count: int = 0

    def acknowledge(self, user: str) -> None:
        """Acknowledge the alarm."""
        if self.state == AlarmState.ACTIVE:
            self.state = AlarmState.ACKNOWLEDGED
            self.acknowledged_by = user
            self.acknowledged_at = datetime.now().isoformat()

    def silence(self, duration_minutes: int = 2) -> None:
        """Temporarily silence the alarm."""
        if self.state in (AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED):
            self.state = AlarmState.SILENCED
            # Note: Actual silencing timeout handled by AlarmManager

    def resolve(self) -> None:
        """Mark alarm as resolved."""
        self.state = AlarmState.RESOLVED
        self.resolved_at = datetime.now().isoformat()

    def escalate(self) -> None:
        """Escalate alarm priority."""
        self.escalation_count += 1
        if self.priority == AlarmPriority.LOW:
            self.priority = AlarmPriority.MEDIUM
        elif self.priority == AlarmPriority.MEDIUM:
            self.priority = AlarmPriority.HIGH

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "alarm_id": self.alarm_id,
            "code": self.code,
            "message": self.message,
            "priority": self.priority.value,
            "category": self.category.value,
            "state": self.state.value,
            "device_serial": self.device_serial,
            "timestamp": self.timestamp,
            "parameter": self.parameter,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass
class AlarmDefinition:
    """Definition of an alarm condition."""
    code: str
    message_template: str
    priority: AlarmPriority
    category: AlarmCategory
    parameter: str
    condition: str  # "above", "below", "outside_range", "equals", "rate_of_change"
    threshold: float | tuple[float, float]
    debounce_seconds: float = 0.0  # Prevent alarm flapping
    auto_resolve: bool = True
    latching: bool = False
    escalation_timeout_minutes: float = 5.0


class AlarmManager:
    """
    Manages device alarms with prioritization and escalation.

    Implements IEC 60601-1-8 alarm system requirements.
    """

    def __init__(self):
        self._definitions: dict[str, AlarmDefinition] = {}
        self._active_alarms: dict[str, Alarm] = {}
        self._alarm_history: list[Alarm] = []
        self._handlers: dict[AlarmPriority, list[Callable]] = {
            AlarmPriority.LOW: [],
            AlarmPriority.MEDIUM: [],
            AlarmPriority.HIGH: [],
        }
        self._silenced_until: dict[str, datetime] = {}
        self._last_values: dict[str, tuple[float, datetime]] = {}
        self._alarm_counter = 0
        self._lock = threading.Lock()

    def register_alarm(self, definition: AlarmDefinition) -> None:
        """Register an alarm definition."""
        self._definitions[definition.code] = definition

    def register_handler(
        self,
        priority: AlarmPriority,
        handler: Callable[[Alarm], None],
    ) -> None:
        """Register a handler for alarms of a specific priority."""
        self._handlers[priority].append(handler)

    def check_condition(
        self,
        device_serial: str,
        parameter: str,
        value: float,
    ) -> list[Alarm]:
        """
        Check if a value triggers any alarm conditions.

        Returns list of triggered alarms.
        """
        triggered = []

        for code, definition in self._definitions.items():
            if definition.parameter != parameter:
                continue

            is_triggered = False
            threshold = definition.threshold

            if definition.condition == "above" and value > threshold:
                is_triggered = True
            elif definition.condition == "below" and value < threshold:
                is_triggered = True
            elif definition.condition == "outside_range":
                if isinstance(threshold, tuple):
                    if value < threshold[0] or value > threshold[1]:
                        is_triggered = True
            elif definition.condition == "equals" and value == threshold:
                is_triggered = True
            elif definition.condition == "rate_of_change":
                # Check rate of change
                key = f"{device_serial}_{parameter}"
                if key in self._last_values:
                    old_value, old_time = self._last_values[key]
                    dt = (datetime.now() - old_time).total_seconds()
                    if dt > 0:
                        rate = abs(value - old_value) / dt
                        if rate > threshold:
                            is_triggered = True
                self._last_values[key] = (value, datetime.now())

            if is_triggered:
                alarm = self._create_alarm(definition, device_serial, value)
                if alarm:
                    triggered.append(alarm)
            elif definition.auto_resolve:
                # Check if alarm should be resolved
                self._auto_resolve(code, device_serial)

        return triggered

    def _create_alarm(
        self,
        definition: AlarmDefinition,
        device_serial: str,
        value: float,
    ) -> Alarm | None:
        """Create an alarm if not already active."""
        alarm_key = f"{definition.code}_{device_serial}"

        with self._lock:
            # Check if alarm is already active
            if alarm_key in self._active_alarms:
                existing = self._active_alarms[alarm_key]
                if existing.state in (AlarmState.ACTIVE, AlarmState.ACKNOWLEDGED):
                    return None  # Don't duplicate

            # Check if silenced
            if alarm_key in self._silenced_until:
                if datetime.now() < self._silenced_until[alarm_key]:
                    return None

            self._alarm_counter += 1
            alarm_id = f"ALM-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._alarm_counter:04d}"

            message = definition.message_template.format(
                value=value,
                threshold=definition.threshold,
                parameter=definition.parameter,
            )

            alarm = Alarm(
                alarm_id=alarm_id,
                code=definition.code,
                message=message,
                priority=definition.priority,
                category=definition.category,
                device_serial=device_serial,
                parameter=definition.parameter,
                value=value,
                threshold=definition.threshold if isinstance(definition.threshold, float) else None,
            )

            if definition.latching:
                alarm.state = AlarmState.LATCHED

            self._active_alarms[alarm_key] = alarm
            self._notify_handlers(alarm)

            return alarm

    def _auto_resolve(self, code: str, device_serial: str) -> None:
        """Auto-resolve an alarm if condition no longer met."""
        alarm_key = f"{code}_{device_serial}"

        with self._lock:
            if alarm_key in self._active_alarms:
                alarm = self._active_alarms[alarm_key]
                if alarm.state != AlarmState.LATCHED:
                    alarm.resolve()
                    self._alarm_history.append(alarm)
                    del self._active_alarms[alarm_key]

    def _notify_handlers(self, alarm: Alarm) -> None:
        """Notify registered handlers of an alarm."""
        for handler in self._handlers[alarm.priority]:
            try:
                handler(alarm)
            except Exception:
                pass  # Handler errors shouldn't affect alarm system

    def acknowledge_alarm(self, alarm_id: str, user: str) -> bool:
        """Acknowledge an alarm by ID."""
        with self._lock:
            for alarm in self._active_alarms.values():
                if alarm.alarm_id == alarm_id:
                    alarm.acknowledge(user)
                    return True
        return False

    def silence_alarm(self, alarm_id: str, duration_minutes: int = 2) -> bool:
        """Silence an alarm temporarily."""
        with self._lock:
            for key, alarm in self._active_alarms.items():
                if alarm.alarm_id == alarm_id:
                    alarm.silence()
                    self._silenced_until[key] = datetime.now() + timedelta(minutes=duration_minutes)
                    return True
        return False

    def reset_latched_alarm(self, alarm_id: str, user: str) -> bool:
        """Reset a latched alarm (requires explicit action)."""
        with self._lock:
            for key, alarm in list(self._active_alarms.items()):
                if alarm.alarm_id == alarm_id and alarm.state == AlarmState.LATCHED:
                    alarm.acknowledge(user)
                    alarm.resolve()
                    self._alarm_history.append(alarm)
                    del self._active_alarms[key]
                    return True
        return False

    def get_active_alarms(
        self,
        device_serial: str | None = None,
        priority: AlarmPriority | None = None,
    ) -> list[Alarm]:
        """Get list of active alarms."""
        alarms = list(self._active_alarms.values())

        if device_serial:
            alarms = [a for a in alarms if a.device_serial == device_serial]

        if priority:
            alarms = [a for a in alarms if a.priority == priority]

        # Sort by priority (HIGH first) then timestamp
        priority_order = {AlarmPriority.HIGH: 0, AlarmPriority.MEDIUM: 1, AlarmPriority.LOW: 2}
        alarms.sort(key=lambda a: (priority_order[a.priority], a.timestamp))

        return alarms

    def get_alarm_history(
        self,
        device_serial: str | None = None,
        hours: float = 24.0,
    ) -> list[Alarm]:
        """Get alarm history."""
        cutoff = datetime.now() - timedelta(hours=hours)

        alarms = self._alarm_history.copy()

        if device_serial:
            alarms = [a for a in alarms if a.device_serial == device_serial]

        alarms = [
            a for a in alarms
            if datetime.fromisoformat(a.timestamp) > cutoff
        ]

        return alarms

    def get_alarm_statistics(self, device_serial: str, hours: float = 24.0) -> dict:
        """Get alarm statistics for a device."""
        history = self.get_alarm_history(device_serial, hours)
        active = self.get_active_alarms(device_serial)

        return {
            "device_serial": device_serial,
            "period_hours": hours,
            "active_count": len(active),
            "active_high": len([a for a in active if a.priority == AlarmPriority.HIGH]),
            "active_medium": len([a for a in active if a.priority == AlarmPriority.MEDIUM]),
            "active_low": len([a for a in active if a.priority == AlarmPriority.LOW]),
            "historical_count": len(history),
            "by_category": {
                cat.value: len([a for a in history if a.category == cat])
                for cat in AlarmCategory
            },
        }


class SafetyInterlock:
    """
    Safety interlock system to prevent hazardous conditions.

    Implements fail-safe mechanisms for critical operations.
    """

    def __init__(self, name: str):
        self.name = name
        self._conditions: list[Callable[[], bool]] = []
        self._is_locked = False
        self._lock_reason: str | None = None
        self._bypass_enabled = False
        self._bypass_user: str | None = None

    def add_condition(self, condition: Callable[[], bool], description: str) -> None:
        """
        Add a safety condition.

        The condition should return True when safe, False when unsafe.
        """
        self._conditions.append((condition, description))

    def check(self) -> tuple[bool, str | None]:
        """
        Check all interlock conditions.

        Returns:
            Tuple of (is_safe, failure_reason)
        """
        if self._bypass_enabled:
            return True, "BYPASSED"

        for condition, description in self._conditions:
            try:
                if not condition():
                    self._is_locked = True
                    self._lock_reason = description
                    return False, description
            except Exception as e:
                self._is_locked = True
                self._lock_reason = f"Condition check failed: {e}"
                return False, self._lock_reason

        self._is_locked = False
        self._lock_reason = None
        return True, None

    def bypass(self, user: str, reason: str) -> None:
        """
        Enable interlock bypass (for authorized emergency use only).

        This should be logged and audited.
        """
        self._bypass_enabled = True
        self._bypass_user = user
        # In production, this would trigger audit logging

    def clear_bypass(self) -> None:
        """Clear interlock bypass."""
        self._bypass_enabled = False
        self._bypass_user = None

    @property
    def is_locked(self) -> bool:
        """Check if interlock is currently locked."""
        return self._is_locked

    @property
    def status(self) -> dict:
        """Get interlock status."""
        return {
            "name": self.name,
            "is_locked": self._is_locked,
            "lock_reason": self._lock_reason,
            "bypass_enabled": self._bypass_enabled,
            "bypass_user": self._bypass_user,
        }


class SafetyMonitor:
    """
    Continuous safety monitoring system.

    Monitors device operation and triggers protective actions.
    """

    def __init__(self, device):
        self._device = device
        self._interlocks: dict[str, SafetyInterlock] = {}
        self._alarm_manager = AlarmManager()
        self._protective_actions: dict[str, Callable] = {}
        self._monitoring_active = False

    def add_interlock(self, interlock: SafetyInterlock) -> None:
        """Add a safety interlock."""
        self._interlocks[interlock.name] = interlock

    def register_protective_action(
        self,
        trigger: str,
        action: Callable,
    ) -> None:
        """Register a protective action for a trigger condition."""
        self._protective_actions[trigger] = action

    def check_all_interlocks(self) -> dict[str, dict]:
        """Check all registered interlocks."""
        results = {}
        for name, interlock in self._interlocks.items():
            is_safe, reason = interlock.check()
            results[name] = {
                "safe": is_safe,
                "reason": reason,
                "status": interlock.status,
            }
        return results

    def process_telemetry(self, parameter: str, value: float) -> list[Alarm]:
        """Process telemetry and check for alarm conditions."""
        device_serial = getattr(self._device, 'model', 'unknown')
        alarms = self._alarm_manager.check_condition(device_serial, parameter, value)

        # Trigger protective actions for high-priority alarms
        for alarm in alarms:
            if alarm.priority == AlarmPriority.HIGH:
                action_key = f"high_alarm_{alarm.code}"
                if action_key in self._protective_actions:
                    self._protective_actions[action_key]()

        return alarms

    def get_safety_summary(self) -> dict:
        """Get comprehensive safety status summary."""
        interlock_status = self.check_all_interlocks()
        active_alarms = self._alarm_manager.get_active_alarms()

        all_interlocks_ok = all(s["safe"] for s in interlock_status.values())
        critical_alarms = [a for a in active_alarms if a.priority == AlarmPriority.HIGH]

        return {
            "overall_safe": all_interlocks_ok and len(critical_alarms) == 0,
            "interlocks_ok": all_interlocks_ok,
            "interlock_details": interlock_status,
            "active_alarms": len(active_alarms),
            "critical_alarms": len(critical_alarms),
            "alarm_summary": [a.to_dict() for a in active_alarms[:10]],
        }


@dataclass
class HazardEntry:
    """Entry in a hazard analysis."""
    hazard_id: str
    description: str
    cause: str
    severity: str  # catastrophic, critical, serious, minor
    probability: str  # frequent, probable, occasional, remote, improbable
    risk_level: str  # high, medium, low
    mitigation: str
    verification: str
    residual_risk: str


class HazardAnalysis:
    """
    Hazard analysis tool for medical device risk management.

    Supports ISO 14971 risk management process.
    """

    def __init__(self, device_name: str):
        self.device_name = device_name
        self._hazards: list[HazardEntry] = []

    def add_hazard(self, hazard: HazardEntry) -> None:
        """Add a hazard to the analysis."""
        self._hazards.append(hazard)

    def get_hazards_by_risk(self, risk_level: str) -> list[HazardEntry]:
        """Get hazards filtered by risk level."""
        return [h for h in self._hazards if h.risk_level == risk_level]

    def calculate_risk_matrix(self) -> dict:
        """Calculate risk matrix summary."""
        severity_order = ["catastrophic", "critical", "serious", "minor"]
        probability_order = ["frequent", "probable", "occasional", "remote", "improbable"]

        matrix = {}
        for severity in severity_order:
            matrix[severity] = {}
            for probability in probability_order:
                count = len([
                    h for h in self._hazards
                    if h.severity == severity and h.probability == probability
                ])
                matrix[severity][probability] = count

        return matrix

    def generate_report(self) -> str:
        """Generate hazard analysis report."""
        lines = [
            f"HAZARD ANALYSIS REPORT: {self.device_name}",
            "=" * 60,
            f"Total Hazards Identified: {len(self._hazards)}",
            f"High Risk: {len(self.get_hazards_by_risk('high'))}",
            f"Medium Risk: {len(self.get_hazards_by_risk('medium'))}",
            f"Low Risk: {len(self.get_hazards_by_risk('low'))}",
            "",
            "DETAILED HAZARDS",
            "-" * 60,
        ]

        for hazard in self._hazards:
            lines.extend([
                f"ID: {hazard.hazard_id}",
                f"Description: {hazard.description}",
                f"Cause: {hazard.cause}",
                f"Severity: {hazard.severity} | Probability: {hazard.probability}",
                f"Risk Level: {hazard.risk_level}",
                f"Mitigation: {hazard.mitigation}",
                f"Residual Risk: {hazard.residual_risk}",
                "-" * 40,
            ])

        return "\n".join(lines)


# Pre-defined alarm definitions for common device conditions
class StandardAlarms:
    """Standard alarm definitions for implantable devices."""

    # LVAD Alarms
    LVAD_LOW_FLOW = AlarmDefinition(
        code="LVAD001",
        message_template="Low pump flow: {value:.1f} L/min (threshold: {threshold} L/min)",
        priority=AlarmPriority.HIGH,
        category=AlarmCategory.PHYSIOLOGICAL,
        parameter="flow_lpm",
        condition="below",
        threshold=2.0,
        latching=True,
    )

    LVAD_HIGH_POWER = AlarmDefinition(
        code="LVAD002",
        message_template="High pump power: {value:.1f} W (threshold: {threshold} W)",
        priority=AlarmPriority.MEDIUM,
        category=AlarmCategory.TECHNICAL,
        parameter="power_watts",
        condition="above",
        threshold=12.0,
    )

    LVAD_LOW_BATTERY = AlarmDefinition(
        code="LVAD003",
        message_template="Low battery: {value:.0f}% (threshold: {threshold}%)",
        priority=AlarmPriority.HIGH,
        category=AlarmCategory.TECHNICAL,
        parameter="battery_percent",
        condition="below",
        threshold=20.0,
    )

    # DBS Alarms
    DBS_HIGH_IMPEDANCE = AlarmDefinition(
        code="DBS001",
        message_template="High electrode impedance: {value:.0f} Ω",
        priority=AlarmPriority.MEDIUM,
        category=AlarmCategory.TECHNICAL,
        parameter="impedance_ohms",
        condition="above",
        threshold=3000.0,
    )

    DBS_LOW_BATTERY = AlarmDefinition(
        code="DBS002",
        message_template="Low battery: {value:.0f}%",
        priority=AlarmPriority.MEDIUM,
        category=AlarmCategory.TECHNICAL,
        parameter="battery_percent",
        condition="below",
        threshold=25.0,
    )

    # Retinal Implant Alarms
    RETINAL_LINK_LOST = AlarmDefinition(
        code="RET001",
        message_template="Wireless link quality degraded: {value:.0f}%",
        priority=AlarmPriority.HIGH,
        category=AlarmCategory.TECHNICAL,
        parameter="link_quality",
        condition="below",
        threshold=30.0,
    )

    @classmethod
    def get_lvad_alarms(cls) -> list[AlarmDefinition]:
        """Get all LVAD alarm definitions."""
        return [cls.LVAD_LOW_FLOW, cls.LVAD_HIGH_POWER, cls.LVAD_LOW_BATTERY]

    @classmethod
    def get_dbs_alarms(cls) -> list[AlarmDefinition]:
        """Get all DBS alarm definitions."""
        return [cls.DBS_HIGH_IMPEDANCE, cls.DBS_LOW_BATTERY]

    @classmethod
    def get_retinal_alarms(cls) -> list[AlarmDefinition]:
        """Get all retinal implant alarm definitions."""
        return [cls.RETINAL_LINK_LOST]

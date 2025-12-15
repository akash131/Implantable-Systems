"""
Clinical Programming Interface for Implantable Devices.

Provides clinical workflow support including:
- Programming sessions with audit trails
- Parameter adjustment with safety validation
- Patient profiles and history
- Follow-up scheduling
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Any
from abc import ABC, abstractmethod
import json
import hashlib


class ProgrammingMode(Enum):
    """Clinical programming modes."""
    READ_ONLY = "read_only"
    PROGRAMMING = "programming"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"


class AdjustmentType(Enum):
    """Types of parameter adjustments."""
    INCREASE = "increase"
    DECREASE = "decrease"
    SET_VALUE = "set_value"
    RESET = "reset"


@dataclass
class Clinician:
    """Represents a clinical user."""
    id: str
    name: str
    credentials: str  # e.g., "MD", "NP", "PA"
    institution: str
    npi: Optional[str] = None  # National Provider Identifier
    specialization: str = "Electrophysiology"
    permissions: list[str] = field(default_factory=lambda: ["read", "program"])

    def has_permission(self, permission: str) -> bool:
        """Check if clinician has specific permission."""
        return permission in self.permissions or "admin" in self.permissions


@dataclass
class Patient:
    """Patient information for device management."""
    id: str
    mrn: str  # Medical Record Number
    name: str
    date_of_birth: str
    implant_date: str
    device_serial: str
    indication: str  # Primary indication for device
    allergies: list[str] = field(default_factory=list)
    comorbidities: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    emergency_contact: Optional[str] = None

    @property
    def time_since_implant(self) -> timedelta:
        """Calculate time since device implantation."""
        implant = datetime.fromisoformat(self.implant_date)
        return datetime.now() - implant


@dataclass
class ParameterChange:
    """Records a single parameter change."""
    parameter_name: str
    old_value: Any
    new_value: Any
    adjustment_type: AdjustmentType
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    rationale: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "parameter": self.parameter_name,
            "old_value": self.old_value,
            "new_value": self.new_value,
            "adjustment_type": self.adjustment_type.value,
            "timestamp": self.timestamp,
            "rationale": self.rationale,
        }


@dataclass
class ProgrammingSession:
    """
    Represents a clinical programming session.

    Tracks all changes made during a session with full audit trail.
    """
    session_id: str
    patient: Patient
    clinician: Clinician
    device_serial: str
    start_time: str = field(default_factory=lambda: datetime.now().isoformat())
    end_time: Optional[str] = None
    mode: ProgrammingMode = ProgrammingMode.READ_ONLY
    changes: list[ParameterChange] = field(default_factory=list)
    notes: str = ""
    telemetry_snapshot: Optional[dict] = None
    is_complete: bool = False

    def record_change(
        self,
        parameter: str,
        old_value: Any,
        new_value: Any,
        adjustment_type: AdjustmentType,
        rationale: str = "",
    ) -> None:
        """Record a parameter change."""
        change = ParameterChange(
            parameter_name=parameter,
            old_value=old_value,
            new_value=new_value,
            adjustment_type=adjustment_type,
            rationale=rationale,
        )
        self.changes.append(change)

    def complete_session(self, notes: str = "") -> None:
        """Mark session as complete."""
        self.end_time = datetime.now().isoformat()
        self.notes = notes
        self.is_complete = True

    def get_duration_minutes(self) -> float:
        """Get session duration in minutes."""
        start = datetime.fromisoformat(self.start_time)
        if self.end_time:
            end = datetime.fromisoformat(self.end_time)
        else:
            end = datetime.now()
        return (end - start).total_seconds() / 60

    def generate_report(self) -> dict:
        """Generate session report."""
        return {
            "session_id": self.session_id,
            "patient_id": self.patient.id,
            "patient_name": self.patient.name,
            "clinician": self.clinician.name,
            "institution": self.clinician.institution,
            "device_serial": self.device_serial,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_minutes": self.get_duration_minutes(),
            "mode": self.mode.value,
            "total_changes": len(self.changes),
            "changes": [c.to_dict() for c in self.changes],
            "notes": self.notes,
        }

    def get_signature(self) -> str:
        """Generate cryptographic signature of session data."""
        data = json.dumps(self.generate_report(), sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()


class SafetyValidator:
    """
    Validates parameter changes against safety limits.

    Implements IEC 62304 and FDA guidance for medical device software.
    """

    def __init__(self):
        self._safety_limits: dict[str, dict] = {}
        self._rate_limits: dict[str, dict] = {}
        self._initialize_default_limits()

    def _initialize_default_limits(self) -> None:
        """Set default safety limits for common parameters."""
        # LVAD limits
        self._safety_limits["lvad_speed_rpm"] = {"min": 2000, "max": 9000}
        self._safety_limits["lvad_flow_target_lpm"] = {"min": 2.0, "max": 8.0}

        # DBS limits
        self._safety_limits["dbs_amplitude_ma"] = {"min": 0.0, "max": 10.0}
        self._safety_limits["dbs_pulse_width_us"] = {"min": 30, "max": 450}
        self._safety_limits["dbs_frequency_hz"] = {"min": 2, "max": 250}

        # Retinal implant limits
        self._safety_limits["retinal_current_ua"] = {"min": 0, "max": 500}
        self._safety_limits["retinal_pulse_width_us"] = {"min": 50, "max": 1000}

        # Rate limits (max change per adjustment)
        self._rate_limits["lvad_speed_rpm"] = {"max_change": 500}
        self._rate_limits["dbs_amplitude_ma"] = {"max_change": 0.5}

    def add_safety_limit(
        self,
        parameter: str,
        min_value: float,
        max_value: float,
    ) -> None:
        """Add or update a safety limit."""
        self._safety_limits[parameter] = {"min": min_value, "max": max_value}

    def add_rate_limit(self, parameter: str, max_change: float) -> None:
        """Add or update a rate limit."""
        self._rate_limits[parameter] = {"max_change": max_change}

    def validate_value(self, parameter: str, value: float) -> tuple[bool, str]:
        """
        Validate a parameter value against safety limits.

        Returns:
            Tuple of (is_valid, message)
        """
        if parameter not in self._safety_limits:
            return True, "No safety limits defined"

        limits = self._safety_limits[parameter]

        if value < limits["min"]:
            return False, f"{parameter} value {value} below minimum {limits['min']}"

        if value > limits["max"]:
            return False, f"{parameter} value {value} exceeds maximum {limits['max']}"

        return True, "Within safety limits"

    def validate_change(
        self,
        parameter: str,
        old_value: float,
        new_value: float,
    ) -> tuple[bool, str]:
        """
        Validate a parameter change.

        Checks both absolute limits and rate limits.
        """
        # Check absolute limits
        is_valid, message = self.validate_value(parameter, new_value)
        if not is_valid:
            return False, message

        # Check rate limits
        if parameter in self._rate_limits:
            max_change = self._rate_limits[parameter]["max_change"]
            actual_change = abs(new_value - old_value)

            if actual_change > max_change:
                return False, (
                    f"Change of {actual_change} exceeds maximum "
                    f"single-step change of {max_change}"
                )

        return True, "Change validated"


class ClinicalProgrammer(ABC):
    """
    Abstract base class for clinical device programmers.

    Provides common functionality for device programming interfaces.
    """

    def __init__(self):
        self._validator = SafetyValidator()
        self._current_session: Optional[ProgrammingSession] = None
        self._session_history: list[ProgrammingSession] = []

    @property
    def is_session_active(self) -> bool:
        """Check if a programming session is active."""
        return self._current_session is not None and not self._current_session.is_complete

    def start_session(
        self,
        patient: Patient,
        clinician: Clinician,
        device_serial: str,
        mode: ProgrammingMode = ProgrammingMode.READ_ONLY,
    ) -> ProgrammingSession:
        """Start a new programming session."""
        if self.is_session_active:
            raise RuntimeError("A session is already active")

        if not clinician.has_permission("read"):
            raise PermissionError("Clinician lacks read permission")

        if mode == ProgrammingMode.PROGRAMMING and not clinician.has_permission("program"):
            raise PermissionError("Clinician lacks programming permission")

        session_id = f"SES-{datetime.now().strftime('%Y%m%d%H%M%S')}-{device_serial[:8]}"

        self._current_session = ProgrammingSession(
            session_id=session_id,
            patient=patient,
            clinician=clinician,
            device_serial=device_serial,
            mode=mode,
        )

        return self._current_session

    def end_session(self, notes: str = "") -> dict:
        """End the current session and return report."""
        if not self.is_session_active:
            raise RuntimeError("No active session")

        self._current_session.complete_session(notes)
        report = self._current_session.generate_report()

        self._session_history.append(self._current_session)
        self._current_session = None

        return report

    @abstractmethod
    def read_device_parameters(self) -> dict:
        """Read all current device parameters."""
        pass

    @abstractmethod
    def apply_parameter(
        self,
        parameter: str,
        value: Any,
        rationale: str = "",
    ) -> bool:
        """Apply a parameter change to the device."""
        pass

    def get_session_history(self, patient_id: Optional[str] = None) -> list[dict]:
        """Get history of programming sessions."""
        sessions = self._session_history

        if patient_id:
            sessions = [s for s in sessions if s.patient.id == patient_id]

        return [s.generate_report() for s in sessions]


class LVADProgrammer(ClinicalProgrammer):
    """Clinical programmer for LVAD devices."""

    def __init__(self, device):
        super().__init__()
        self._device = device

    def read_device_parameters(self) -> dict:
        """Read current LVAD parameters."""
        status = self._device.get_comprehensive_status()

        params = {
            "device_state": status.get("device_state"),
            "flow_type": status.get("flow_type"),
            "current_flow_lpm": status.get("current_flow_lpm"),
            "current_power_watts": status.get("current_power_watts"),
            "battery_level": status.get("power_status", {}).get("charge_level_percent"),
        }

        if hasattr(self._device, '_controller'):
            params["target_flow_lpm"] = self._device._controller.target_flow_lpm
            params["min_speed_rpm"] = self._device._controller.min_speed_rpm
            params["max_speed_rpm"] = self._device._controller.max_speed_rpm

        if hasattr(self._device, '_pump'):
            params["current_speed_rpm"] = getattr(self._device._pump, '_current_speed', 0)

        if self._current_session:
            self._current_session.telemetry_snapshot = params

        return params

    def apply_parameter(
        self,
        parameter: str,
        value: Any,
        rationale: str = "",
    ) -> bool:
        """Apply parameter change to LVAD."""
        if not self.is_session_active:
            raise RuntimeError("No active session")

        if self._current_session.mode == ProgrammingMode.READ_ONLY:
            raise PermissionError("Session is read-only")

        # Get current value
        current_params = self.read_device_parameters()
        old_value = current_params.get(parameter)

        # Validate change
        if isinstance(value, (int, float)):
            is_valid, message = self._validator.validate_change(
                f"lvad_{parameter}", old_value or 0, value
            )
            if not is_valid:
                raise ValueError(message)

        # Apply change
        success = False

        if parameter == "target_flow_lpm" and hasattr(self._device, '_controller'):
            self._device._controller.target_flow_lpm = value
            success = True

        elif parameter == "min_speed_rpm" and hasattr(self._device, '_controller'):
            self._device._controller.min_speed_rpm = value
            success = True

        elif parameter == "max_speed_rpm" and hasattr(self._device, '_controller'):
            self._device._controller.max_speed_rpm = value
            success = True

        if success:
            self._current_session.record_change(
                parameter=parameter,
                old_value=old_value,
                new_value=value,
                adjustment_type=AdjustmentType.SET_VALUE,
                rationale=rationale,
            )

        return success


class DBSProgrammer(ClinicalProgrammer):
    """Clinical programmer for DBS devices."""

    def __init__(self, device):
        super().__init__()
        self._device = device

    def read_device_parameters(self) -> dict:
        """Read current DBS parameters."""
        status = self._device.get_comprehensive_status()

        params = {
            "device_state": status.get("device_state"),
            "target_region": status.get("target_region"),
            "stimulation_active": status.get("stimulation_active"),
            "mode": status.get("mode"),
            "battery_level": status.get("battery_status", {}).get("charge_level_percent"),
        }

        # Add stimulation parameters
        if "right_parameters" in status:
            rp = status["right_parameters"]
            params["right_amplitude_ma"] = rp.get("amplitude_ma")
            params["right_pulse_width_us"] = rp.get("pulse_width_us")
            params["right_frequency_hz"] = rp.get("frequency_hz")

        if self._current_session:
            self._current_session.telemetry_snapshot = params

        return params

    def apply_parameter(
        self,
        parameter: str,
        value: Any,
        rationale: str = "",
    ) -> bool:
        """Apply parameter change to DBS."""
        if not self.is_session_active:
            raise RuntimeError("No active session")

        if self._current_session.mode == ProgrammingMode.READ_ONLY:
            raise PermissionError("Session is read-only")

        current_params = self.read_device_parameters()
        old_value = current_params.get(parameter)

        # Validate
        if isinstance(value, (int, float)):
            is_valid, message = self._validator.validate_change(
                f"dbs_{parameter}", old_value or 0, value
            )
            if not is_valid:
                raise ValueError(message)

        success = False

        # Apply changes based on parameter
        if parameter in ["right_amplitude_ma", "right_pulse_width_us", "right_frequency_hz"]:
            # Update stimulation parameters
            from .deep_brain_stimulator import StimulationParameters

            current_right = self._device._stim_params_right
            if current_right is None:
                current_right = StimulationParameters()

            if parameter == "right_amplitude_ma":
                new_params = StimulationParameters(
                    amplitude_ma=value,
                    pulse_width_us=current_right.pulse_width_us,
                    frequency_hz=current_right.frequency_hz,
                    active_contacts=current_right.active_contacts,
                )
            elif parameter == "right_pulse_width_us":
                new_params = StimulationParameters(
                    amplitude_ma=current_right.amplitude_ma,
                    pulse_width_us=value,
                    frequency_hz=current_right.frequency_hz,
                    active_contacts=current_right.active_contacts,
                )
            elif parameter == "right_frequency_hz":
                new_params = StimulationParameters(
                    amplitude_ma=current_right.amplitude_ma,
                    pulse_width_us=current_right.pulse_width_us,
                    frequency_hz=value,
                    active_contacts=current_right.active_contacts,
                )

            success = self._device.set_stimulation_parameters(new_params, side="right")

        if success:
            self._current_session.record_change(
                parameter=parameter,
                old_value=old_value,
                new_value=value,
                adjustment_type=AdjustmentType.SET_VALUE,
                rationale=rationale,
            )

        return success

    def run_impedance_check(self) -> dict:
        """Run electrode impedance check."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "right_lead": {},
            "left_lead": {},
        }

        if self._device._right_lead:
            impedances = self._device._right_lead.measure_all_impedances()
            results["right_lead"] = {
                f"contact_{i}": z for i, z in enumerate(impedances)
            }

        if self._device._left_lead:
            impedances = self._device._left_lead.measure_all_impedances()
            results["left_lead"] = {
                f"contact_{i}": z for i, z in enumerate(impedances)
            }

        return results


@dataclass
class FollowUpSchedule:
    """Follow-up appointment scheduling."""
    patient_id: str
    device_serial: str
    scheduled_date: str
    visit_type: str  # "routine", "urgent", "post_adjustment"
    clinic_location: str
    assigned_clinician: Optional[str] = None
    notes: str = ""
    completed: bool = False

    @classmethod
    def schedule_routine(
        cls,
        patient: Patient,
        interval_days: int = 90,
        clinic: str = "Main Clinic",
    ) -> "FollowUpSchedule":
        """Schedule a routine follow-up."""
        next_date = datetime.now() + timedelta(days=interval_days)
        return cls(
            patient_id=patient.id,
            device_serial=patient.device_serial,
            scheduled_date=next_date.isoformat(),
            visit_type="routine",
            clinic_location=clinic,
        )


class ClinicWorkflow:
    """
    Manages clinical workflow for device follow-ups.

    Coordinates programming sessions, follow-up scheduling, and patient management.
    """

    def __init__(self):
        self._patients: dict[str, Patient] = {}
        self._clinicians: dict[str, Clinician] = {}
        self._schedules: list[FollowUpSchedule] = []
        self._programmers: dict[str, ClinicalProgrammer] = {}

    def register_patient(self, patient: Patient) -> None:
        """Register a patient in the system."""
        self._patients[patient.id] = patient

    def register_clinician(self, clinician: Clinician) -> None:
        """Register a clinician in the system."""
        self._clinicians[clinician.id] = clinician

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Get patient by ID."""
        return self._patients.get(patient_id)

    def get_clinician(self, clinician_id: str) -> Optional[Clinician]:
        """Get clinician by ID."""
        return self._clinicians.get(clinician_id)

    def schedule_followup(
        self,
        patient_id: str,
        visit_type: str,
        days_from_now: int,
        clinic: str,
    ) -> FollowUpSchedule:
        """Schedule a follow-up visit."""
        patient = self._patients.get(patient_id)
        if not patient:
            raise ValueError(f"Unknown patient: {patient_id}")

        schedule = FollowUpSchedule(
            patient_id=patient_id,
            device_serial=patient.device_serial,
            scheduled_date=(datetime.now() + timedelta(days=days_from_now)).isoformat(),
            visit_type=visit_type,
            clinic_location=clinic,
        )
        self._schedules.append(schedule)
        return schedule

    def get_upcoming_appointments(self, days: int = 30) -> list[FollowUpSchedule]:
        """Get appointments scheduled within the next N days."""
        cutoff = datetime.now() + timedelta(days=days)
        upcoming = []

        for schedule in self._schedules:
            if schedule.completed:
                continue
            scheduled = datetime.fromisoformat(schedule.scheduled_date)
            if scheduled <= cutoff:
                upcoming.append(schedule)

        return sorted(upcoming, key=lambda s: s.scheduled_date)

    def create_session_report(self, session: ProgrammingSession) -> str:
        """Generate a formatted clinical report."""
        report = session.generate_report()

        lines = [
            "=" * 60,
            "DEVICE PROGRAMMING SESSION REPORT",
            "=" * 60,
            f"Session ID: {report['session_id']}",
            f"Date: {report['start_time'][:10]}",
            f"Duration: {report['duration_minutes']:.1f} minutes",
            "",
            "PATIENT INFORMATION",
            "-" * 40,
            f"Name: {report['patient_name']}",
            f"Patient ID: {report['patient_id']}",
            "",
            "CLINICIAN",
            "-" * 40,
            f"Name: {report['clinician']}",
            f"Institution: {report['institution']}",
            "",
            "DEVICE",
            "-" * 40,
            f"Serial: {report['device_serial']}",
            f"Mode: {report['mode']}",
            "",
            "PARAMETER CHANGES",
            "-" * 40,
        ]

        if report['changes']:
            for change in report['changes']:
                lines.append(
                    f"  {change['parameter']}: {change['old_value']} → {change['new_value']}"
                )
                if change['rationale']:
                    lines.append(f"    Rationale: {change['rationale']}")
        else:
            lines.append("  No changes made")

        lines.extend([
            "",
            "NOTES",
            "-" * 40,
            report['notes'] or "None",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

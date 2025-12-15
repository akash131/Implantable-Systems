"""
Base classes for implantable medical device systems.

Provides foundational abstractions for all implantable devices including
power management, control systems, and electrode interfaces.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import math


class DeviceState(Enum):
    """Operating states for implantable devices."""
    OFF = "off"
    STANDBY = "standby"
    ACTIVE = "active"
    LOW_POWER = "low_power"
    CHARGING = "charging"
    ERROR = "error"
    MRI_SAFE = "mri_safe"


class BiocompatibilityRating(Enum):
    """ISO 10993 biocompatibility classification."""
    CLASS_I = "limited_contact"      # < 24 hours
    CLASS_II = "prolonged_contact"   # 24 hours to 30 days
    CLASS_III = "permanent_contact"  # > 30 days


@dataclass
class MaterialProperties:
    """Properties of biocompatible materials used in implants."""
    name: str
    biocompatibility_rating: BiocompatibilityRating
    corrosion_resistance: float  # 0-1 scale
    mri_compatibility: bool
    thermal_conductivity: float  # W/(m·K)
    electrical_conductivity: float  # S/m
    youngs_modulus: float  # GPa

    @classmethod
    def titanium_grade5(cls) -> "MaterialProperties":
        """Ti-6Al-4V commonly used in implants."""
        return cls(
            name="Titanium Grade 5 (Ti-6Al-4V)",
            biocompatibility_rating=BiocompatibilityRating.CLASS_III,
            corrosion_resistance=0.95,
            mri_compatibility=True,
            thermal_conductivity=6.7,
            electrical_conductivity=5.8e5,
            youngs_modulus=114.0
        )

    @classmethod
    def platinum_iridium(cls) -> "MaterialProperties":
        """Pt-Ir alloy for electrodes."""
        return cls(
            name="Platinum-Iridium (90/10)",
            biocompatibility_rating=BiocompatibilityRating.CLASS_III,
            corrosion_resistance=0.99,
            mri_compatibility=True,
            thermal_conductivity=71.6,
            electrical_conductivity=9.4e6,
            youngs_modulus=200.0
        )

    @classmethod
    def silicone_medical(cls) -> "MaterialProperties":
        """Medical grade silicone for encapsulation."""
        return cls(
            name="Medical Grade Silicone",
            biocompatibility_rating=BiocompatibilityRating.CLASS_III,
            corrosion_resistance=0.98,
            mri_compatibility=True,
            thermal_conductivity=0.2,
            electrical_conductivity=1e-14,
            youngs_modulus=0.01
        )


@dataclass
class PhysiologicalState:
    """Represents patient physiological parameters."""
    heart_rate: float = 70.0  # bpm
    blood_pressure_systolic: float = 120.0  # mmHg
    blood_pressure_diastolic: float = 80.0  # mmHg
    cardiac_output: float = 5.0  # L/min
    oxygen_saturation: float = 98.0  # %
    body_temperature: float = 37.0  # Celsius
    activity_level: float = 0.0  # 0-1 (rest to max exercise)

    @property
    def mean_arterial_pressure(self) -> float:
        """Calculate mean arterial pressure (MAP)."""
        return self.blood_pressure_diastolic + (
            (self.blood_pressure_systolic - self.blood_pressure_diastolic) / 3
        )


class ImplantableDevice(ABC):
    """
    Abstract base class for all implantable medical devices.

    Provides common functionality for power management, state control,
    biocompatibility assessment, and safety monitoring.
    """

    def __init__(
        self,
        name: str,
        manufacturer: str,
        model: str,
        materials: list[MaterialProperties] | None = None,
    ):
        self.name = name
        self.manufacturer = manufacturer
        self.model = model
        self.materials = materials or []
        self._state = DeviceState.OFF
        self._power_system: Optional["PowerSystem"] = None
        self._error_log: list[str] = []
        self._telemetry_data: list[dict] = []

    @property
    def state(self) -> DeviceState:
        """Current operating state of the device."""
        return self._state

    @state.setter
    def state(self, new_state: DeviceState) -> None:
        if self._validate_state_transition(new_state):
            self._state = new_state
        else:
            raise ValueError(f"Invalid state transition: {self._state} -> {new_state}")

    def _validate_state_transition(self, new_state: DeviceState) -> bool:
        """Validate that the state transition is allowed."""
        # Define allowed transitions
        allowed_transitions = {
            DeviceState.OFF: {DeviceState.STANDBY, DeviceState.ERROR},
            DeviceState.STANDBY: {DeviceState.ACTIVE, DeviceState.OFF, DeviceState.CHARGING, DeviceState.MRI_SAFE},
            DeviceState.ACTIVE: {DeviceState.STANDBY, DeviceState.LOW_POWER, DeviceState.ERROR, DeviceState.MRI_SAFE},
            DeviceState.LOW_POWER: {DeviceState.ACTIVE, DeviceState.STANDBY, DeviceState.CHARGING, DeviceState.ERROR},
            DeviceState.CHARGING: {DeviceState.STANDBY, DeviceState.ACTIVE, DeviceState.ERROR},
            DeviceState.ERROR: {DeviceState.OFF, DeviceState.STANDBY},
            DeviceState.MRI_SAFE: {DeviceState.STANDBY, DeviceState.ACTIVE},
        }
        return new_state in allowed_transitions.get(self._state, set())

    @abstractmethod
    def activate(self) -> bool:
        """Activate the device. Returns True if successful."""
        pass

    @abstractmethod
    def deactivate(self) -> bool:
        """Deactivate the device. Returns True if successful."""
        pass

    @abstractmethod
    def perform_self_test(self) -> dict[str, bool]:
        """Perform device self-test and return results."""
        pass

    def get_biocompatibility_assessment(self) -> dict:
        """Assess overall biocompatibility of the device."""
        if not self.materials:
            return {"status": "unknown", "materials": []}

        worst_rating = max(
            m.biocompatibility_rating.value for m in self.materials
        )
        avg_corrosion = sum(m.corrosion_resistance for m in self.materials) / len(self.materials)
        mri_safe = all(m.mri_compatibility for m in self.materials)

        return {
            "status": "assessed",
            "worst_biocompatibility": worst_rating,
            "average_corrosion_resistance": avg_corrosion,
            "mri_compatible": mri_safe,
            "materials": [m.name for m in self.materials]
        }

    def log_error(self, error_message: str) -> None:
        """Log an error to the device error log."""
        from datetime import datetime
        timestamp = datetime.now().isoformat()
        self._error_log.append(f"[{timestamp}] {error_message}")

    def record_telemetry(self, data: dict) -> None:
        """Record telemetry data for later analysis."""
        from datetime import datetime
        data["timestamp"] = datetime.now().isoformat()
        self._telemetry_data.append(data)

    def enter_mri_safe_mode(self) -> bool:
        """
        Enter MRI-safe mode if supported.

        This typically involves:
        - Reducing or stopping stimulation
        - Entering a passive monitoring state
        - Minimizing electromagnetic interference
        """
        assessment = self.get_biocompatibility_assessment()
        if not assessment.get("mri_compatible", False):
            self.log_error("Cannot enter MRI-safe mode: device contains non-MRI-compatible materials")
            return False

        try:
            self.state = DeviceState.MRI_SAFE
            return True
        except ValueError as e:
            self.log_error(f"Failed to enter MRI-safe mode: {e}")
            return False


class PowerSystem(ABC):
    """
    Abstract base class for implantable device power systems.

    Handles battery management, wireless charging, and power distribution.
    """

    def __init__(
        self,
        capacity_wh: float,
        nominal_voltage: float,
        max_discharge_rate: float,
    ):
        self.capacity_wh = capacity_wh
        self.nominal_voltage = nominal_voltage
        self.max_discharge_rate = max_discharge_rate
        self._current_charge_wh = capacity_wh
        self._cycle_count = 0
        self._is_charging = False

    @property
    def charge_level(self) -> float:
        """Current charge level as percentage (0-100)."""
        return (self._current_charge_wh / self.capacity_wh) * 100

    @property
    def remaining_runtime_hours(self) -> float:
        """Estimated remaining runtime in hours at current discharge rate."""
        if self._current_discharge_rate <= 0:
            return float('inf')
        return self._current_charge_wh / self._current_discharge_rate

    @property
    def _current_discharge_rate(self) -> float:
        """Current power draw in watts."""
        return 0.0  # Override in subclass

    @abstractmethod
    def charge(self, power_watts: float, duration_hours: float) -> float:
        """
        Charge the battery.

        Args:
            power_watts: Charging power in watts
            duration_hours: Charging duration in hours

        Returns:
            Final charge level as percentage
        """
        pass

    @abstractmethod
    def discharge(self, power_watts: float, duration_hours: float) -> bool:
        """
        Discharge the battery.

        Args:
            power_watts: Power draw in watts
            duration_hours: Duration in hours

        Returns:
            True if discharge was successful, False if insufficient charge
        """
        pass

    def get_health_status(self) -> dict:
        """Get battery health status."""
        # Simplified capacity fade model based on cycle count
        capacity_fade = 1.0 - (self._cycle_count * 0.0002)  # 0.02% per cycle
        effective_capacity = self.capacity_wh * max(capacity_fade, 0.7)

        return {
            "charge_level_percent": self.charge_level,
            "cycle_count": self._cycle_count,
            "capacity_fade_percent": (1 - capacity_fade) * 100,
            "effective_capacity_wh": effective_capacity,
            "is_charging": self._is_charging,
        }


class Controller(ABC):
    """
    Abstract base class for device controllers.

    Implements control algorithms for physiological response and
    closed-loop feedback systems.
    """

    def __init__(
        self,
        sampling_rate_hz: float,
        control_algorithm: str,
    ):
        self.sampling_rate_hz = sampling_rate_hz
        self.control_algorithm = control_algorithm
        self._setpoints: dict[str, float] = {}
        self._gains: dict[str, float] = {}
        self._history: list[dict] = []

    def set_parameter(self, name: str, value: float) -> None:
        """Set a control parameter."""
        self._setpoints[name] = value

    def get_parameter(self, name: str) -> float | None:
        """Get a control parameter."""
        return self._setpoints.get(name)

    def set_gain(self, name: str, value: float) -> None:
        """Set a controller gain (P, I, D, etc.)."""
        self._gains[name] = value

    @abstractmethod
    def compute_output(self, measurement: float, setpoint: float) -> float:
        """
        Compute control output based on measurement and setpoint.

        Args:
            measurement: Current measured value
            setpoint: Desired target value

        Returns:
            Control output value
        """
        pass

    @abstractmethod
    def adapt_to_physiology(self, state: PhysiologicalState) -> dict[str, float]:
        """
        Adapt controller parameters based on physiological state.

        Args:
            state: Current physiological state

        Returns:
            Updated control parameters
        """
        pass


class PIDController(Controller):
    """
    PID (Proportional-Integral-Derivative) controller implementation.

    Common in implantable devices for stable closed-loop control.
    """

    def __init__(
        self,
        sampling_rate_hz: float,
        kp: float = 1.0,
        ki: float = 0.0,
        kd: float = 0.0,
        output_limits: tuple[float, float] | None = None,
    ):
        super().__init__(sampling_rate_hz, "PID")
        self._gains = {"kp": kp, "ki": ki, "kd": kd}
        self._integral = 0.0
        self._last_error = 0.0
        self._output_limits = output_limits

    def compute_output(self, measurement: float, setpoint: float) -> float:
        """Compute PID control output."""
        error = setpoint - measurement
        dt = 1.0 / self.sampling_rate_hz

        # Proportional term
        p_term = self._gains["kp"] * error

        # Integral term with anti-windup
        self._integral += error * dt
        if self._output_limits:
            self._integral = max(
                self._output_limits[0] / max(self._gains["ki"], 0.001),
                min(self._output_limits[1] / max(self._gains["ki"], 0.001), self._integral)
            )
        i_term = self._gains["ki"] * self._integral

        # Derivative term
        d_term = self._gains["kd"] * (error - self._last_error) / dt
        self._last_error = error

        # Compute output
        output = p_term + i_term + d_term

        # Apply output limits
        if self._output_limits:
            output = max(self._output_limits[0], min(self._output_limits[1], output))

        # Record history
        self._history.append({
            "error": error,
            "p_term": p_term,
            "i_term": i_term,
            "d_term": d_term,
            "output": output,
        })

        return output

    def adapt_to_physiology(self, state: PhysiologicalState) -> dict[str, float]:
        """Adapt PID gains based on activity level."""
        # Increase responsiveness during activity
        activity_factor = 1 + state.activity_level * 0.5

        adapted_gains = {
            "kp": self._gains["kp"] * activity_factor,
            "ki": self._gains["ki"],
            "kd": self._gains["kd"] * activity_factor,
        }

        return adapted_gains

    def reset(self) -> None:
        """Reset controller state."""
        self._integral = 0.0
        self._last_error = 0.0
        self._history.clear()


@dataclass
class ElectrodeConfiguration:
    """Configuration for electrode arrays."""
    num_electrodes: int
    electrode_diameter_um: float
    inter_electrode_spacing_um: float
    material: MaterialProperties
    impedance_range_ohms: tuple[float, float]


class Electrode:
    """
    Represents a single electrode or electrode array for neural interfaces.
    """

    def __init__(
        self,
        config: ElectrodeConfiguration,
        position: tuple[float, float, float] | None = None,
    ):
        self.config = config
        self.position = position or (0.0, 0.0, 0.0)
        self._impedances: list[float] = [
            (config.impedance_range_ohms[0] + config.impedance_range_ohms[1]) / 2
            for _ in range(config.num_electrodes)
        ]
        self._is_active: list[bool] = [True] * config.num_electrodes

    @property
    def active_electrode_count(self) -> int:
        """Number of currently active electrodes."""
        return sum(self._is_active)

    def measure_impedance(self, electrode_index: int) -> float:
        """Measure impedance of a specific electrode."""
        if 0 <= electrode_index < self.config.num_electrodes:
            return self._impedances[electrode_index]
        raise IndexError(f"Electrode index {electrode_index} out of range")

    def measure_all_impedances(self) -> list[float]:
        """Measure impedance of all electrodes."""
        return self._impedances.copy()

    def set_electrode_active(self, electrode_index: int, active: bool) -> None:
        """Enable or disable a specific electrode."""
        if 0 <= electrode_index < self.config.num_electrodes:
            self._is_active[electrode_index] = active
        else:
            raise IndexError(f"Electrode index {electrode_index} out of range")

    def get_charge_density(self, current_ua: float, pulse_width_us: float) -> float:
        """
        Calculate charge density for safety assessment.

        Args:
            current_ua: Stimulation current in microamps
            pulse_width_us: Pulse width in microseconds

        Returns:
            Charge density in uC/cm^2
        """
        # Calculate electrode area in cm^2
        radius_cm = (self.config.electrode_diameter_um / 2) * 1e-4
        area_cm2 = math.pi * radius_cm ** 2

        # Calculate charge in uC
        charge_uc = current_ua * pulse_width_us * 1e-6

        return charge_uc / area_cm2

    def is_charge_density_safe(self, current_ua: float, pulse_width_us: float) -> bool:
        """
        Check if charge density is within safe limits.

        Shannon limit: k = log(Q/A) + log(Q) where k < 1.85 is generally safe
        """
        charge_density = self.get_charge_density(current_ua, pulse_width_us)
        # Simplified safety check: < 30 uC/cm^2 is generally considered safe
        return charge_density < 30.0

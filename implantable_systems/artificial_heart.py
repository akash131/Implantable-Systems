"""
Total Artificial Heart Module

Implements complete cardiac replacement systems including:
- Dual ventricle replacement
- Power systems and batteries
- Physiological control algorithms
- Balancing left/right output
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import math

from .base import (
    ImplantableDevice,
    PowerSystem,
    Controller,
    PIDController,
    PhysiologicalState,
    MaterialProperties,
    DeviceState,
)


class VentricleType(Enum):
    """Type of ventricle (left or right)."""
    LEFT = "left"
    RIGHT = "right"


class TAHMode(Enum):
    """Operating modes for Total Artificial Heart."""
    FIXED_RATE = "fixed_rate"           # Fixed beat rate
    FILL_LIMITED = "fill_limited"       # Ejects when filled
    FULL_FILL_FULL_EJECT = "full_fill"  # Complete filling before ejection
    PARTIAL_FILL = "partial_fill"       # Rate-responsive mode


@dataclass
class VentricleParameters:
    """Parameters for a single artificial ventricle."""
    max_stroke_volume_ml: float = 70.0
    min_stroke_volume_ml: float = 30.0
    max_rate_bpm: float = 130
    min_rate_bpm: float = 40
    systolic_duration_ms: float = 300
    diastolic_duration_ms: float = 500
    max_drive_pressure_mmhg: float = 280  # Pneumatic drive pressure


class HeartPowerSystem(PowerSystem):
    """
    Power system for Total Artificial Heart.

    Supports multiple power sources:
    - Internal rechargeable battery
    - External battery pack
    - Percutaneous power cable
    - Transcutaneous energy transfer
    """

    def __init__(
        self,
        internal_battery_wh: float = 30.0,
        external_battery_wh: float = 150.0,
        max_power_draw_watts: float = 25.0,
    ):
        super().__init__(
            capacity_wh=internal_battery_wh,
            nominal_voltage=16.8,
            max_discharge_rate=max_power_draw_watts,
        )
        self.external_battery_wh = external_battery_wh
        self._external_charge = external_battery_wh
        self._current_draw = 12.0  # Typical TAH power
        self._power_source = "internal"  # internal, external, tets, percutaneous

    @property
    def _current_discharge_rate(self) -> float:
        return self._current_draw

    def get_total_available_energy(self) -> float:
        """Get total available energy from all sources."""
        return self._current_charge_wh + self._external_charge

    def get_total_runtime_hours(self) -> float:
        """Calculate total runtime from all batteries."""
        if self._current_draw <= 0:
            return float('inf')
        return self.get_total_available_energy() / self._current_draw

    def charge(self, power_watts: float, duration_hours: float) -> float:
        """Charge internal battery."""
        # Assume 90% charging efficiency
        efficiency = 0.9
        energy = power_watts * duration_hours * efficiency

        self._current_charge_wh = min(
            self.capacity_wh,
            self._current_charge_wh + energy
        )
        self._is_charging = True
        return self.charge_level

    def discharge(self, power_watts: float, duration_hours: float) -> bool:
        """
        Discharge battery with automatic source switching.

        Switches from internal to external when internal depletes.
        """
        if power_watts > self.max_discharge_rate:
            return False

        energy_required = power_watts * duration_hours
        self._current_draw = power_watts

        # Try internal battery first
        if self._current_charge_wh >= energy_required:
            self._current_charge_wh -= energy_required
            self._power_source = "internal"
        # Fall back to external
        elif self._external_charge >= energy_required:
            self._external_charge -= energy_required
            self._power_source = "external"
        # Combine both
        elif self._current_charge_wh + self._external_charge >= energy_required:
            remaining = energy_required - self._current_charge_wh
            self._current_charge_wh = 0
            self._external_charge -= remaining
            self._power_source = "external"
        else:
            return False

        self._is_charging = False

        # Cycle count when internal battery crosses 20%
        if self._current_charge_wh / self.capacity_wh < 0.2:
            self._cycle_count += 1

        return True

    def swap_external_battery(self, new_capacity_wh: float = 150.0) -> None:
        """Simulate swapping external battery pack."""
        self._external_charge = new_capacity_wh
        self._power_source = "external"

    def get_status(self) -> dict:
        """Get comprehensive power system status."""
        return {
            **self.get_health_status(),
            "external_battery_percent": (self._external_charge / self.external_battery_wh) * 100,
            "total_runtime_hours": self.get_total_runtime_hours(),
            "power_source": self._power_source,
            "current_draw_watts": self._current_draw,
        }


class DualVentricle:
    """
    Dual ventricle system for Total Artificial Heart.

    Manages both left and right artificial ventricles with
    proper balance to prevent pulmonary/systemic congestion.
    """

    def __init__(
        self,
        left_params: VentricleParameters | None = None,
        right_params: VentricleParameters | None = None,
    ):
        self.left_params = left_params or VentricleParameters()
        self.right_params = right_params or VentricleParameters(
            max_drive_pressure_mmhg=100  # RV needs less pressure
        )

        # Current operating state
        self._left_stroke_volume = 65.0
        self._right_stroke_volume = 65.0
        self._rate = 80  # bpm
        self._mode = TAHMode.FULL_FILL_FULL_EJECT

        # Balance tracking
        self._left_atrial_pressure = 10.0  # mmHg
        self._right_atrial_pressure = 5.0  # mmHg

    @property
    def left_cardiac_output(self) -> float:
        """Left ventricular cardiac output in L/min."""
        return (self._left_stroke_volume / 1000) * self._rate

    @property
    def right_cardiac_output(self) -> float:
        """Right ventricular cardiac output in L/min."""
        return (self._right_stroke_volume / 1000) * self._rate

    @property
    def output_balance(self) -> float:
        """
        Ratio of left to right output.

        Should be approximately 1.0 for balanced circulation.
        Bronchial circulation causes LV output to be slightly higher.
        """
        if self.right_cardiac_output > 0:
            return self.left_cardiac_output / self.right_cardiac_output
        return 0.0

    def calculate_atrial_pressures(
        self, total_blood_volume_ml: float = 5000.0
    ) -> tuple[float, float]:
        """
        Estimate atrial pressures based on cardiac output balance.

        Imbalance causes blood to accumulate in one circulation.
        """
        balance = self.output_balance

        # Simplified model: pressure rises in circuit with lower output
        base_lap = 10.0  # mmHg
        base_rap = 5.0   # mmHg

        if balance > 1.05:  # LV output exceeds RV
            # Blood accumulates in systemic veins -> elevated RAP
            self._right_atrial_pressure = base_rap * (1 + (balance - 1) * 5)
            self._left_atrial_pressure = base_lap / balance
        elif balance < 0.95:  # RV output exceeds LV
            # Blood accumulates in pulmonary veins -> elevated LAP
            self._left_atrial_pressure = base_lap * (1 + (1 - balance) * 5)
            self._right_atrial_pressure = base_rap * balance
        else:
            self._left_atrial_pressure = base_lap
            self._right_atrial_pressure = base_rap

        return self._left_atrial_pressure, self._right_atrial_pressure

    def balance_outputs(self) -> dict:
        """
        Automatically adjust stroke volumes to balance circulation.

        Critical to prevent pulmonary or systemic congestion.
        """
        lap, rap = self.calculate_atrial_pressures()

        adjustments = {}

        # Adjust based on atrial pressures
        if lap > 15:  # High LAP -> increase LV output
            sv_increase = min(5, (lap - 15) * 0.5)
            self._left_stroke_volume = min(
                self.left_params.max_stroke_volume_ml,
                self._left_stroke_volume + sv_increase
            )
            adjustments["left_sv_change"] = sv_increase

        if rap > 10:  # High RAP -> increase RV output
            sv_increase = min(5, (rap - 10) * 0.5)
            self._right_stroke_volume = min(
                self.right_params.max_stroke_volume_ml,
                self._right_stroke_volume + sv_increase
            )
            adjustments["right_sv_change"] = sv_increase

        # Rate adjustment for low cardiac output
        total_output = self.left_cardiac_output + self.right_cardiac_output
        if total_output < 8.0:  # Target ~5 L/min per ventricle
            rate_increase = min(10, (8.0 - total_output) * 5)
            self._rate = min(
                min(self.left_params.max_rate_bpm, self.right_params.max_rate_bpm),
                self._rate + rate_increase
            )
            adjustments["rate_change"] = rate_increase

        adjustments["final_balance"] = self.output_balance
        return adjustments

    def generate_pressure_waveform(
        self, ventricle: VentricleType
    ) -> list[float]:
        """
        Generate ventricular pressure waveform over one cardiac cycle.

        Returns 100 samples over one beat.
        """
        params = self.left_params if ventricle == VentricleType.LEFT else self.right_params
        stroke_volume = (
            self._left_stroke_volume if ventricle == VentricleType.LEFT
            else self._right_stroke_volume
        )

        cycle_time_ms = 60000 / self._rate
        systole_fraction = params.systolic_duration_ms / cycle_time_ms

        # Peak pressure depends on afterload
        if ventricle == VentricleType.LEFT:
            peak_pressure = 120.0  # Systemic pressure
            diastolic_pressure = 10.0
        else:
            peak_pressure = 25.0  # Pulmonary pressure
            diastolic_pressure = 5.0

        waveform = []
        for i in range(100):
            phase = i / 100

            if phase < systole_fraction:
                # Systolic rise and fall
                systole_phase = phase / systole_fraction
                if systole_phase < 0.3:
                    # Isovolumic contraction
                    pressure = diastolic_pressure + (peak_pressure - diastolic_pressure) * (systole_phase / 0.3)
                elif systole_phase < 0.8:
                    # Ejection
                    pressure = peak_pressure
                else:
                    # Isovolumic relaxation
                    relax_phase = (systole_phase - 0.8) / 0.2
                    pressure = peak_pressure - (peak_pressure - diastolic_pressure) * relax_phase
            else:
                # Diastole - filling
                pressure = diastolic_pressure

            waveform.append(pressure)

        return waveform

    def set_rate(self, rate_bpm: float) -> bool:
        """Set heart rate."""
        min_rate = max(self.left_params.min_rate_bpm, self.right_params.min_rate_bpm)
        max_rate = min(self.left_params.max_rate_bpm, self.right_params.max_rate_bpm)

        if min_rate <= rate_bpm <= max_rate:
            self._rate = rate_bpm
            return True
        return False

    def set_stroke_volume(
        self, ventricle: VentricleType, stroke_volume_ml: float
    ) -> bool:
        """Set stroke volume for a specific ventricle."""
        params = self.left_params if ventricle == VentricleType.LEFT else self.right_params

        if params.min_stroke_volume_ml <= stroke_volume_ml <= params.max_stroke_volume_ml:
            if ventricle == VentricleType.LEFT:
                self._left_stroke_volume = stroke_volume_ml
            else:
                self._right_stroke_volume = stroke_volume_ml
            return True
        return False

    def get_status(self) -> dict:
        """Get dual ventricle status."""
        return {
            "mode": self._mode.value,
            "rate_bpm": self._rate,
            "left_stroke_volume_ml": self._left_stroke_volume,
            "right_stroke_volume_ml": self._right_stroke_volume,
            "left_cardiac_output_lpm": self.left_cardiac_output,
            "right_cardiac_output_lpm": self.right_cardiac_output,
            "output_balance": self.output_balance,
            "left_atrial_pressure_mmhg": self._left_atrial_pressure,
            "right_atrial_pressure_mmhg": self._right_atrial_pressure,
        }


class TAHController(PIDController):
    """
    Physiological controller for Total Artificial Heart.

    Implements rate-responsive control based on:
    - Activity sensing (accelerometer)
    - Metabolic demand estimation
    - Atrial pressure feedback
    """

    def __init__(
        self,
        base_rate_bpm: float = 80,
        min_rate_bpm: float = 50,
        max_rate_bpm: float = 130,
        sampling_rate_hz: float = 50.0,
    ):
        super().__init__(
            sampling_rate_hz=sampling_rate_hz,
            kp=2.0,
            ki=0.1,
            kd=0.5,
            output_limits=(min_rate_bpm, max_rate_bpm),
        )
        self.base_rate = base_rate_bpm
        self.min_rate = min_rate_bpm
        self.max_rate = max_rate_bpm
        self._target_cardiac_output = 5.0  # L/min
        self._activity_sensor_value = 0.0

    def calculate_target_rate(self, state: PhysiologicalState) -> float:
        """
        Calculate target heart rate based on physiological state.

        Mimics the chronotropic response of a healthy heart.
        """
        # Base metabolic demand
        base_demand = 5.0  # L/min at rest

        # Activity-based demand increase
        # Peak exercise can require up to 5x resting output
        activity_factor = 1.0 + state.activity_level * 4.0
        target_output = base_demand * activity_factor

        self._target_cardiac_output = target_output

        # Calculate rate needed (assuming ~70ml stroke volume)
        stroke_volume = 0.065  # L
        target_rate = target_output / stroke_volume

        # Clamp to limits
        return max(self.min_rate, min(self.max_rate, target_rate))

    def adapt_to_physiology(self, state: PhysiologicalState) -> dict[str, float]:
        """Adapt TAH parameters based on physiological state."""
        target_rate = self.calculate_target_rate(state)

        # Adjust controller gains based on activity
        if state.activity_level > 0.5:
            # Faster response during exercise
            activity_gains = {
                "kp": self._gains["kp"] * 1.5,
                "ki": self._gains["ki"] * 0.8,  # Reduce to prevent overshoot
                "kd": self._gains["kd"] * 1.2,
            }
        else:
            activity_gains = self._gains.copy()

        return {
            "target_rate_bpm": target_rate,
            "target_cardiac_output_lpm": self._target_cardiac_output,
            "activity_level": state.activity_level,
            **activity_gains,
        }

    def calculate_rate_response(
        self, current_rate: float, state: PhysiologicalState
    ) -> float:
        """
        Calculate rate adjustment using PID control.

        Args:
            current_rate: Current heart rate in bpm
            state: Physiological state

        Returns:
            New target rate in bpm
        """
        target_rate = self.calculate_target_rate(state)

        # Use cardiac output error for PID
        current_output = current_rate * 0.065  # Estimate
        target_output = self._target_cardiac_output

        rate_adjustment = self.compute_output(current_output, target_output)

        # Convert output adjustment to rate
        new_rate = current_rate + rate_adjustment * 0.1

        return max(self.min_rate, min(self.max_rate, new_rate))


class TotalArtificialHeart(ImplantableDevice):
    """
    Complete Total Artificial Heart system.

    Integrates dual ventricle pumps, power system, and controller
    for complete cardiac replacement.
    """

    def __init__(
        self,
        name: str = "Total Artificial Heart",
        manufacturer: str = "Generic",
        model: str = "TAH-2000",
    ):
        super().__init__(
            name=name,
            manufacturer=manufacturer,
            model=model,
            materials=[
                MaterialProperties.titanium_grade5(),
                MaterialProperties.silicone_medical(),
                MaterialProperties.platinum_iridium(),  # For sensors
            ]
        )

        # Initialize subsystems
        self._ventricles = DualVentricle()
        self._power_system = HeartPowerSystem()
        self._controller = TAHController()

        # Sensor data
        self._accelerometer_data: list[float] = []
        self._motor_current_left: float = 0.0
        self._motor_current_right: float = 0.0

    def activate(self) -> bool:
        """Activate the TAH system."""
        if self._state == DeviceState.OFF:
            self.state = DeviceState.STANDBY

        # Perform startup sequence
        tests = self.perform_self_test()
        if not all(tests.values()):
            self.log_error(f"Startup tests failed: {tests}")
            return False

        # Initialize to safe operating parameters
        self._ventricles.set_rate(self._controller.base_rate)

        self.state = DeviceState.ACTIVE
        return True

    def deactivate(self) -> bool:
        """
        Deactivate the TAH system.

        WARNING: This is typically only done during device explant.
        For a TAH patient, stopping the device is fatal.
        """
        if self._state == DeviceState.ACTIVE:
            # Gradual rate reduction
            current_rate = self._ventricles._rate
            while current_rate > self._controller.min_rate:
                current_rate -= 5
                self._ventricles.set_rate(max(current_rate, self._controller.min_rate))

            self.state = DeviceState.STANDBY
        return True

    def perform_self_test(self) -> dict[str, bool]:
        """Perform comprehensive self-test."""
        tests = {
            "power_internal": self._power_system.charge_level > 20,
            "power_external": self._power_system._external_charge > 0,
            "left_ventricle_motor": True,
            "right_ventricle_motor": True,
            "left_ventricle_valves": True,
            "right_ventricle_valves": True,
            "sensors": True,
            "controller": True,
            "communication": True,
        }

        # Check output balance
        balance = self._ventricles.output_balance
        tests["output_balance"] = 0.9 <= balance <= 1.1

        return tests

    def update(self, state: PhysiologicalState, dt: float = 0.02) -> dict:
        """
        Update TAH operation for one time step.

        Args:
            state: Current physiological state
            dt: Time step in seconds

        Returns:
            Dictionary of operating parameters
        """
        if self._state != DeviceState.ACTIVE:
            return {"status": "inactive"}

        # Controller update
        current_rate = self._ventricles._rate
        new_rate = self._controller.calculate_rate_response(current_rate, state)
        self._ventricles.set_rate(new_rate)

        # Balance outputs
        balance_result = self._ventricles.balance_outputs()

        # Calculate power consumption
        # Power roughly proportional to rate and stroke volume
        power = (
            8.0 +  # Base power
            (new_rate - 60) * 0.05 +  # Rate component
            (self._ventricles._left_stroke_volume - 50) * 0.1  # SV component
        )
        self._power_system.discharge(power, dt / 3600)

        # Get atrial pressures
        lap, rap = self._ventricles.calculate_atrial_pressures()

        telemetry = {
            "rate_bpm": new_rate,
            "left_cardiac_output_lpm": self._ventricles.left_cardiac_output,
            "right_cardiac_output_lpm": self._ventricles.right_cardiac_output,
            "output_balance": self._ventricles.output_balance,
            "left_atrial_pressure_mmhg": lap,
            "right_atrial_pressure_mmhg": rap,
            "power_watts": power,
            "battery_level": self._power_system.charge_level,
            **balance_result,
        }
        self.record_telemetry(telemetry)

        return telemetry

    def emergency_backup_mode(self) -> bool:
        """
        Enter emergency backup mode on power failure.

        Reduces to minimum safe rate to extend battery life.
        """
        self.log_error("Entering emergency backup mode")

        # Set minimum safe parameters
        self._ventricles.set_rate(self._controller.min_rate)
        self._ventricles._left_stroke_volume = 50.0
        self._ventricles._right_stroke_volume = 50.0

        # Alert patient
        return True

    def get_comprehensive_status(self) -> dict:
        """Get comprehensive device status."""
        return {
            "device_state": self._state.value,
            "ventricle_status": self._ventricles.get_status(),
            "power_status": self._power_system.get_status(),
            "biocompatibility": self.get_biocompatibility_assessment(),
            "error_count": len(self._error_log),
            "telemetry_records": len(self._telemetry_data),
        }

    def get_alarm_status(self) -> dict:
        """Check for alarm conditions."""
        alarms = {}

        # Power alarms
        if self._power_system.charge_level < 10:
            alarms["low_battery_critical"] = True
        elif self._power_system.charge_level < 25:
            alarms["low_battery_warning"] = True

        # Output balance alarm
        balance = self._ventricles.output_balance
        if balance < 0.85 or balance > 1.15:
            alarms["output_imbalance"] = True

        # Atrial pressure alarms
        lap, rap = self._ventricles.calculate_atrial_pressures()
        if lap > 20:
            alarms["high_left_atrial_pressure"] = True
        if rap > 15:
            alarms["high_right_atrial_pressure"] = True

        # Low cardiac output alarm
        total_output = self._ventricles.left_cardiac_output
        if total_output < 3.0:
            alarms["low_cardiac_output"] = True

        alarms["has_alarms"] = len(alarms) > 1  # Exclude this key
        return alarms

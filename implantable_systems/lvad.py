"""
Ventricular Assist Device (LVAD) Module

Implements Left Ventricular Assist Device systems including:
- Continuous flow pumps (axial and centrifugal)
- Pulsatile flow pumps
- Transcutaneous energy transfer systems
- Physiological control algorithms
- Hemolysis and biocompatibility considerations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
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


class FlowType(Enum):
    """Type of blood flow produced by the pump."""
    CONTINUOUS = "continuous"
    PULSATILE = "pulsatile"
    HYBRID = "hybrid"  # Continuous with artificial pulsatility


class PumpType(Enum):
    """Mechanical pump design."""
    AXIAL = "axial"           # HeartMate II style
    CENTRIFUGAL = "centrifugal"  # HeartWare/HeartMate 3 style
    DISPLACEMENT = "displacement"  # Pulsatile pumps


@dataclass
class HemolysisParameters:
    """Parameters for hemolysis (red blood cell damage) modeling."""
    shear_stress_threshold: float = 150.0  # Pa - threshold for hemolysis
    exposure_time_constant: float = 0.0001  # seconds
    power_law_exponent: float = 2.416  # Giersiepen model
    time_exponent: float = 0.785

    def calculate_hemolysis_index(
        self, shear_stress: float, exposure_time: float
    ) -> float:
        """
        Calculate normalized hemolysis index using power law model.

        Based on Giersiepen correlation: HI = C * tau^alpha * t^beta

        Args:
            shear_stress: Shear stress in Pa
            exposure_time: Exposure time in seconds

        Returns:
            Hemolysis index (dimensionless)
        """
        if shear_stress < self.shear_stress_threshold:
            return 0.0

        C = 3.62e-7  # Empirical constant
        hi = C * (shear_stress ** self.power_law_exponent) * (
            exposure_time ** self.time_exponent
        )
        return min(hi, 1.0)  # Cap at 100% hemolysis


@dataclass
class BiocompatibleCoating:
    """Surface coating for blood-contacting components."""
    name: str
    thickness_um: float
    hemocompatibility_score: float  # 0-1, higher is better
    durability_years: float
    thrombogenicity: float  # 0-1, lower is better (anti-thrombogenic)

    @classmethod
    def diamond_like_carbon(cls) -> "BiocompatibleCoating":
        """DLC coating for improved hemocompatibility."""
        return cls(
            name="Diamond-Like Carbon (DLC)",
            thickness_um=1.0,
            hemocompatibility_score=0.9,
            durability_years=10.0,
            thrombogenicity=0.1
        )

    @classmethod
    def titanium_nitride(cls) -> "BiocompatibleCoating":
        """TiN coating for wear resistance and biocompatibility."""
        return cls(
            name="Titanium Nitride (TiN)",
            thickness_um=3.0,
            hemocompatibility_score=0.85,
            durability_years=15.0,
            thrombogenicity=0.15
        )

    @classmethod
    def textured_titanium(cls) -> "BiocompatibleCoating":
        """Textured surface for neointima formation."""
        return cls(
            name="Textured Sintered Titanium",
            thickness_um=50.0,
            hemocompatibility_score=0.95,
            durability_years=20.0,
            thrombogenicity=0.05
        )


class TranscutaneousEnergyTransfer(PowerSystem):
    """
    Transcutaneous Energy Transfer System (TETS).

    Wireless power transfer through intact skin using inductive coupling.
    Eliminates percutaneous driveline infection risks.
    """

    def __init__(
        self,
        internal_battery_wh: float = 20.0,
        external_battery_wh: float = 100.0,
        transfer_frequency_khz: float = 200.0,
        coil_diameter_mm: float = 60.0,
        max_power_watts: float = 15.0,
    ):
        super().__init__(
            capacity_wh=internal_battery_wh,
            nominal_voltage=14.4,
            max_discharge_rate=max_power_watts,
        )
        self.external_battery_wh = external_battery_wh
        self.transfer_frequency_khz = transfer_frequency_khz
        self.coil_diameter_mm = coil_diameter_mm
        self.max_power_watts = max_power_watts
        self._external_charge = external_battery_wh
        self._coil_alignment = 1.0  # 0-1, perfect alignment = 1
        self._skin_thickness_mm = 10.0
        self._transfer_efficiency = 0.0
        self._current_power_draw = 8.0  # Typical LVAD power draw

    @property
    def _current_discharge_rate(self) -> float:
        return self._current_power_draw

    def calculate_transfer_efficiency(self) -> float:
        """
        Calculate power transfer efficiency based on coil alignment and skin thickness.

        Returns:
            Efficiency as decimal (0-1)
        """
        # Coupling coefficient decreases with misalignment and distance
        # Simplified model: k = k0 * exp(-d/d0) * alignment
        k0 = 0.6  # Maximum coupling coefficient
        d0 = 15.0  # Characteristic distance in mm

        coupling = k0 * math.exp(-self._skin_thickness_mm / d0) * self._coil_alignment

        # Efficiency depends on coupling and quality factor
        Q = 100  # Quality factor of resonant circuit
        efficiency = (coupling ** 2 * Q ** 2) / (
            (1 + coupling ** 2 * Q ** 2) * (1 + 1 / Q)
        )

        self._transfer_efficiency = min(efficiency, 0.95)  # Cap at 95%
        return self._transfer_efficiency

    def set_coil_alignment(self, alignment: float) -> None:
        """Set coil alignment factor (0-1)."""
        self._coil_alignment = max(0.0, min(1.0, alignment))
        self.calculate_transfer_efficiency()

    def charge(self, power_watts: float, duration_hours: float) -> float:
        """Charge internal battery via transcutaneous transfer."""
        efficiency = self.calculate_transfer_efficiency()
        effective_power = power_watts * efficiency

        # Check external battery
        energy_needed = effective_power * duration_hours
        if self._external_charge < energy_needed / efficiency:
            # Limited by external battery
            energy_available = self._external_charge * efficiency
            duration_hours = energy_available / effective_power

        # Charge internal battery
        energy_delivered = effective_power * duration_hours
        self._current_charge_wh = min(
            self.capacity_wh,
            self._current_charge_wh + energy_delivered
        )

        # Deplete external battery
        self._external_charge -= energy_delivered / efficiency

        self._is_charging = True
        return self.charge_level

    def discharge(self, power_watts: float, duration_hours: float) -> bool:
        """Discharge internal battery."""
        if power_watts > self.max_discharge_rate:
            return False

        energy_required = power_watts * duration_hours
        if energy_required > self._current_charge_wh:
            return False

        self._current_charge_wh -= energy_required
        self._current_power_draw = power_watts
        self._is_charging = False

        # Increment cycle count when crossing 20% threshold
        if self._current_charge_wh / self.capacity_wh < 0.2:
            self._cycle_count += 1

        return True

    def get_status(self) -> dict:
        """Get comprehensive TETS status."""
        return {
            **self.get_health_status(),
            "external_battery_percent": (self._external_charge / self.external_battery_wh) * 100,
            "transfer_efficiency_percent": self._transfer_efficiency * 100,
            "coil_alignment": self._coil_alignment,
            "skin_thickness_mm": self._skin_thickness_mm,
            "power_draw_watts": self._current_power_draw,
        }


class PhysiologicalController(PIDController):
    """
    Advanced controller for LVAD physiological response.

    Implements speed modulation based on preload, afterload, and
    patient activity level.
    """

    def __init__(
        self,
        target_flow_lpm: float = 5.0,
        min_speed_rpm: float = 2000,
        max_speed_rpm: float = 9000,
        sampling_rate_hz: float = 100.0,
    ):
        super().__init__(
            sampling_rate_hz=sampling_rate_hz,
            kp=100.0,
            ki=10.0,
            kd=5.0,
            output_limits=(min_speed_rpm, max_speed_rpm),
        )
        self.target_flow_lpm = target_flow_lpm
        self.min_speed_rpm = min_speed_rpm
        self.max_speed_rpm = max_speed_rpm
        self._current_speed = (min_speed_rpm + max_speed_rpm) / 2
        self._suction_event_count = 0

    def estimate_flow_from_power(
        self, power_watts: float, speed_rpm: float, pressure_head_mmhg: float
    ) -> float:
        """
        Estimate blood flow using pump power consumption (sensorless method).

        Based on pump affinity laws and hydraulic efficiency.
        """
        # Simplified model: Q = a*omega - b*deltaP + c*P/omega
        # where Q is flow, omega is speed, deltaP is pressure, P is power
        a = 0.001  # Flow coefficient
        b = 0.02   # Pressure coefficient
        c = 50.0   # Power coefficient

        flow = a * speed_rpm - b * pressure_head_mmhg + c * power_watts / speed_rpm
        return max(0.0, flow)

    def detect_suction_event(self, flow_waveform: list[float]) -> bool:
        """
        Detect ventricular suction events from flow waveform.

        Suction occurs when the ventricle is over-emptied, indicated by
        characteristic flow oscillations.
        """
        if len(flow_waveform) < 10:
            return False

        # Calculate flow variability
        mean_flow = sum(flow_waveform) / len(flow_waveform)
        variance = sum((f - mean_flow) ** 2 for f in flow_waveform) / len(flow_waveform)
        cv = (variance ** 0.5) / mean_flow if mean_flow > 0 else 0

        # High coefficient of variation indicates suction
        is_suction = cv > 0.3 or min(flow_waveform) < 0.5
        if is_suction:
            self._suction_event_count += 1
        return is_suction

    def adapt_to_physiology(self, state: PhysiologicalState) -> dict[str, float]:
        """
        Adapt pump speed based on physiological state.

        Implements rate-responsive control similar to pacemakers.
        """
        # Base flow requirement
        base_flow = 4.5  # L/min at rest

        # Activity-based flow increase (up to 2x at peak exercise)
        activity_multiplier = 1.0 + state.activity_level * 1.0
        target_flow = base_flow * activity_multiplier

        # Afterload consideration (higher MAP = higher pressure head)
        pressure_factor = state.mean_arterial_pressure / 93.0  # Normalized to normal MAP

        # Adjust target based on measured cardiac output
        if state.cardiac_output < target_flow * 0.8:
            target_flow = min(target_flow * 1.1, 8.0)  # Increase slightly

        self.target_flow_lpm = target_flow
        self._setpoints["target_flow"] = target_flow

        return {
            "target_flow_lpm": target_flow,
            "activity_multiplier": activity_multiplier,
            "pressure_factor": pressure_factor,
        }

    def calculate_speed_command(
        self, current_flow: float, state: PhysiologicalState
    ) -> float:
        """Calculate optimal pump speed command."""
        # Update targets based on physiology
        self.adapt_to_physiology(state)

        # PID control for flow regulation
        speed_adjustment = self.compute_output(current_flow, self.target_flow_lpm)

        # Apply rate limiting for smooth transitions
        max_rate = 100  # RPM per control cycle
        speed_change = speed_adjustment - self._current_speed
        speed_change = max(-max_rate, min(max_rate, speed_change))

        self._current_speed += speed_change
        return self._current_speed


class ContinuousFlowPump:
    """
    Continuous flow blood pump (axial or centrifugal).

    Models pump hydraulics, motor characteristics, and hemolysis.
    """

    def __init__(
        self,
        pump_type: PumpType = PumpType.CENTRIFUGAL,
        max_speed_rpm: float = 9000,
        max_flow_lpm: float = 10.0,
        rotor_diameter_mm: float = 50.0,
        coating: BiocompatibleCoating | None = None,
    ):
        self.pump_type = pump_type
        self.max_speed_rpm = max_speed_rpm
        self.max_flow_lpm = max_flow_lpm
        self.rotor_diameter_mm = rotor_diameter_mm
        self.coating = coating or BiocompatibleCoating.textured_titanium()
        self._current_speed = 0.0
        self._hemolysis_params = HemolysisParameters()
        self._bearing_type = "magnetic" if pump_type == PumpType.CENTRIFUGAL else "hydrodynamic"

    def calculate_head_flow_curve(self, speed_rpm: float) -> Callable[[float], float]:
        """
        Generate H-Q curve for current speed.

        Returns function: pressure_head = f(flow_rate)
        """
        # Normalized curve parameters (typical for centrifugal pump)
        H0 = 1.0  # Shut-off head coefficient
        slope = -0.1  # H-Q curve slope

        # Scale with speed squared (affinity laws)
        speed_ratio = speed_rpm / self.max_speed_rpm
        max_head = 150 * speed_ratio ** 2  # mmHg at zero flow

        def head_curve(flow_lpm: float) -> float:
            normalized_flow = flow_lpm / self.max_flow_lpm
            normalized_head = H0 + slope * normalized_flow ** 2
            return max_head * normalized_head

        return head_curve

    def calculate_operating_point(
        self, speed_rpm: float, afterload_mmhg: float
    ) -> tuple[float, float]:
        """
        Calculate pump operating point (flow rate) for given speed and afterload.

        Args:
            speed_rpm: Pump rotational speed
            afterload_mmhg: Systemic afterload pressure

        Returns:
            Tuple of (flow_lpm, power_watts)
        """
        head_curve = self.calculate_head_flow_curve(speed_rpm)

        # Find flow where pump head equals afterload
        # Simple bisection search
        flow_low, flow_high = 0.0, self.max_flow_lpm
        for _ in range(20):
            flow_mid = (flow_low + flow_high) / 2
            head = head_curve(flow_mid)
            if head > afterload_mmhg:
                flow_low = flow_mid
            else:
                flow_high = flow_mid

        flow = (flow_low + flow_high) / 2

        # Calculate hydraulic power: P = Q * deltaP / eta
        efficiency = self._calculate_efficiency(flow, speed_rpm)
        # Convert: L/min * mmHg -> Watts (1 L/min * mmHg ≈ 0.00222 W)
        hydraulic_power = flow * afterload_mmhg * 0.00222
        electrical_power = hydraulic_power / max(efficiency, 0.1)

        return flow, electrical_power

    def _calculate_efficiency(self, flow_lpm: float, speed_rpm: float) -> float:
        """Calculate pump hydraulic efficiency."""
        # Efficiency curve peaks around 70-80% at design point
        design_flow = self.max_flow_lpm * 0.6
        design_speed = self.max_speed_rpm * 0.7

        flow_factor = 1 - abs(flow_lpm - design_flow) / design_flow * 0.3
        speed_factor = 1 - abs(speed_rpm - design_speed) / design_speed * 0.2

        return 0.7 * flow_factor * speed_factor

    def estimate_hemolysis(self, flow_lpm: float, speed_rpm: float) -> float:
        """
        Estimate hemolysis index based on operating conditions.

        Args:
            flow_lpm: Current flow rate
            speed_rpm: Current pump speed

        Returns:
            Hemolysis index (fraction of RBCs damaged per pass)
        """
        # Estimate shear stress based on rotor tip speed and gap
        tip_speed = math.pi * self.rotor_diameter_mm * 1e-3 * speed_rpm / 60  # m/s
        gap = 0.0005  # 500 um typical gap

        # Shear stress estimate: tau = mu * du/dr
        blood_viscosity = 0.0035  # Pa·s
        shear_stress = blood_viscosity * tip_speed / gap

        # Exposure time estimate based on pump volume and flow
        pump_volume = 0.05  # liters (50 mL typical)
        if flow_lpm > 0:
            residence_time = pump_volume / flow_lpm * 60  # seconds
        else:
            residence_time = 1.0

        return self._hemolysis_params.calculate_hemolysis_index(shear_stress, residence_time)

    def set_speed(self, speed_rpm: float) -> bool:
        """Set pump speed with safety checks."""
        if 0 <= speed_rpm <= self.max_speed_rpm:
            self._current_speed = speed_rpm
            return True
        return False

    def get_status(self) -> dict:
        """Get pump status."""
        return {
            "pump_type": self.pump_type.value,
            "current_speed_rpm": self._current_speed,
            "bearing_type": self._bearing_type,
            "coating": self.coating.name,
            "coating_thrombogenicity": self.coating.thrombogenicity,
        }


class PulsatileFlowPump:
    """
    Pulsatile flow blood pump using displacement mechanism.

    Mimics natural cardiac pulsatility for improved end-organ perfusion.
    """

    def __init__(
        self,
        stroke_volume_ml: float = 70.0,
        max_rate_bpm: float = 120,
        systolic_duration_fraction: float = 0.35,
    ):
        self.stroke_volume_ml = stroke_volume_ml
        self.max_rate_bpm = max_rate_bpm
        self.systolic_duration_fraction = systolic_duration_fraction
        self._current_rate = 70  # bpm
        self._valve_state = "closed"
        self._hemolysis_params = HemolysisParameters()

    def calculate_flow_waveform(self, rate_bpm: float) -> list[float]:
        """
        Generate pulsatile flow waveform.

        Returns flow values over one cardiac cycle (100 samples).
        """
        cycle_time_s = 60 / rate_bpm
        systole_time = cycle_time_s * self.systolic_duration_fraction

        waveform = []
        for i in range(100):
            t = i / 100 * cycle_time_s

            if t < systole_time:
                # Systolic ejection - sinusoidal approximation
                phase = math.pi * t / systole_time
                # Convert stroke volume to instantaneous flow in L/min
                peak_flow = self.stroke_volume_ml / 1000 / systole_time * 60
                flow = peak_flow * math.sin(phase)
            else:
                # Diastolic - minimal flow
                flow = 0.1  # Small regurgitant flow

            waveform.append(flow)

        return waveform

    def calculate_cardiac_output(self, rate_bpm: float) -> float:
        """Calculate cardiac output in L/min."""
        return (self.stroke_volume_ml / 1000) * rate_bpm

    def calculate_pulse_pressure(
        self, rate_bpm: float, vascular_resistance: float
    ) -> tuple[float, float]:
        """
        Calculate resulting pulse pressure.

        Args:
            rate_bpm: Pump rate
            vascular_resistance: Systemic vascular resistance (PRU)

        Returns:
            Tuple of (systolic, diastolic) pressures in mmHg
        """
        cardiac_output = self.calculate_cardiac_output(rate_bpm)
        mean_pressure = cardiac_output * vascular_resistance

        # Pulse pressure depends on arterial compliance
        arterial_compliance = 1.5  # mL/mmHg typical
        pulse_pressure = self.stroke_volume_ml / arterial_compliance

        systolic = mean_pressure + pulse_pressure * 0.6
        diastolic = mean_pressure - pulse_pressure * 0.4

        return systolic, diastolic

    def set_rate(self, rate_bpm: float) -> bool:
        """Set pumping rate."""
        if 30 <= rate_bpm <= self.max_rate_bpm:
            self._current_rate = rate_bpm
            return True
        return False

    def estimate_hemolysis(self) -> float:
        """
        Estimate hemolysis for pulsatile pump.

        Generally higher than continuous flow due to valve interactions.
        """
        # Valve closure creates high shear
        valve_shear_stress = 200  # Pa estimate
        valve_exposure = 0.01  # seconds per cycle

        # Chamber shear is lower
        chamber_shear = 50  # Pa
        chamber_exposure = 60 / self._current_rate - valve_exposure

        hi_valve = self._hemolysis_params.calculate_hemolysis_index(
            valve_shear_stress, valve_exposure
        )
        hi_chamber = self._hemolysis_params.calculate_hemolysis_index(
            chamber_shear, chamber_exposure
        )

        return hi_valve + hi_chamber


class LVAD(ImplantableDevice):
    """
    Complete Left Ventricular Assist Device system.

    Integrates pump, power system, controller, and monitoring.
    """

    def __init__(
        self,
        name: str = "LVAD System",
        manufacturer: str = "Generic",
        model: str = "VAD-1000",
        flow_type: FlowType = FlowType.CONTINUOUS,
        pump_type: PumpType = PumpType.CENTRIFUGAL,
    ):
        super().__init__(
            name=name,
            manufacturer=manufacturer,
            model=model,
            materials=[
                MaterialProperties.titanium_grade5(),
                MaterialProperties.silicone_medical(),
            ]
        )
        self.flow_type = flow_type
        self.pump_type = pump_type

        # Initialize subsystems
        if flow_type in (FlowType.CONTINUOUS, FlowType.HYBRID):
            self._pump = ContinuousFlowPump(pump_type=pump_type)
        else:
            self._pump = PulsatileFlowPump()

        self._power_system = TranscutaneousEnergyTransfer()
        self._controller = PhysiologicalController()

        # Monitoring data
        self._current_flow = 0.0
        self._current_power = 0.0
        self._pulsatility_index = 0.0

    def activate(self) -> bool:
        """Activate the LVAD system."""
        if self._state == DeviceState.OFF:
            self.state = DeviceState.STANDBY

        # Perform startup sequence
        startup_tests = self.perform_self_test()
        if not all(startup_tests.values()):
            self.log_error(f"Startup test failed: {startup_tests}")
            return False

        # Ramp up to minimum operating speed
        if isinstance(self._pump, ContinuousFlowPump):
            self._pump.set_speed(self._controller.min_speed_rpm)

        self.state = DeviceState.ACTIVE
        return True

    def deactivate(self) -> bool:
        """Deactivate the LVAD system safely."""
        if isinstance(self._pump, ContinuousFlowPump):
            # Gradual ramp-down
            current = self._pump._current_speed
            while current > 0:
                current = max(0, current - 500)
                self._pump.set_speed(current)

        self.state = DeviceState.STANDBY
        return True

    def perform_self_test(self) -> dict[str, bool]:
        """Perform comprehensive self-test."""
        tests = {
            "power_system": self._power_system.charge_level > 10,
            "controller": True,  # Always passes in simulation
            "pump_bearings": True,
            "pump_motor": True,
            "sensors": True,
            "communication": True,
        }

        # Check for excessive hemolysis history
        if isinstance(self._pump, ContinuousFlowPump):
            hi = self._pump.estimate_hemolysis(5.0, 6000)
            tests["hemolysis_safe"] = hi < 0.01

        return tests

    def update(self, state: PhysiologicalState, dt: float = 0.01) -> dict:
        """
        Update LVAD operation for one time step.

        Args:
            state: Current physiological state
            dt: Time step in seconds

        Returns:
            Dictionary of current operating parameters
        """
        if self._state != DeviceState.ACTIVE:
            return {"status": "inactive"}

        # Controller update
        target_params = self._controller.adapt_to_physiology(state)

        if isinstance(self._pump, ContinuousFlowPump):
            # Calculate current operating point
            speed = self._controller._current_speed
            afterload = state.mean_arterial_pressure

            flow, power = self._pump.calculate_operating_point(speed, afterload)

            # Update speed based on flow error
            new_speed = self._controller.calculate_speed_command(flow, state)
            self._pump.set_speed(new_speed)

            # Calculate hemolysis
            hemolysis = self._pump.estimate_hemolysis(flow, new_speed)

            self._current_flow = flow
            self._current_power = power

            # Power system discharge
            self._power_system.discharge(power, dt / 3600)

        else:  # Pulsatile pump
            flow = self._pump.calculate_cardiac_output(self._pump._current_rate)
            hemolysis = self._pump.estimate_hemolysis()
            self._current_flow = flow

        # Record telemetry
        telemetry = {
            "flow_lpm": self._current_flow,
            "power_watts": self._current_power,
            "speed_rpm": self._pump._current_speed if isinstance(self._pump, ContinuousFlowPump) else 0,
            "hemolysis_index": hemolysis,
            "battery_level": self._power_system.charge_level,
            **target_params,
        }
        self.record_telemetry(telemetry)

        return telemetry

    def get_comprehensive_status(self) -> dict:
        """Get comprehensive device status."""
        return {
            "device_state": self._state.value,
            "flow_type": self.flow_type.value,
            "pump_status": self._pump.get_status(),
            "power_status": self._power_system.get_status(),
            "biocompatibility": self.get_biocompatibility_assessment(),
            "current_flow_lpm": self._current_flow,
            "current_power_watts": self._current_power,
            "error_count": len(self._error_log),
        }

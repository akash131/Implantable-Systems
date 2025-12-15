"""
Deep Brain Stimulator Module

Implements Deep Brain Stimulation (DBS) systems including:
- Multi-channel electrode arrays
- Closed-loop sensing and stimulation
- MRI-conditional designs
- Adaptive stimulation algorithms
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import math

from .base import (
    ImplantableDevice,
    PowerSystem,
    Controller,
    PIDController,
    Electrode,
    ElectrodeConfiguration,
    PhysiologicalState,
    MaterialProperties,
    DeviceState,
)


class BrainTarget(Enum):
    """Target brain regions for DBS therapy."""
    STN = "subthalamic_nucleus"           # Parkinson's disease
    GPI = "globus_pallidus_interna"       # Parkinson's, dystonia
    VIM = "ventral_intermediate_nucleus"  # Essential tremor
    ANT = "anterior_nucleus_thalamus"     # Epilepsy
    NACC = "nucleus_accumbens"            # OCD, depression
    SCC = "subcallosal_cingulate"         # Depression
    PPN = "pedunculopontine_nucleus"      # Gait disorders


class StimulationMode(Enum):
    """DBS stimulation modes."""
    CONTINUOUS = "continuous"         # Constant stimulation
    CYCLING = "cycling"              # On/off cycling
    CLOSED_LOOP = "closed_loop"      # Feedback-controlled
    TRIGGERED = "triggered"          # Event-triggered


class MRICompatibility(Enum):
    """MRI compatibility classifications."""
    MR_UNSAFE = "mr_unsafe"           # Not safe for MRI
    MR_CONDITIONAL = "mr_conditional" # Safe under specific conditions
    MR_SAFE = "mr_safe"               # Safe under all conditions


@dataclass
class StimulationParameters:
    """Parameters for DBS stimulation."""
    amplitude_ma: float = 2.0           # Current amplitude (mA)
    pulse_width_us: float = 90.0        # Pulse width (microseconds)
    frequency_hz: float = 130.0         # Stimulation frequency (Hz)
    active_contacts: list[int] = field(default_factory=lambda: [0])
    polarity: str = "monopolar"         # monopolar or bipolar
    interleaving: bool = False          # Interleaved stimulation

    @property
    def total_charge_per_pulse_uc(self) -> float:
        """Calculate charge delivered per pulse in microcoulombs."""
        return self.amplitude_ma * self.pulse_width_us / 1000

    @property
    def power_consumption_uw(self) -> float:
        """Estimate power consumption in microwatts."""
        # Simplified model: P = I^2 * R * duty_cycle
        impedance = 1000  # Ohms, typical tissue impedance
        duty_cycle = self.pulse_width_us * self.frequency_hz / 1e6
        return (self.amplitude_ma ** 2) * impedance * duty_cycle * 1000


@dataclass
class NeuralSignal:
    """Recorded neural signal characteristics."""
    lfp_power_beta: float = 0.0    # Beta band (13-30 Hz) power
    lfp_power_gamma: float = 0.0   # Gamma band (30-100 Hz) power
    lfp_power_theta: float = 0.0   # Theta band (4-8 Hz) power
    spike_rate_hz: float = 0.0     # Detected spike rate
    artifact_level: float = 0.0    # Stimulation artifact level

    @property
    def beta_ratio(self) -> float:
        """Beta power relative to total."""
        total = self.lfp_power_beta + self.lfp_power_gamma + self.lfp_power_theta
        if total > 0:
            return self.lfp_power_beta / total
        return 0.0


class MRIConditionalDesign:
    """
    MRI-conditional design features for DBS systems.

    Implements safety features required for MRI scanning with implanted DBS.
    """

    def __init__(
        self,
        max_sar_w_kg: float = 0.1,      # Specific absorption rate limit
        max_b1_rms_ut: float = 2.0,      # B1+RMS limit
        max_gradient_slew_t_m_s: float = 200.0,
        approved_field_strengths: list[float] | None = None,
    ):
        self.max_sar_w_kg = max_sar_w_kg
        self.max_b1_rms_ut = max_b1_rms_ut
        self.max_gradient_slew_t_m_s = max_gradient_slew_t_m_s
        self.approved_field_strengths = approved_field_strengths or [1.5, 3.0]
        self._mri_mode_active = False
        self._lead_configuration = "single"  # single or bilateral

    def calculate_heating_risk(
        self,
        field_strength_t: float,
        sar_w_kg: float,
        scan_duration_min: float,
    ) -> dict:
        """
        Assess RF heating risk during MRI.

        Returns risk assessment and recommendations.
        """
        risk_factors = []

        # Check field strength approval
        if field_strength_t not in self.approved_field_strengths:
            risk_factors.append(f"Field strength {field_strength_t}T not approved")

        # Check SAR
        if sar_w_kg > self.max_sar_w_kg:
            risk_factors.append(f"SAR {sar_w_kg} W/kg exceeds limit {self.max_sar_w_kg}")

        # Bilateral leads increase risk
        heating_factor = 1.0
        if self._lead_configuration == "bilateral":
            heating_factor = 1.5  # Increased coupling risk

        # Estimate temperature rise (simplified model)
        # dT ≈ SAR * time / specific_heat
        specific_heat = 3.5  # J/(g·K) for brain tissue
        duration_s = scan_duration_min * 60
        temp_rise = (sar_w_kg * duration_s / specific_heat) * heating_factor

        return {
            "is_safe": len(risk_factors) == 0 and temp_rise < 2.0,
            "risk_factors": risk_factors,
            "estimated_temp_rise_c": temp_rise,
            "heating_factor": heating_factor,
            "recommendations": self._get_mri_recommendations(temp_rise),
        }

    def _get_mri_recommendations(self, temp_rise: float) -> list[str]:
        """Generate MRI safety recommendations."""
        recs = []
        if temp_rise > 1.0:
            recs.append("Use low SAR sequences when possible")
            recs.append("Consider scan breaks for longer examinations")
        if temp_rise > 1.5:
            recs.append("Limit scan duration to 15 minutes")
            recs.append("Use receive-only head coil")
        recs.append("Verify device in MRI-safe mode before scanning")
        recs.append("Monitor patient throughout scan")
        return recs

    def enter_mri_mode(self) -> dict:
        """
        Configure device for MRI scanning.

        - Disables stimulation
        - Configures impedance monitoring
        - Sets MRI-specific parameters
        """
        self._mri_mode_active = True
        return {
            "mri_mode": True,
            "stimulation_disabled": True,
            "impedance_monitoring": "active",
            "approved_field_strengths": self.approved_field_strengths,
            "max_sar_w_kg": self.max_sar_w_kg,
        }

    def exit_mri_mode(self) -> dict:
        """Exit MRI mode and restore normal operation."""
        self._mri_mode_active = False
        return {
            "mri_mode": False,
            "stimulation_restored": True,
        }


class DBSElectrodeArray(Electrode):
    """
    Multi-channel electrode array for DBS.

    Typically 4-8 contacts per lead, with directional or ring contacts.
    """

    def __init__(
        self,
        num_contacts: int = 8,
        contact_height_mm: float = 1.5,
        contact_spacing_mm: float = 0.5,
        lead_diameter_mm: float = 1.27,
        is_directional: bool = True,
    ):
        config = ElectrodeConfiguration(
            num_electrodes=num_contacts,
            electrode_diameter_um=lead_diameter_mm * 1000,
            inter_electrode_spacing_um=contact_spacing_mm * 1000,
            material=MaterialProperties.platinum_iridium(),
            impedance_range_ohms=(500, 3000),
        )
        super().__init__(config)

        self.num_contacts = num_contacts
        self.contact_height_mm = contact_height_mm
        self.contact_spacing_mm = contact_spacing_mm
        self.lead_diameter_mm = lead_diameter_mm
        self.is_directional = is_directional

        # Directional electrodes have segmented contacts
        self._segments_per_contact = 3 if is_directional else 1
        self._contact_positions: list[float] = []  # mm from tip
        self._calculate_contact_positions()

    def _calculate_contact_positions(self) -> None:
        """Calculate axial positions of each contact."""
        self._contact_positions = []
        for i in range(self.num_contacts):
            position = i * (self.contact_height_mm + self.contact_spacing_mm)
            self._contact_positions.append(position)

    def get_contact_position_mm(self, contact_index: int) -> float:
        """Get axial position of contact from electrode tip."""
        if contact_index < len(self._contact_positions):
            return self._contact_positions[contact_index]
        raise IndexError(f"Contact {contact_index} out of range")

    def configure_directional_steering(
        self,
        contact_index: int,
        segment_weights: list[float],
    ) -> bool:
        """
        Configure directional current steering for a contact.

        Args:
            contact_index: Index of the contact ring
            segment_weights: Weights for each segment (should sum to 1)

        Returns:
            True if configuration successful
        """
        if not self.is_directional:
            return False

        if len(segment_weights) != self._segments_per_contact:
            return False

        if abs(sum(segment_weights) - 1.0) > 0.01:
            return False

        # Store configuration (simplified)
        return True

    def estimate_volume_of_tissue_activated(
        self,
        current_ma: float,
        pulse_width_us: float,
        contact_index: int,
    ) -> float:
        """
        Estimate volume of tissue activated (VTA) in mm³.

        Simplified spherical model based on current and pulse width.
        """
        # Activation threshold approximately 200 mV/mm
        # Radius where field drops to threshold

        # Simplified model: r = k * sqrt(I * pw)
        k = 0.5  # Empirical constant
        radius_mm = k * math.sqrt(current_ma * pulse_width_us / 100)

        # Volume of sphere
        volume = (4 / 3) * math.pi * (radius_mm ** 3)

        return volume


class ClosedLoopController(Controller):
    """
    Closed-loop controller for adaptive DBS.

    Adjusts stimulation based on recorded neural biomarkers.
    """

    def __init__(
        self,
        biomarker: str = "beta_power",
        target_value: float = 0.3,
        min_amplitude_ma: float = 0.5,
        max_amplitude_ma: float = 5.0,
        adaptation_rate: float = 0.1,
        sampling_rate_hz: float = 250.0,
    ):
        super().__init__(sampling_rate_hz, "adaptive_closed_loop")

        self.biomarker = biomarker
        self.target_value = target_value
        self.min_amplitude = min_amplitude_ma
        self.max_amplitude = max_amplitude_ma
        self.adaptation_rate = adaptation_rate

        self._current_amplitude = 2.0
        self._biomarker_history: list[float] = []
        self._amplitude_history: list[float] = []

        # PID gains for biomarker control
        self._gains = {"kp": 1.0, "ki": 0.1, "kd": 0.05}
        self._integral = 0.0
        self._last_error = 0.0

    def compute_output(self, measurement: float, setpoint: float) -> float:
        """Compute amplitude adjustment based on biomarker."""
        error = setpoint - measurement
        dt = 1.0 / self.sampling_rate_hz

        # PID computation
        p_term = self._gains["kp"] * error
        self._integral += error * dt
        self._integral = max(-10, min(10, self._integral))  # Anti-windup
        i_term = self._gains["ki"] * self._integral
        d_term = self._gains["kd"] * (error - self._last_error) / dt
        self._last_error = error

        adjustment = (p_term + i_term + d_term) * self.adaptation_rate

        # Apply adjustment
        new_amplitude = self._current_amplitude + adjustment
        new_amplitude = max(self.min_amplitude, min(self.max_amplitude, new_amplitude))

        self._current_amplitude = new_amplitude
        self._amplitude_history.append(new_amplitude)

        return new_amplitude

    def adapt_to_physiology(self, state: PhysiologicalState) -> dict[str, float]:
        """
        Adapt controller based on patient state.

        Activity and sleep states affect optimal biomarker targets.
        """
        adapted_target = self.target_value

        # Reduce target during sleep (lower beta is normal)
        if state.activity_level < 0.1:
            adapted_target = self.target_value * 0.8

        # Increase target during high activity
        if state.activity_level > 0.7:
            adapted_target = self.target_value * 1.2

        return {
            "target_biomarker": adapted_target,
            "current_amplitude_ma": self._current_amplitude,
            "activity_level": state.activity_level,
        }

    def process_neural_signal(self, signal: NeuralSignal) -> StimulationParameters:
        """
        Process neural signal and compute new stimulation parameters.

        Args:
            signal: Recorded neural signal

        Returns:
            Updated stimulation parameters
        """
        # Extract biomarker
        if self.biomarker == "beta_power":
            biomarker_value = signal.beta_ratio
        elif self.biomarker == "spike_rate":
            biomarker_value = signal.spike_rate_hz / 100  # Normalize
        else:
            biomarker_value = signal.lfp_power_beta

        self._biomarker_history.append(biomarker_value)

        # Compute new amplitude
        new_amplitude = self.compute_output(biomarker_value, self.target_value)

        return StimulationParameters(
            amplitude_ma=new_amplitude,
            pulse_width_us=90.0,
            frequency_hz=130.0,
        )

    def get_therapy_metrics(self) -> dict:
        """Get closed-loop therapy performance metrics."""
        if not self._biomarker_history:
            return {"status": "no_data"}

        recent_biomarker = self._biomarker_history[-100:] if len(self._biomarker_history) > 100 else self._biomarker_history

        return {
            "mean_biomarker": sum(recent_biomarker) / len(recent_biomarker),
            "biomarker_variance": sum((x - sum(recent_biomarker)/len(recent_biomarker))**2 for x in recent_biomarker) / len(recent_biomarker),
            "target_biomarker": self.target_value,
            "current_amplitude_ma": self._current_amplitude,
            "amplitude_adjustments": len(self._amplitude_history),
            "time_on_target_percent": sum(1 for x in recent_biomarker if abs(x - self.target_value) < 0.1) / len(recent_biomarker) * 100,
        }


class DBSBattery(PowerSystem):
    """
    Implantable pulse generator battery for DBS.

    Long-life primary cell or rechargeable options.
    """

    def __init__(
        self,
        capacity_wh: float = 15.0,
        is_rechargeable: bool = True,
        longevity_years: float = 15.0,
    ):
        super().__init__(
            capacity_wh=capacity_wh,
            nominal_voltage=3.7,
            max_discharge_rate=0.05,  # Low power device
        )
        self.is_rechargeable = is_rechargeable
        self.longevity_years = longevity_years
        self._typical_power_uw = 100.0  # Typical DBS power consumption

    @property
    def _current_discharge_rate(self) -> float:
        return self._typical_power_uw / 1e6  # Convert to watts

    def estimate_remaining_longevity(self, power_uw: float) -> float:
        """
        Estimate remaining battery life in years.

        Args:
            power_uw: Current power consumption in microwatts

        Returns:
            Estimated years of remaining battery life
        """
        power_wh_per_year = power_uw * 1e-6 * 24 * 365
        if power_wh_per_year <= 0:
            return float('inf')
        return self._current_charge_wh / power_wh_per_year

    def charge(self, power_watts: float, duration_hours: float) -> float:
        """Charge rechargeable battery."""
        if not self.is_rechargeable:
            return self.charge_level

        # 85% charging efficiency typical for Li-ion
        efficiency = 0.85
        energy = power_watts * duration_hours * efficiency

        self._current_charge_wh = min(self.capacity_wh, self._current_charge_wh + energy)
        self._is_charging = True
        return self.charge_level

    def discharge(self, power_watts: float, duration_hours: float) -> bool:
        """Discharge battery for stimulation."""
        energy = power_watts * duration_hours

        if energy > self._current_charge_wh:
            return False

        self._current_charge_wh -= energy
        self._typical_power_uw = power_watts * 1e6

        if self._current_charge_wh / self.capacity_wh < 0.2:
            self._cycle_count += 1

        return True

    def get_status(self) -> dict:
        """Get battery status with longevity estimate."""
        return {
            **self.get_health_status(),
            "is_rechargeable": self.is_rechargeable,
            "estimated_longevity_years": self.estimate_remaining_longevity(self._typical_power_uw),
            "typical_power_uw": self._typical_power_uw,
        }


class DeepBrainStimulator(ImplantableDevice):
    """
    Complete Deep Brain Stimulation system.

    Integrates implantable pulse generator, leads, and control algorithms.
    """

    def __init__(
        self,
        name: str = "Deep Brain Stimulator",
        manufacturer: str = "Generic",
        model: str = "DBS-3000",
        target: BrainTarget = BrainTarget.STN,
        bilateral: bool = True,
        mri_conditional: bool = True,
    ):
        super().__init__(
            name=name,
            manufacturer=manufacturer,
            model=model,
            materials=[
                MaterialProperties.titanium_grade5(),  # IPG housing
                MaterialProperties.platinum_iridium(),  # Electrodes
                MaterialProperties.silicone_medical(),  # Lead insulation
            ]
        )

        self.target = target
        self.bilateral = bilateral

        # Initialize subsystems
        self._left_lead = DBSElectrodeArray() if bilateral else None
        self._right_lead = DBSElectrodeArray()
        self._power_system = DBSBattery()
        self._controller = ClosedLoopController()

        # MRI safety
        if mri_conditional:
            self._mri_design = MRIConditionalDesign()
            self._mri_design._lead_configuration = "bilateral" if bilateral else "single"
        else:
            self._mri_design = None

        # Stimulation state
        self._mode = StimulationMode.CONTINUOUS
        self._stim_params_left: Optional[StimulationParameters] = None
        self._stim_params_right: StimulationParameters = StimulationParameters()
        self._stimulation_active = False

    def activate(self) -> bool:
        """Activate the DBS system."""
        if self._state == DeviceState.OFF:
            self.state = DeviceState.STANDBY

        # Perform self-test
        tests = self.perform_self_test()
        if not all(tests.values()):
            self.log_error(f"Self-test failed: {tests}")
            return False

        self.state = DeviceState.ACTIVE
        return True

    def deactivate(self) -> bool:
        """Deactivate DBS stimulation."""
        self.stop_stimulation()
        self.state = DeviceState.STANDBY
        return True

    def perform_self_test(self) -> dict[str, bool]:
        """Perform comprehensive self-test."""
        tests = {
            "battery_level": self._power_system.charge_level > 10,
            "right_lead_impedance": True,
            "communication": True,
            "memory": True,
        }

        # Check lead impedances
        right_impedances = self._right_lead.measure_all_impedances()
        tests["right_lead_impedance"] = all(
            self._right_lead.config.impedance_range_ohms[0] <= z <= self._right_lead.config.impedance_range_ohms[1]
            for z in right_impedances
        )

        if self._left_lead:
            left_impedances = self._left_lead.measure_all_impedances()
            tests["left_lead_impedance"] = all(
                self._left_lead.config.impedance_range_ohms[0] <= z <= self._left_lead.config.impedance_range_ohms[1]
                for z in left_impedances
            )

        return tests

    def set_stimulation_parameters(
        self,
        params: StimulationParameters,
        side: str = "right",
    ) -> bool:
        """
        Set stimulation parameters for one side.

        Args:
            params: Stimulation parameters
            side: "left" or "right"

        Returns:
            True if parameters set successfully
        """
        # Safety checks
        if params.amplitude_ma > 10.0:
            self.log_error(f"Amplitude {params.amplitude_ma} mA exceeds safety limit")
            return False

        if params.pulse_width_us > 500:
            self.log_error(f"Pulse width {params.pulse_width_us} us exceeds limit")
            return False

        if params.frequency_hz > 250:
            self.log_error(f"Frequency {params.frequency_hz} Hz exceeds limit")
            return False

        if side == "left" and self._left_lead:
            self._stim_params_left = params
        elif side == "right":
            self._stim_params_right = params
        else:
            return False

        return True

    def start_stimulation(self) -> bool:
        """Start DBS stimulation."""
        if self._state != DeviceState.ACTIVE:
            return False

        if self._stim_params_right is None:
            self.log_error("No stimulation parameters set")
            return False

        self._stimulation_active = True
        return True

    def stop_stimulation(self) -> bool:
        """Stop DBS stimulation."""
        self._stimulation_active = False
        return True

    def set_mode(self, mode: StimulationMode) -> None:
        """Set stimulation mode."""
        self._mode = mode

        if mode == StimulationMode.CLOSED_LOOP:
            # Initialize closed-loop controller
            self._controller.reset() if hasattr(self._controller, 'reset') else None

    def update(
        self,
        neural_signal: NeuralSignal | None = None,
        dt: float = 0.004,  # 250 Hz default
    ) -> dict:
        """
        Update DBS operation for one time step.

        Args:
            neural_signal: Optional recorded neural signal for closed-loop
            dt: Time step in seconds

        Returns:
            Current operating parameters
        """
        if self._state != DeviceState.ACTIVE or not self._stimulation_active:
            return {"status": "inactive"}

        # Closed-loop adaptation
        if self._mode == StimulationMode.CLOSED_LOOP and neural_signal:
            new_params = self._controller.process_neural_signal(neural_signal)
            self._stim_params_right = new_params
            if self._stim_params_left:
                self._stim_params_left = StimulationParameters(
                    amplitude_ma=new_params.amplitude_ma,
                    pulse_width_us=self._stim_params_left.pulse_width_us,
                    frequency_hz=self._stim_params_left.frequency_hz,
                    active_contacts=self._stim_params_left.active_contacts,
                )

        # Calculate power consumption
        power_right = self._stim_params_right.power_consumption_uw if self._stim_params_right else 0
        power_left = self._stim_params_left.power_consumption_uw if self._stim_params_left else 0
        total_power_uw = power_right + power_left + 50  # 50 uW baseline

        # Discharge battery
        self._power_system.discharge(total_power_uw / 1e6, dt / 3600)

        # Calculate VTA
        vta_right = 0.0
        if self._stim_params_right:
            vta_right = self._right_lead.estimate_volume_of_tissue_activated(
                self._stim_params_right.amplitude_ma,
                self._stim_params_right.pulse_width_us,
                self._stim_params_right.active_contacts[0] if self._stim_params_right.active_contacts else 0,
            )

        telemetry = {
            "mode": self._mode.value,
            "stimulation_active": self._stimulation_active,
            "right_amplitude_ma": self._stim_params_right.amplitude_ma if self._stim_params_right else 0,
            "right_frequency_hz": self._stim_params_right.frequency_hz if self._stim_params_right else 0,
            "right_vta_mm3": vta_right,
            "total_power_uw": total_power_uw,
            "battery_level": self._power_system.charge_level,
        }

        if self._mode == StimulationMode.CLOSED_LOOP:
            telemetry["controller_metrics"] = self._controller.get_therapy_metrics()

        self.record_telemetry(telemetry)
        return telemetry

    def enter_mri_safe_mode(self) -> bool:
        """Enter MRI-safe mode."""
        if self._mri_design is None:
            self.log_error("Device is not MRI-conditional")
            return False

        # Stop stimulation
        self.stop_stimulation()

        # Configure MRI mode
        mri_config = self._mri_design.enter_mri_mode()

        self.state = DeviceState.MRI_SAFE
        return True

    def exit_mri_safe_mode(self) -> bool:
        """Exit MRI-safe mode and restore stimulation."""
        if self._mri_design is None:
            return False

        self._mri_design.exit_mri_mode()
        self.state = DeviceState.ACTIVE

        return True

    def assess_mri_safety(
        self,
        field_strength_t: float,
        sar_w_kg: float,
        duration_min: float,
    ) -> dict:
        """
        Assess safety of proposed MRI scan.

        Args:
            field_strength_t: MRI field strength in Tesla
            sar_w_kg: Whole-body SAR in W/kg
            duration_min: Scan duration in minutes

        Returns:
            Safety assessment with recommendations
        """
        if self._mri_design is None:
            return {
                "is_safe": False,
                "reason": "Device is not MRI-conditional",
            }

        return self._mri_design.calculate_heating_risk(
            field_strength_t, sar_w_kg, duration_min
        )

    def get_comprehensive_status(self) -> dict:
        """Get comprehensive device status."""
        status = {
            "device_state": self._state.value,
            "target_region": self.target.value,
            "bilateral": self.bilateral,
            "mode": self._mode.value,
            "stimulation_active": self._stimulation_active,
            "battery_status": self._power_system.get_status(),
            "biocompatibility": self.get_biocompatibility_assessment(),
            "error_count": len(self._error_log),
        }

        if self._stim_params_right:
            status["right_parameters"] = {
                "amplitude_ma": self._stim_params_right.amplitude_ma,
                "pulse_width_us": self._stim_params_right.pulse_width_us,
                "frequency_hz": self._stim_params_right.frequency_hz,
                "power_uw": self._stim_params_right.power_consumption_uw,
            }

        if self._mri_design:
            status["mri_conditional"] = True
            status["mri_mode_active"] = self._mri_design._mri_mode_active

        return status

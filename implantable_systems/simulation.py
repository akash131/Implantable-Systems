"""
Simulation Framework for Implantable Devices.

Provides:
- Time-based device simulation
- Patient physiological models
- Scenario-based testing
- Long-term outcome modeling
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Any, Iterator
from abc import ABC, abstractmethod
import math
import random


class SimulationState(Enum):
    """Simulation states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"


@dataclass
class SimulationConfig:
    """Configuration for device simulation."""
    time_step_seconds: float = 0.01  # 100 Hz default
    real_time: bool = False  # Run at wall-clock speed
    duration_hours: float = 1.0
    random_seed: int | None = None
    log_interval_seconds: float = 1.0
    enable_noise: bool = True
    noise_level: float = 0.02  # 2% noise


class PatientModel(ABC):
    """
    Abstract patient physiological model.

    Simulates patient cardiovascular, neurological, or visual system.
    """

    @abstractmethod
    def update(self, dt: float) -> dict:
        """
        Update patient state for one time step.

        Args:
            dt: Time step in seconds

        Returns:
            Dictionary of current physiological parameters
        """
        pass

    @abstractmethod
    def apply_intervention(self, intervention: str, params: dict) -> None:
        """Apply a therapeutic intervention."""
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """Get current patient state."""
        pass


class CardiovascularModel(PatientModel):
    """
    Cardiovascular system model for LVAD/TAH simulation.

    Models heart function, circulation, and response to mechanical support.
    """

    def __init__(
        self,
        native_heart_function: float = 0.2,  # 0-1, 0 = no function
        body_surface_area: float = 1.9,  # m²
        baseline_heart_rate: float = 80.0,  # bpm
    ):
        self.native_heart_function = native_heart_function
        self.body_surface_area = body_surface_area
        self.baseline_heart_rate = baseline_heart_rate

        # State variables
        self._heart_rate = baseline_heart_rate
        self._mean_arterial_pressure = 70.0  # mmHg (low due to heart failure)
        self._cardiac_output = 3.0  # L/min (reduced)
        self._pulmonary_arterial_pressure = 35.0  # mmHg
        self._central_venous_pressure = 15.0  # mmHg (elevated)
        self._systemic_vascular_resistance = 1200.0  # dyn·s/cm⁵
        self._activity_level = 0.0  # 0-1

        # Device interaction
        self._pump_flow = 0.0  # L/min from mechanical support
        self._pump_speed = 0.0

    def update(self, dt: float) -> dict:
        """Update cardiovascular state."""
        # Calculate native cardiac output based on remaining function
        native_sv = 70.0 * self.native_heart_function  # mL
        native_co = (native_sv / 1000) * self._heart_rate  # L/min

        # Total cardiac output = native + pump
        total_co = native_co + self._pump_flow

        # Autoregulation of SVR based on cardiac output
        target_map = 80.0 + self._activity_level * 20.0
        co_ratio = total_co / 5.0  # Normalized to normal CO

        # SVR adjusts to maintain pressure
        self._systemic_vascular_resistance = 1000.0 / max(co_ratio, 0.3)

        # Calculate pressures
        # MAP = CO × SVR (simplified)
        self._mean_arterial_pressure = total_co * self._systemic_vascular_resistance / 80.0

        # Heart rate response (baroreceptor reflex)
        if self._mean_arterial_pressure < target_map:
            self._heart_rate = min(150, self._heart_rate + 0.5 * dt)
        elif self._mean_arterial_pressure > target_map + 10:
            self._heart_rate = max(50, self._heart_rate - 0.3 * dt)

        # CVP decreases with better pump support
        self._central_venous_pressure = 15.0 - (self._pump_flow / 5.0) * 8.0
        self._central_venous_pressure = max(2, self._central_venous_pressure)

        # PAP follows CVP with offset
        self._pulmonary_arterial_pressure = self._central_venous_pressure + 20.0

        # Update cardiac output
        self._cardiac_output = total_co

        return self.get_state()

    def apply_intervention(self, intervention: str, params: dict) -> None:
        """Apply cardiovascular intervention."""
        if intervention == "pump_support":
            self._pump_flow = params.get("flow_lpm", 0.0)
            self._pump_speed = params.get("speed_rpm", 0.0)
        elif intervention == "exercise":
            self._activity_level = params.get("level", 0.0)
        elif intervention == "medication":
            # Vasodilator
            if params.get("type") == "vasodilator":
                self._systemic_vascular_resistance *= 0.85
            # Inotrope
            elif params.get("type") == "inotrope":
                self.native_heart_function = min(1.0, self.native_heart_function * 1.2)

    def get_state(self) -> dict:
        """Get current cardiovascular state."""
        return {
            "heart_rate_bpm": self._heart_rate,
            "mean_arterial_pressure_mmhg": self._mean_arterial_pressure,
            "cardiac_output_lpm": self._cardiac_output,
            "native_cardiac_output_lpm": self._cardiac_output - self._pump_flow,
            "pump_flow_lpm": self._pump_flow,
            "central_venous_pressure_mmhg": self._central_venous_pressure,
            "pulmonary_arterial_pressure_mmhg": self._pulmonary_arterial_pressure,
            "systemic_vascular_resistance": self._systemic_vascular_resistance,
            "activity_level": self._activity_level,
        }


class NeurologicalModel(PatientModel):
    """
    Neurological model for DBS simulation.

    Models neural activity and response to stimulation.
    """

    def __init__(
        self,
        condition: str = "parkinsons",  # parkinsons, essential_tremor, dystonia
        baseline_symptoms: float = 0.7,  # 0-1 severity
    ):
        self.condition = condition
        self.baseline_symptoms = baseline_symptoms

        # State variables
        self._symptom_severity = baseline_symptoms
        self._beta_power = 0.6  # Pathological beta oscillations
        self._tremor_amplitude = 0.5 if condition == "essential_tremor" else 0.3
        self._bradykinesia_score = 0.6 if condition == "parkinsons" else 0.0
        self._medication_effect = 0.0  # 0-1
        self._stimulation_effect = 0.0  # 0-1

        # Stimulation parameters
        self._stim_amplitude = 0.0
        self._stim_frequency = 0.0

    def update(self, dt: float) -> dict:
        """Update neurological state."""
        # Stimulation effect on beta power
        if self._stim_amplitude > 0 and self._stim_frequency > 100:
            # High-frequency stimulation suppresses beta
            target_beta = 0.2 + 0.4 * math.exp(-self._stim_amplitude / 2.0)
        else:
            target_beta = 0.6 + 0.2 * (1 - self._medication_effect)

        # Gradual change in beta power
        self._beta_power += (target_beta - self._beta_power) * dt * 0.5

        # Symptom severity based on beta power and medication
        self._symptom_severity = self._beta_power * (1 - self._medication_effect * 0.5)

        # Condition-specific symptoms
        if self.condition == "parkinsons":
            self._tremor_amplitude = self._symptom_severity * 0.4
            self._bradykinesia_score = self._symptom_severity * 0.8
        elif self.condition == "essential_tremor":
            self._tremor_amplitude = self._symptom_severity * 0.9
            self._bradykinesia_score = 0.0

        # Medication wearing off
        if self._medication_effect > 0:
            self._medication_effect = max(0, self._medication_effect - dt * 0.0001)

        return self.get_state()

    def apply_intervention(self, intervention: str, params: dict) -> None:
        """Apply neurological intervention."""
        if intervention == "stimulation":
            self._stim_amplitude = params.get("amplitude_ma", 0.0)
            self._stim_frequency = params.get("frequency_hz", 0.0)
        elif intervention == "medication":
            # Levodopa or similar
            self._medication_effect = min(1.0, self._medication_effect + params.get("dose", 0.5))

    def get_state(self) -> dict:
        """Get current neurological state."""
        return {
            "symptom_severity": self._symptom_severity,
            "beta_power": self._beta_power,
            "tremor_amplitude": self._tremor_amplitude,
            "bradykinesia_score": self._bradykinesia_score,
            "medication_effect": self._medication_effect,
            "stimulation_amplitude_ma": self._stim_amplitude,
            "stimulation_frequency_hz": self._stim_frequency,
        }


class VisualModel(PatientModel):
    """
    Visual system model for retinal implant simulation.

    Models visual perception and phosphene generation.
    """

    def __init__(
        self,
        residual_vision: float = 0.0,  # 0-1
        retinal_degeneration: str = "rp",  # rp (retinitis pigmentosa), amd
    ):
        self.residual_vision = residual_vision
        self.retinal_degeneration = retinal_degeneration

        # State variables
        self._phosphene_count = 0
        self._avg_brightness = 0.0
        self._visual_acuity = residual_vision * 0.05  # LogMAR approximation
        self._adaptation_level = 0.0  # Neural adaptation to stimulation

    def update(self, dt: float) -> dict:
        """Update visual state."""
        # Adaptation increases over time with stimulation
        if self._phosphene_count > 0:
            self._adaptation_level = min(1.0, self._adaptation_level + dt * 0.001)
        else:
            self._adaptation_level = max(0.0, self._adaptation_level - dt * 0.01)

        # Visual acuity improves with adaptation
        implant_contribution = self._phosphene_count / 100 * (1 + self._adaptation_level)
        self._visual_acuity = self.residual_vision * 0.05 + implant_contribution * 0.1

        return self.get_state()

    def apply_intervention(self, intervention: str, params: dict) -> None:
        """Apply visual intervention."""
        if intervention == "stimulation":
            self._phosphene_count = params.get("phosphene_count", 0)
            self._avg_brightness = params.get("avg_brightness", 0.0)

    def get_state(self) -> dict:
        """Get current visual state."""
        return {
            "phosphene_count": self._phosphene_count,
            "avg_brightness": self._avg_brightness,
            "visual_acuity_logmar": self._visual_acuity,
            "adaptation_level": self._adaptation_level,
        }


@dataclass
class SimulationEvent:
    """Event during simulation."""
    time_seconds: float
    event_type: str
    parameters: dict
    description: str = ""


class Scenario:
    """
    Simulation scenario with predefined events.

    Allows reproducible testing of device behavior.
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._events: list[SimulationEvent] = []

    def add_event(
        self,
        time_seconds: float,
        event_type: str,
        parameters: dict,
        description: str = "",
    ) -> None:
        """Add an event to the scenario."""
        event = SimulationEvent(
            time_seconds=time_seconds,
            event_type=event_type,
            parameters=parameters,
            description=description,
        )
        self._events.append(event)
        self._events.sort(key=lambda e: e.time_seconds)

    def get_events_at_time(self, time_seconds: float, tolerance: float = 0.01) -> list[SimulationEvent]:
        """Get events that should trigger at given time."""
        return [
            e for e in self._events
            if abs(e.time_seconds - time_seconds) < tolerance
        ]

    @classmethod
    def exercise_test(cls, duration_minutes: float = 10.0) -> "Scenario":
        """Create standard exercise test scenario."""
        scenario = cls(
            name="Exercise Test",
            description="Simulated exercise stress test with ramp-up and recovery"
        )

        # Ramp up
        for t in range(0, int(duration_minutes * 60 / 2), 30):
            level = min(1.0, t / (duration_minutes * 30))
            scenario.add_event(
                time_seconds=float(t),
                event_type="exercise",
                parameters={"level": level},
                description=f"Exercise ramp to {level*100:.0f}%"
            )

        # Peak
        scenario.add_event(
            time_seconds=duration_minutes * 30,
            event_type="exercise",
            parameters={"level": 1.0},
            description="Peak exercise"
        )

        # Recovery
        for t in range(int(duration_minutes * 30), int(duration_minutes * 60), 30):
            level = max(0.0, 1.0 - (t - duration_minutes * 30) / (duration_minutes * 30))
            scenario.add_event(
                time_seconds=float(t),
                event_type="exercise",
                parameters={"level": level},
                description=f"Recovery to {level*100:.0f}%"
            )

        return scenario

    @classmethod
    def battery_depletion(cls, hours: float = 24.0) -> "Scenario":
        """Create battery depletion scenario."""
        scenario = cls(
            name="Battery Depletion",
            description="Simulate battery drain over extended period"
        )

        # Just let time pass - battery drains naturally
        scenario.add_event(
            time_seconds=0,
            event_type="log",
            parameters={"message": "Battery test started"},
        )

        return scenario


class DeviceSimulator:
    """
    Main simulation engine for implantable devices.

    Coordinates device, patient model, and scenario execution.
    """

    def __init__(
        self,
        device,
        patient_model: PatientModel,
        config: SimulationConfig | None = None,
    ):
        self._device = device
        self._patient = patient_model
        self._config = config or SimulationConfig()

        self._state = SimulationState.STOPPED
        self._current_time = 0.0
        self._scenario: Scenario | None = None
        self._log: list[dict] = []
        self._callbacks: list[Callable] = []

        if self._config.random_seed is not None:
            random.seed(self._config.random_seed)

    @property
    def state(self) -> SimulationState:
        """Current simulation state."""
        return self._state

    @property
    def current_time(self) -> float:
        """Current simulation time in seconds."""
        return self._current_time

    def load_scenario(self, scenario: Scenario) -> None:
        """Load a scenario for execution."""
        self._scenario = scenario

    def add_callback(self, callback: Callable[[float, dict], None]) -> None:
        """Add callback for each simulation step."""
        self._callbacks.append(callback)

    def reset(self) -> None:
        """Reset simulation to initial state."""
        self._current_time = 0.0
        self._state = SimulationState.STOPPED
        self._log.clear()

    def run(self) -> Iterator[dict]:
        """
        Run simulation and yield results at each log interval.

        Yields:
            Dictionary with device and patient state
        """
        self._state = SimulationState.RUNNING
        dt = self._config.time_step_seconds
        duration_seconds = self._config.duration_hours * 3600
        log_interval = self._config.log_interval_seconds

        last_log_time = 0.0

        while self._current_time < duration_seconds:
            if self._state == SimulationState.PAUSED:
                continue
            if self._state == SimulationState.STOPPED:
                break

            # Process scenario events
            if self._scenario:
                events = self._scenario.get_events_at_time(self._current_time, dt)
                for event in events:
                    self._process_event(event)

            # Update patient model
            patient_state = self._patient.update(dt)

            # Create physiological state for device
            from .base import PhysiologicalState
            physio = PhysiologicalState(
                heart_rate=patient_state.get("heart_rate_bpm", 70),
                blood_pressure_systolic=patient_state.get("mean_arterial_pressure_mmhg", 80) * 1.2,
                blood_pressure_diastolic=patient_state.get("mean_arterial_pressure_mmhg", 80) * 0.8,
                cardiac_output=patient_state.get("cardiac_output_lpm", 5.0),
                activity_level=patient_state.get("activity_level", 0.0),
            )

            # Update device
            device_state = self._device.update(physio, dt) if hasattr(self._device, 'update') else {}

            # Apply device output to patient
            self._apply_device_to_patient(device_state)

            # Add noise if enabled
            if self._config.enable_noise:
                device_state = self._add_noise(device_state)

            # Log at intervals
            if self._current_time - last_log_time >= log_interval:
                log_entry = {
                    "time": self._current_time,
                    "device": device_state,
                    "patient": patient_state,
                }
                self._log.append(log_entry)

                for callback in self._callbacks:
                    callback(self._current_time, log_entry)

                yield log_entry
                last_log_time = self._current_time

            self._current_time += dt

        self._state = SimulationState.COMPLETED

    def _process_event(self, event: SimulationEvent) -> None:
        """Process a scenario event."""
        if event.event_type == "exercise":
            self._patient.apply_intervention("exercise", event.parameters)
        elif event.event_type == "stimulation":
            self._patient.apply_intervention("stimulation", event.parameters)
        elif event.event_type == "medication":
            self._patient.apply_intervention("medication", event.parameters)

    def _apply_device_to_patient(self, device_state: dict) -> None:
        """Apply device output to patient model."""
        if "flow_lpm" in device_state:
            self._patient.apply_intervention("pump_support", {
                "flow_lpm": device_state.get("flow_lpm", 0),
                "speed_rpm": device_state.get("speed_rpm", 0),
            })

        if "right_amplitude_ma" in device_state:
            self._patient.apply_intervention("stimulation", {
                "amplitude_ma": device_state.get("right_amplitude_ma", 0),
                "frequency_hz": device_state.get("right_frequency_hz", 130),
            })

    def _add_noise(self, state: dict) -> dict:
        """Add measurement noise to state values."""
        noisy = {}
        for key, value in state.items():
            if isinstance(value, (int, float)):
                noise = random.gauss(0, abs(value) * self._config.noise_level)
                noisy[key] = value + noise
            else:
                noisy[key] = value
        return noisy

    def pause(self) -> None:
        """Pause simulation."""
        if self._state == SimulationState.RUNNING:
            self._state = SimulationState.PAUSED

    def resume(self) -> None:
        """Resume paused simulation."""
        if self._state == SimulationState.PAUSED:
            self._state = SimulationState.RUNNING

    def stop(self) -> None:
        """Stop simulation."""
        self._state = SimulationState.STOPPED

    def get_results(self) -> dict:
        """Get simulation results summary."""
        if not self._log:
            return {"error": "No simulation data"}

        # Extract time series
        times = [entry["time"] for entry in self._log]

        # Calculate statistics
        device_params = {}
        patient_params = {}

        for key in self._log[0].get("device", {}):
            values = [
                entry["device"][key] for entry in self._log
                if isinstance(entry["device"].get(key), (int, float))
            ]
            if values:
                device_params[key] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "final": values[-1],
                }

        for key in self._log[0].get("patient", {}):
            values = [
                entry["patient"][key] for entry in self._log
                if isinstance(entry["patient"].get(key), (int, float))
            ]
            if values:
                patient_params[key] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": sum(values) / len(values),
                    "final": values[-1],
                }

        return {
            "duration_seconds": times[-1] if times else 0,
            "samples": len(self._log),
            "device_statistics": device_params,
            "patient_statistics": patient_params,
            "scenario": self._scenario.name if self._scenario else None,
        }

    def export_log(self, format: str = "json") -> str:
        """Export simulation log."""
        import json
        if format == "json":
            return json.dumps(self._log, indent=2)
        elif format == "csv":
            import csv
            import io
            output = io.StringIO()
            if self._log:
                # Flatten nested dicts
                flat_log = []
                for entry in self._log:
                    flat = {"time": entry["time"]}
                    for prefix, data in [("device_", entry.get("device", {})),
                                         ("patient_", entry.get("patient", {}))]:
                        for k, v in data.items():
                            if isinstance(v, (int, float)):
                                flat[prefix + k] = v
                    flat_log.append(flat)

                writer = csv.DictWriter(output, fieldnames=flat_log[0].keys())
                writer.writeheader()
                writer.writerows(flat_log)
            return output.getvalue()
        return ""

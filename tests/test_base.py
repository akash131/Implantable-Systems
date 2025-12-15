"""Tests for base classes and common functionality."""

import pytest
from implantable_systems.base import (
    ImplantableDevice,
    PowerSystem,
    PIDController,
    Electrode,
    ElectrodeConfiguration,
    MaterialProperties,
    PhysiologicalState,
    DeviceState,
    BiocompatibilityRating,
)


class TestMaterialProperties:
    """Tests for MaterialProperties dataclass."""

    def test_titanium_grade5(self):
        """Test titanium grade 5 material properties."""
        ti = MaterialProperties.titanium_grade5()
        assert ti.name == "Titanium Grade 5 (Ti-6Al-4V)"
        assert ti.biocompatibility_rating == BiocompatibilityRating.CLASS_III
        assert ti.corrosion_resistance == 0.95
        assert ti.mri_compatibility is True

    def test_platinum_iridium(self):
        """Test platinum-iridium material properties."""
        pt_ir = MaterialProperties.platinum_iridium()
        assert pt_ir.name == "Platinum-Iridium (90/10)"
        assert pt_ir.corrosion_resistance == 0.99
        assert pt_ir.mri_compatibility is True

    def test_silicone_medical(self):
        """Test medical grade silicone properties."""
        silicone = MaterialProperties.silicone_medical()
        assert silicone.biocompatibility_rating == BiocompatibilityRating.CLASS_III
        assert silicone.electrical_conductivity < 1e-10  # Insulator


class TestPhysiologicalState:
    """Tests for PhysiologicalState dataclass."""

    def test_default_values(self):
        """Test default physiological values."""
        state = PhysiologicalState()
        assert state.heart_rate == 70.0
        assert state.blood_pressure_systolic == 120.0
        assert state.blood_pressure_diastolic == 80.0
        assert state.cardiac_output == 5.0

    def test_mean_arterial_pressure(self):
        """Test MAP calculation."""
        state = PhysiologicalState(
            blood_pressure_systolic=120.0,
            blood_pressure_diastolic=80.0
        )
        # MAP = DBP + (SBP - DBP) / 3 = 80 + 40/3 ≈ 93.33
        assert abs(state.mean_arterial_pressure - 93.33) < 0.1

    def test_exercise_state(self):
        """Test physiological state during exercise."""
        state = PhysiologicalState(
            heart_rate=150.0,
            blood_pressure_systolic=180.0,
            blood_pressure_diastolic=90.0,
            cardiac_output=15.0,
            activity_level=0.8
        )
        assert state.activity_level == 0.8
        assert state.cardiac_output == 15.0


class TestPIDController:
    """Tests for PID controller."""

    def test_proportional_only(self):
        """Test proportional-only control."""
        controller = PIDController(
            sampling_rate_hz=100.0,
            kp=1.0,
            ki=0.0,
            kd=0.0
        )
        # Error of 10 should produce output of 10 with kp=1
        output = controller.compute_output(measurement=90.0, setpoint=100.0)
        assert abs(output - 10.0) < 0.01

    def test_output_limits(self):
        """Test output limiting."""
        controller = PIDController(
            sampling_rate_hz=100.0,
            kp=10.0,
            ki=0.0,
            kd=0.0,
            output_limits=(0.0, 5.0)
        )
        # Large error should be clamped to output limits
        output = controller.compute_output(measurement=0.0, setpoint=100.0)
        assert output == 5.0

    def test_integral_accumulation(self):
        """Test integral term accumulation."""
        controller = PIDController(
            sampling_rate_hz=100.0,
            kp=0.0,
            ki=1.0,
            kd=0.0
        )
        # Multiple calls should accumulate integral
        controller.compute_output(10.0, 20.0)
        output1 = controller.compute_output(10.0, 20.0)
        output2 = controller.compute_output(10.0, 20.0)
        assert output2 > output1  # Integral should grow

    def test_reset(self):
        """Test controller reset."""
        controller = PIDController(
            sampling_rate_hz=100.0,
            kp=1.0,
            ki=1.0,
            kd=1.0
        )
        controller.compute_output(10.0, 20.0)
        controller.reset()
        assert controller._integral == 0.0
        assert controller._last_error == 0.0


class TestElectrode:
    """Tests for Electrode class."""

    def test_electrode_creation(self):
        """Test electrode array creation."""
        config = ElectrodeConfiguration(
            num_electrodes=16,
            electrode_diameter_um=100.0,
            inter_electrode_spacing_um=400.0,
            material=MaterialProperties.platinum_iridium(),
            impedance_range_ohms=(1000, 50000)
        )
        electrode = Electrode(config)
        assert electrode.active_electrode_count == 16

    def test_impedance_measurement(self):
        """Test electrode impedance measurement."""
        config = ElectrodeConfiguration(
            num_electrodes=4,
            electrode_diameter_um=100.0,
            inter_electrode_spacing_um=200.0,
            material=MaterialProperties.platinum_iridium(),
            impedance_range_ohms=(1000, 10000)
        )
        electrode = Electrode(config)
        impedances = electrode.measure_all_impedances()
        assert len(impedances) == 4
        # Default impedance is midpoint of range
        assert impedances[0] == 5500.0

    def test_electrode_disable(self):
        """Test disabling electrodes."""
        config = ElectrodeConfiguration(
            num_electrodes=4,
            electrode_diameter_um=100.0,
            inter_electrode_spacing_um=200.0,
            material=MaterialProperties.platinum_iridium(),
            impedance_range_ohms=(1000, 10000)
        )
        electrode = Electrode(config)
        electrode.set_electrode_active(0, False)
        assert electrode.active_electrode_count == 3

    def test_charge_density_safety(self):
        """Test charge density safety check."""
        config = ElectrodeConfiguration(
            num_electrodes=1,
            electrode_diameter_um=200.0,  # 200 um diameter
            inter_electrode_spacing_um=0.0,
            material=MaterialProperties.platinum_iridium(),
            impedance_range_ohms=(1000, 10000)
        )
        electrode = Electrode(config)

        # Low current/pulse width should be safe
        assert electrode.is_charge_density_safe(current_ua=50, pulse_width_us=100)

        # Very high current should exceed safety limit
        assert not electrode.is_charge_density_safe(current_ua=5000, pulse_width_us=500)

    def test_invalid_electrode_index(self):
        """Test accessing invalid electrode index."""
        config = ElectrodeConfiguration(
            num_electrodes=4,
            electrode_diameter_um=100.0,
            inter_electrode_spacing_um=200.0,
            material=MaterialProperties.platinum_iridium(),
            impedance_range_ohms=(1000, 10000)
        )
        electrode = Electrode(config)
        with pytest.raises(IndexError):
            electrode.measure_impedance(10)

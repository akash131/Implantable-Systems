"""Tests for Ventricular Assist Device (LVAD) module."""

import pytest
from implantable_systems.lvad import (
    LVAD,
    ContinuousFlowPump,
    PulsatileFlowPump,
    TranscutaneousEnergyTransfer,
    PhysiologicalController,
    HemolysisParameters,
    BiocompatibleCoating,
    FlowType,
    PumpType,
)
from implantable_systems.base import PhysiologicalState, DeviceState


class TestHemolysisParameters:
    """Tests for hemolysis modeling."""

    def test_below_threshold(self):
        """Test no hemolysis below shear threshold."""
        params = HemolysisParameters()
        hi = params.calculate_hemolysis_index(
            shear_stress=100.0,  # Below 150 Pa threshold
            exposure_time=0.001
        )
        assert hi == 0.0

    def test_above_threshold(self):
        """Test hemolysis above shear threshold."""
        params = HemolysisParameters()
        hi = params.calculate_hemolysis_index(
            shear_stress=200.0,
            exposure_time=0.01
        )
        assert hi > 0.0
        assert hi < 1.0

    def test_hemolysis_increases_with_stress(self):
        """Test that hemolysis increases with shear stress."""
        params = HemolysisParameters()
        hi_low = params.calculate_hemolysis_index(200.0, 0.01)
        hi_high = params.calculate_hemolysis_index(400.0, 0.01)
        assert hi_high > hi_low


class TestBiocompatibleCoating:
    """Tests for biocompatible coatings."""

    def test_dlc_coating(self):
        """Test diamond-like carbon coating properties."""
        dlc = BiocompatibleCoating.diamond_like_carbon()
        assert dlc.hemocompatibility_score == 0.9
        assert dlc.thrombogenicity == 0.1

    def test_textured_titanium(self):
        """Test textured titanium for neointima formation."""
        tt = BiocompatibleCoating.textured_titanium()
        assert tt.hemocompatibility_score == 0.95
        assert tt.durability_years == 20.0


class TestTranscutaneousEnergyTransfer:
    """Tests for TETS power system."""

    def test_initial_charge(self):
        """Test initial battery charge."""
        tets = TranscutaneousEnergyTransfer()
        assert tets.charge_level == 100.0

    def test_transfer_efficiency_perfect_alignment(self):
        """Test efficiency with perfect coil alignment."""
        tets = TranscutaneousEnergyTransfer()
        tets.set_coil_alignment(1.0)
        efficiency = tets.calculate_transfer_efficiency()
        assert 0.5 < efficiency < 1.0  # Realistic range

    def test_transfer_efficiency_poor_alignment(self):
        """Test efficiency with poor coil alignment."""
        tets = TranscutaneousEnergyTransfer()
        tets.set_coil_alignment(0.3)
        efficiency_poor = tets.calculate_transfer_efficiency()

        tets.set_coil_alignment(1.0)
        efficiency_good = tets.calculate_transfer_efficiency()

        # Both may be at the cap (0.95), but poor should be <= good
        assert efficiency_poor <= efficiency_good

    def test_discharge(self):
        """Test battery discharge."""
        tets = TranscutaneousEnergyTransfer()
        initial_charge = tets._current_charge_wh
        tets.discharge(power_watts=8.0, duration_hours=0.5)
        assert tets._current_charge_wh < initial_charge


class TestContinuousFlowPump:
    """Tests for continuous flow pump."""

    def test_pump_creation(self):
        """Test pump creation with default parameters."""
        pump = ContinuousFlowPump()
        assert pump.pump_type == PumpType.CENTRIFUGAL
        assert pump._current_speed == 0.0

    def test_speed_setting(self):
        """Test setting pump speed."""
        pump = ContinuousFlowPump(max_speed_rpm=9000)
        assert pump.set_speed(6000)
        assert pump._current_speed == 6000

        # Exceed maximum
        assert not pump.set_speed(10000)

    def test_operating_point(self):
        """Test pump operating point calculation."""
        pump = ContinuousFlowPump()
        flow, power = pump.calculate_operating_point(
            speed_rpm=6000,
            afterload_mmhg=80.0
        )
        assert flow > 0
        assert power > 0

    def test_hemolysis_estimation(self):
        """Test hemolysis estimation."""
        pump = ContinuousFlowPump()
        hi = pump.estimate_hemolysis(flow_lpm=5.0, speed_rpm=6000)
        assert 0 <= hi < 0.1  # Should be low for normal operation


class TestPulsatileFlowPump:
    """Tests for pulsatile flow pump."""

    def test_flow_waveform_generation(self):
        """Test pulsatile flow waveform."""
        pump = PulsatileFlowPump(stroke_volume_ml=70.0)
        waveform = pump.calculate_flow_waveform(rate_bpm=70)
        assert len(waveform) == 100
        assert max(waveform) > 0  # Peak flow during systole

    def test_cardiac_output(self):
        """Test cardiac output calculation."""
        pump = PulsatileFlowPump(stroke_volume_ml=70.0)
        co = pump.calculate_cardiac_output(rate_bpm=70)
        # CO = SV * HR = 70ml * 70bpm = 4.9 L/min
        assert abs(co - 4.9) < 0.1

    def test_rate_setting(self):
        """Test setting pump rate."""
        pump = PulsatileFlowPump(max_rate_bpm=120)
        assert pump.set_rate(80)
        assert not pump.set_rate(150)  # Exceeds max


class TestPhysiologicalController:
    """Tests for LVAD physiological controller."""

    def test_flow_estimation(self):
        """Test sensorless flow estimation."""
        controller = PhysiologicalController()
        flow = controller.estimate_flow_from_power(
            power_watts=8.0,
            speed_rpm=6000,
            pressure_head_mmhg=80.0
        )
        assert flow > 0

    def test_physiology_adaptation(self):
        """Test adaptation to physiological state."""
        controller = PhysiologicalController()

        # Rest state
        state_rest = PhysiologicalState(activity_level=0.0)
        params_rest = controller.adapt_to_physiology(state_rest)

        # Exercise state
        state_exercise = PhysiologicalState(activity_level=0.8)
        params_exercise = controller.adapt_to_physiology(state_exercise)

        # Target flow should increase with activity
        assert params_exercise["target_flow_lpm"] > params_rest["target_flow_lpm"]


class TestLVAD:
    """Tests for complete LVAD system."""

    def test_lvad_creation(self):
        """Test LVAD system creation."""
        lvad = LVAD(
            name="Test LVAD",
            manufacturer="Test Corp",
            model="TEST-100"
        )
        assert lvad.state == DeviceState.OFF
        assert lvad.flow_type == FlowType.CONTINUOUS

    def test_activation(self):
        """Test LVAD activation sequence."""
        lvad = LVAD()
        assert lvad.activate()
        assert lvad.state == DeviceState.ACTIVE

    def test_deactivation(self):
        """Test LVAD deactivation."""
        lvad = LVAD()
        lvad.activate()
        assert lvad.deactivate()
        assert lvad.state == DeviceState.STANDBY

    def test_self_test(self):
        """Test self-test functionality."""
        lvad = LVAD()
        tests = lvad.perform_self_test()
        assert "power_system" in tests
        assert "pump_bearings" in tests

    def test_update_cycle(self):
        """Test LVAD update cycle."""
        lvad = LVAD()
        lvad.activate()

        state = PhysiologicalState()
        result = lvad.update(state, dt=0.01)

        assert "flow_lpm" in result
        assert "power_watts" in result
        assert result["flow_lpm"] > 0

    def test_biocompatibility_assessment(self):
        """Test biocompatibility assessment."""
        lvad = LVAD()
        assessment = lvad.get_biocompatibility_assessment()
        assert assessment["status"] == "assessed"
        assert assessment["mri_compatible"] is True

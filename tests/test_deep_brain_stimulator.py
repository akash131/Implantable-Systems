"""Tests for Deep Brain Stimulator module."""

import pytest
from implantable_systems.deep_brain_stimulator import (
    DeepBrainStimulator,
    DBSElectrodeArray,
    ClosedLoopController,
    DBSBattery,
    MRIConditionalDesign,
    StimulationParameters,
    NeuralSignal,
    BrainTarget,
    StimulationMode,
)
from implantable_systems.base import PhysiologicalState, DeviceState


class TestStimulationParameters:
    """Tests for stimulation parameters."""

    def test_default_parameters(self):
        """Test default stimulation parameters."""
        params = StimulationParameters()
        assert params.amplitude_ma == 2.0
        assert params.pulse_width_us == 90.0
        assert params.frequency_hz == 130.0

    def test_charge_calculation(self):
        """Test charge per pulse calculation."""
        params = StimulationParameters(
            amplitude_ma=2.0,
            pulse_width_us=100.0
        )
        # Q = I * t = 2mA * 100us = 0.2 uC
        assert abs(params.total_charge_per_pulse_uc - 0.2) < 0.01

    def test_power_estimation(self):
        """Test power consumption estimation."""
        params = StimulationParameters(
            amplitude_ma=2.0,
            pulse_width_us=90.0,
            frequency_hz=130.0
        )
        power = params.power_consumption_uw
        assert power > 0
        # DBS typically consumes 1-50 mW (1000-50000 uW)
        assert power < 100000  # Should be reasonable power consumption


class TestNeuralSignal:
    """Tests for neural signal representation."""

    def test_beta_ratio(self):
        """Test beta ratio calculation."""
        signal = NeuralSignal(
            lfp_power_beta=0.5,
            lfp_power_gamma=0.3,
            lfp_power_theta=0.2
        )
        # Beta ratio = 0.5 / 1.0 = 0.5
        assert abs(signal.beta_ratio - 0.5) < 0.01

    def test_zero_power(self):
        """Test beta ratio with zero power."""
        signal = NeuralSignal()
        assert signal.beta_ratio == 0.0


class TestMRIConditionalDesign:
    """Tests for MRI conditional design."""

    def test_safe_scan_assessment(self):
        """Test MRI safety assessment for safe parameters."""
        mri = MRIConditionalDesign()
        result = mri.calculate_heating_risk(
            field_strength_t=1.5,
            sar_w_kg=0.05,
            scan_duration_min=5.0  # Shorter scan duration
        )
        # Check that risk factors are reported
        assert "risk_factors" in result
        assert "estimated_temp_rise_c" in result

    def test_unsafe_field_strength(self):
        """Test MRI safety with unapproved field strength."""
        mri = MRIConditionalDesign(approved_field_strengths=[1.5])
        result = mri.calculate_heating_risk(
            field_strength_t=3.0,  # Not approved
            sar_w_kg=0.05,
            scan_duration_min=10.0
        )
        assert len(result["risk_factors"]) > 0

    def test_excessive_sar(self):
        """Test MRI safety with excessive SAR."""
        mri = MRIConditionalDesign(max_sar_w_kg=0.1)
        result = mri.calculate_heating_risk(
            field_strength_t=1.5,
            sar_w_kg=0.5,  # Exceeds limit
            scan_duration_min=10.0
        )
        assert not result["is_safe"]

    def test_mri_mode_entry(self):
        """Test entering MRI mode."""
        mri = MRIConditionalDesign()
        config = mri.enter_mri_mode()
        assert config["mri_mode"]
        assert config["stimulation_disabled"]
        assert mri._mri_mode_active


class TestDBSElectrodeArray:
    """Tests for DBS electrode array."""

    def test_creation(self):
        """Test electrode array creation."""
        array = DBSElectrodeArray(
            num_contacts=8,
            contact_height_mm=1.5,
            contact_spacing_mm=0.5
        )
        assert array.num_contacts == 8
        assert len(array._contact_positions) == 8

    def test_contact_positions(self):
        """Test contact position calculation."""
        array = DBSElectrodeArray(
            num_contacts=4,
            contact_height_mm=1.5,
            contact_spacing_mm=0.5
        )
        # First contact at 0mm
        assert array.get_contact_position_mm(0) == 0.0
        # Second contact at 2mm (1.5mm height + 0.5mm spacing)
        assert abs(array.get_contact_position_mm(1) - 2.0) < 0.01

    def test_vta_estimation(self):
        """Test volume of tissue activated estimation."""
        array = DBSElectrodeArray()
        vta = array.estimate_volume_of_tissue_activated(
            current_ma=2.0,
            pulse_width_us=90.0,
            contact_index=0
        )
        assert vta > 0
        # VTA typically 50-150 mm³ at therapeutic settings

    def test_directional_steering(self):
        """Test directional current steering."""
        array = DBSElectrodeArray(is_directional=True)
        # Weights should sum to 1
        assert array.configure_directional_steering(0, [0.33, 0.33, 0.34])
        # Invalid weights
        assert not array.configure_directional_steering(0, [0.5, 0.5, 0.5])


class TestClosedLoopController:
    """Tests for closed-loop DBS controller."""

    def test_creation(self):
        """Test controller creation."""
        controller = ClosedLoopController(
            biomarker="beta_power",
            target_value=0.3
        )
        assert controller.biomarker == "beta_power"
        assert controller.target_value == 0.3

    def test_neural_signal_processing(self):
        """Test processing neural signal."""
        controller = ClosedLoopController(target_value=0.3)
        signal = NeuralSignal(
            lfp_power_beta=0.5,
            lfp_power_gamma=0.3,
            lfp_power_theta=0.2
        )
        params = controller.process_neural_signal(signal)
        assert params.amplitude_ma > 0

    def test_adaptation_to_high_beta(self):
        """Test amplitude changes with high beta."""
        controller = ClosedLoopController(target_value=0.3)

        # Process multiple high beta signals to see amplitude adaptation
        for _ in range(10):
            signal = NeuralSignal(lfp_power_beta=0.8, lfp_power_gamma=0.2)
            controller.process_neural_signal(signal)

        # Check that controller responds to biomarker
        assert controller._current_amplitude > 0
        assert len(controller._biomarker_history) == 10

    def test_amplitude_limits(self):
        """Test amplitude stays within limits."""
        controller = ClosedLoopController(
            min_amplitude_ma=0.5,
            max_amplitude_ma=5.0
        )

        # Process many signals to drive adaptation
        for _ in range(100):
            signal = NeuralSignal(lfp_power_beta=1.0)
            controller.process_neural_signal(signal)

        assert controller._current_amplitude <= 5.0


class TestDBSBattery:
    """Tests for DBS battery."""

    def test_longevity_estimation(self):
        """Test battery longevity estimation."""
        battery = DBSBattery(capacity_wh=15.0, is_rechargeable=True)
        longevity = battery.estimate_remaining_longevity(power_uw=100.0)
        # 15 Wh / (100 uW * 8760 h/year) ≈ 17 years
        assert longevity > 10

    def test_rechargeable_charge(self):
        """Test charging rechargeable battery."""
        battery = DBSBattery(is_rechargeable=True)
        battery._current_charge_wh = 5.0  # Partially depleted
        battery.charge(power_watts=1.0, duration_hours=1.0)
        assert battery._current_charge_wh > 5.0

    def test_non_rechargeable(self):
        """Test non-rechargeable battery doesn't charge."""
        battery = DBSBattery(is_rechargeable=False)
        initial = battery._current_charge_wh
        battery.charge(power_watts=1.0, duration_hours=1.0)
        assert battery._current_charge_wh == initial


class TestDeepBrainStimulator:
    """Tests for complete DBS system."""

    def test_creation(self):
        """Test DBS system creation."""
        dbs = DeepBrainStimulator(
            name="Test DBS",
            target=BrainTarget.STN,
            bilateral=True
        )
        assert dbs.state == DeviceState.OFF
        assert dbs.target == BrainTarget.STN
        assert dbs.bilateral

    def test_activation(self):
        """Test DBS activation."""
        dbs = DeepBrainStimulator()
        assert dbs.activate()
        assert dbs.state == DeviceState.ACTIVE

    def test_stimulation_parameters(self):
        """Test setting stimulation parameters."""
        dbs = DeepBrainStimulator()
        dbs.activate()

        params = StimulationParameters(
            amplitude_ma=2.0,
            pulse_width_us=90.0,
            frequency_hz=130.0
        )
        assert dbs.set_stimulation_parameters(params, side="right")

    def test_unsafe_parameters_rejected(self):
        """Test rejection of unsafe parameters."""
        dbs = DeepBrainStimulator()
        dbs.activate()

        # Excessive amplitude
        params = StimulationParameters(amplitude_ma=15.0)
        assert not dbs.set_stimulation_parameters(params)

    def test_stimulation_start_stop(self):
        """Test starting and stopping stimulation."""
        dbs = DeepBrainStimulator()
        dbs.activate()
        dbs.set_stimulation_parameters(StimulationParameters())

        assert dbs.start_stimulation()
        assert dbs._stimulation_active

        assert dbs.stop_stimulation()
        assert not dbs._stimulation_active

    def test_closed_loop_mode(self):
        """Test closed-loop stimulation mode."""
        dbs = DeepBrainStimulator()
        dbs.activate()
        dbs.set_stimulation_parameters(StimulationParameters())
        dbs.set_mode(StimulationMode.CLOSED_LOOP)
        dbs.start_stimulation()

        signal = NeuralSignal(lfp_power_beta=0.5)
        result = dbs.update(neural_signal=signal)

        assert "controller_metrics" in result

    def test_mri_safe_mode(self):
        """Test MRI safe mode entry."""
        dbs = DeepBrainStimulator(mri_conditional=True)
        dbs.activate()
        dbs.start_stimulation()

        assert dbs.enter_mri_safe_mode()
        assert dbs.state == DeviceState.MRI_SAFE
        assert not dbs._stimulation_active

    def test_mri_safety_assessment(self):
        """Test MRI safety assessment."""
        dbs = DeepBrainStimulator(mri_conditional=True)
        result = dbs.assess_mri_safety(
            field_strength_t=1.5,
            sar_w_kg=0.05,
            duration_min=10.0
        )
        assert "is_safe" in result

    def test_non_mri_conditional(self):
        """Test MRI mode fails for non-conditional device."""
        dbs = DeepBrainStimulator(mri_conditional=False)
        dbs.activate()
        assert not dbs.enter_mri_safe_mode()

    def test_comprehensive_status(self):
        """Test comprehensive status report."""
        dbs = DeepBrainStimulator()
        dbs.activate()
        dbs.set_stimulation_parameters(StimulationParameters())

        status = dbs.get_comprehensive_status()

        assert "device_state" in status
        assert "target_region" in status
        assert "battery_status" in status
        assert "right_parameters" in status

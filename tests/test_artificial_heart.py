"""Tests for Total Artificial Heart module."""

import pytest
from implantable_systems.artificial_heart import (
    TotalArtificialHeart,
    DualVentricle,
    HeartPowerSystem,
    TAHController,
    VentricleType,
    TAHMode,
    VentricleParameters,
)
from implantable_systems.base import PhysiologicalState, DeviceState


class TestHeartPowerSystem:
    """Tests for TAH power system."""

    def test_initial_state(self):
        """Test initial power system state."""
        power = HeartPowerSystem()
        assert power.charge_level == 100.0
        assert power._external_charge == power.external_battery_wh

    def test_total_available_energy(self):
        """Test total energy calculation."""
        power = HeartPowerSystem(
            internal_battery_wh=30.0,
            external_battery_wh=150.0
        )
        total = power.get_total_available_energy()
        assert total == 180.0

    def test_discharge_internal_first(self):
        """Test discharge uses internal battery first."""
        power = HeartPowerSystem()
        power.discharge(power_watts=10.0, duration_hours=0.1)
        assert power._power_source == "internal"

    def test_automatic_source_switching(self):
        """Test automatic switch to external when internal depletes."""
        power = HeartPowerSystem(internal_battery_wh=1.0)
        # Discharge more than internal capacity
        power.discharge(power_watts=10.0, duration_hours=0.2)
        assert power._power_source == "external"

    def test_external_battery_swap(self):
        """Test external battery swap."""
        power = HeartPowerSystem()
        power._external_charge = 10.0  # Low external
        power.swap_external_battery(new_capacity_wh=150.0)
        assert power._external_charge == 150.0


class TestDualVentricle:
    """Tests for dual ventricle system."""

    def test_creation(self):
        """Test dual ventricle creation."""
        ventricles = DualVentricle()
        assert ventricles._rate == 80
        assert ventricles._left_stroke_volume == 65.0
        assert ventricles._right_stroke_volume == 65.0

    def test_cardiac_output(self):
        """Test cardiac output calculation."""
        ventricles = DualVentricle()
        ventricles._rate = 70
        ventricles._left_stroke_volume = 70.0
        # CO = 70ml * 70bpm = 4.9 L/min
        assert abs(ventricles.left_cardiac_output - 4.9) < 0.1

    def test_output_balance(self):
        """Test output balance calculation."""
        ventricles = DualVentricle()
        ventricles._left_stroke_volume = 65.0
        ventricles._right_stroke_volume = 65.0
        assert abs(ventricles.output_balance - 1.0) < 0.01

    def test_imbalanced_output(self):
        """Test detection of imbalanced output."""
        ventricles = DualVentricle()
        ventricles._left_stroke_volume = 70.0
        ventricles._right_stroke_volume = 50.0
        assert ventricles.output_balance > 1.2  # LV > RV

    def test_atrial_pressure_calculation(self):
        """Test atrial pressure estimation."""
        ventricles = DualVentricle()
        lap, rap = ventricles.calculate_atrial_pressures()
        # With balanced output, pressures should be normal
        assert 5 < lap < 15
        assert 2 < rap < 10

    def test_balance_outputs(self):
        """Test automatic output balancing."""
        ventricles = DualVentricle()
        # Create imbalance by setting different stroke volumes
        ventricles._left_stroke_volume = 50.0  # Low LV output
        ventricles._right_stroke_volume = 70.0  # Higher RV output
        # This will cause elevated LAP when calculate_atrial_pressures is called
        result = ventricles.balance_outputs()
        # Result should include final_balance
        assert "final_balance" in result

    def test_pressure_waveform(self):
        """Test pressure waveform generation."""
        ventricles = DualVentricle()
        waveform = ventricles.generate_pressure_waveform(VentricleType.LEFT)
        assert len(waveform) == 100
        assert max(waveform) > 100  # LV peak > 100 mmHg

        # RV pressure should be lower
        waveform_rv = ventricles.generate_pressure_waveform(VentricleType.RIGHT)
        assert max(waveform_rv) < max(waveform)

    def test_rate_limits(self):
        """Test rate setting limits."""
        ventricles = DualVentricle()
        assert ventricles.set_rate(80)
        assert not ventricles.set_rate(200)  # Exceeds max


class TestTAHController:
    """Tests for TAH controller."""

    def test_target_rate_at_rest(self):
        """Test target rate calculation at rest."""
        controller = TAHController()
        state = PhysiologicalState(activity_level=0.0)
        rate = controller.calculate_target_rate(state)
        assert 50 < rate < 90  # Rest rate

    def test_target_rate_during_exercise(self):
        """Test target rate calculation during exercise."""
        controller = TAHController()
        state_rest = PhysiologicalState(activity_level=0.0)
        state_exercise = PhysiologicalState(activity_level=0.8)

        rate_rest = controller.calculate_target_rate(state_rest)
        rate_exercise = controller.calculate_target_rate(state_exercise)

        assert rate_exercise > rate_rest

    def test_rate_limits(self):
        """Test rate stays within limits."""
        controller = TAHController(min_rate_bpm=50, max_rate_bpm=130)
        state = PhysiologicalState(activity_level=1.0)  # Max activity
        rate = controller.calculate_target_rate(state)
        assert 50 <= rate <= 130


class TestTotalArtificialHeart:
    """Tests for complete TAH system."""

    def test_creation(self):
        """Test TAH system creation."""
        tah = TotalArtificialHeart(
            name="Test TAH",
            manufacturer="Test Corp",
            model="TAH-TEST"
        )
        assert tah.state == DeviceState.OFF

    def test_activation(self):
        """Test TAH activation."""
        tah = TotalArtificialHeart()
        assert tah.activate()
        assert tah.state == DeviceState.ACTIVE

    def test_self_test(self):
        """Test TAH self-test."""
        tah = TotalArtificialHeart()
        tests = tah.perform_self_test()
        assert "power_internal" in tests
        assert "left_ventricle_motor" in tests
        assert "output_balance" in tests

    def test_update_cycle(self):
        """Test TAH update cycle."""
        tah = TotalArtificialHeart()
        tah.activate()

        state = PhysiologicalState()
        result = tah.update(state, dt=0.02)

        assert "rate_bpm" in result
        assert "left_cardiac_output_lpm" in result
        assert "output_balance" in result

    def test_emergency_backup_mode(self):
        """Test emergency backup mode."""
        tah = TotalArtificialHeart()
        tah.activate()
        assert tah.emergency_backup_mode()
        # Should be at minimum rate
        assert tah._ventricles._rate == tah._controller.min_rate

    def test_alarm_status(self):
        """Test alarm detection."""
        tah = TotalArtificialHeart()
        tah.activate()

        alarms = tah.get_alarm_status()
        # With normal operation, should have no critical alarms
        assert not alarms.get("low_battery_critical", False)

    def test_low_battery_alarm(self):
        """Test low battery alarm."""
        tah = TotalArtificialHeart()
        tah._power_system._current_charge_wh = 1.0  # Very low
        tah.activate()

        alarms = tah.get_alarm_status()
        assert alarms.get("low_battery_critical") or alarms.get("low_battery_warning")

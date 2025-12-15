"""Tests for Retinal Implant module."""

import pytest
from implantable_systems.retinal_implant import (
    RetinalImplant,
    MicroelectrodeArray,
    ImageProcessor,
    WirelessPowerDataLink,
    PhospheneModel,
    ImageProcessingParams,
    ImplantLocation,
    StimulationStrategy,
)
from implantable_systems.base import DeviceState


class TestPhospheneModel:
    """Tests for phosphene prediction model."""

    def test_below_threshold(self):
        """Test no phosphene below threshold current."""
        phosphene = PhospheneModel.from_stimulation(
            current_ua=10.0,  # Below typical threshold
            pulse_width_us=100.0,
            frequency_hz=50.0,
            electrode_size_um=200.0
        )
        assert phosphene.brightness == 0.0

    def test_above_threshold(self):
        """Test phosphene generation above threshold."""
        phosphene = PhospheneModel.from_stimulation(
            current_ua=100.0,
            pulse_width_us=200.0,
            frequency_hz=50.0,
            electrode_size_um=200.0
        )
        assert phosphene.brightness > 0

    def test_brightness_saturation(self):
        """Test brightness saturation at high currents."""
        phosphene_low = PhospheneModel.from_stimulation(100.0, 200.0, 50.0, 200.0)
        phosphene_high = PhospheneModel.from_stimulation(1000.0, 200.0, 50.0, 200.0)

        assert phosphene_high.brightness > phosphene_low.brightness
        assert phosphene_high.brightness <= 1.0


class TestMicroelectrodeArray:
    """Tests for microelectrode array."""

    def test_creation(self):
        """Test array creation."""
        array = MicroelectrodeArray(
            rows=10,
            cols=10,
            electrode_diameter_um=200.0,
            pitch_um=400.0
        )
        assert array.config.num_electrodes == 100
        assert array.rows == 10
        assert array.cols == 10

    def test_electrode_position(self):
        """Test electrode position calculation."""
        array = MicroelectrodeArray(
            rows=3,
            cols=3,
            pitch_um=500.0
        )
        # Electrode 4 is at row 1, col 1
        x, y = array.get_electrode_position(4)
        assert abs(x - 0.5) < 0.01  # 500um = 0.5mm
        assert abs(y - 0.5) < 0.01

    def test_visual_field_mapping(self):
        """Test mapping to visual field coordinates."""
        array = MicroelectrodeArray(
            rows=5,
            cols=5,
            pitch_um=400.0
        )
        # Center electrode should be near (0, 0) visual degrees
        center_idx = 12  # Row 2, Col 2
        x_deg, y_deg = array.map_to_visual_field(center_idx)
        assert abs(x_deg) < 1.0
        assert abs(y_deg) < 1.0

    def test_stimulation_pattern(self):
        """Test setting stimulation pattern."""
        array = MicroelectrodeArray(rows=3, cols=3, electrode_diameter_um=400.0)  # Larger electrodes
        pattern = [
            (0, 50.0, 100.0),   # electrode 0, 50uA, 100us - safe charge density
            (4, 75.0, 100.0),   # electrode 4, 75uA, 100us - safe charge density
        ]
        array.set_stimulation_pattern(pattern)
        assert len(array._stimulation_map) == 2

    def test_array_coverage(self):
        """Test visual field coverage calculation."""
        array = MicroelectrodeArray(
            rows=10,
            cols=10,
            pitch_um=400.0  # 4mm total
        )
        width_deg, height_deg = array.get_array_coverage_deg()
        assert width_deg > 0
        assert height_deg > 0


class TestImageProcessor:
    """Tests for image processing unit."""

    def test_creation(self):
        """Test image processor creation."""
        array = MicroelectrodeArray(rows=6, cols=6)
        processor = ImageProcessor(array)
        assert processor.electrode_array == array

    def test_frame_processing(self):
        """Test image frame processing."""
        array = MicroelectrodeArray(rows=3, cols=3)
        processor = ImageProcessor(array)

        # Create a simple 9x9 test image
        image = [[0.5] * 9 for _ in range(9)]
        pattern = processor.process_frame(image)

        assert len(pattern) > 0  # Should generate some stimulation
        for electrode, current, width in pattern:
            assert 0 <= electrode < 9
            assert current > 0
            assert width > 0

    def test_empty_image(self):
        """Test processing empty (black) image."""
        array = MicroelectrodeArray(rows=3, cols=3)
        processor = ImageProcessor(array)

        image = [[0.0] * 9 for _ in range(9)]
        pattern = processor.process_frame(image)

        # Black image should produce minimal stimulation
        assert len(pattern) == 0 or all(c < 10 for _, c, _ in pattern)

    def test_strategy_change(self):
        """Test changing stimulation strategy."""
        array = MicroelectrodeArray(rows=3, cols=3)
        processor = ImageProcessor(array)

        processor.set_strategy(StimulationStrategy.PULSE_WIDTH_CODING)
        assert processor._strategy == StimulationStrategy.PULSE_WIDTH_CODING


class TestWirelessPowerDataLink:
    """Tests for wireless power and data system."""

    def test_link_establishment(self):
        """Test wireless link establishment."""
        link = WirelessPowerDataLink()
        assert link.establish_link(distance_mm=20.0)
        assert link._is_linked

    def test_link_quality_vs_distance(self):
        """Test link quality decreases with distance."""
        link = WirelessPowerDataLink()

        link.establish_link(distance_mm=10.0)
        quality_close = link._link_quality

        link.establish_link(distance_mm=30.0)
        quality_far = link._link_quality

        assert quality_close > quality_far

    def test_link_failure_at_distance(self):
        """Test link fails at large distances."""
        link = WirelessPowerDataLink()
        assert not link.establish_link(distance_mm=50.0)
        assert not link._is_linked

    def test_data_transmission(self):
        """Test data transmission time calculation."""
        link = WirelessPowerDataLink(data_rate_kbps=1000.0)
        link.establish_link(distance_mm=15.0)

        # 1000 bits at 1000 kbps = 1 ms (at perfect link quality)
        time_ms = link.transmit_data(data_bits=1000)
        assert time_ms > 0
        assert time_ms < 10  # Should be fast


class TestRetinalImplant:
    """Tests for complete retinal implant system."""

    def test_creation(self):
        """Test implant creation."""
        implant = RetinalImplant(
            name="Test Implant",
            array_rows=6,
            array_cols=10,
            electrode_size_um=200.0
        )
        assert implant.state == DeviceState.OFF
        assert implant._electrode_array.rows == 6
        assert implant._electrode_array.cols == 10

    def test_wireless_link_required(self):
        """Test that activation requires wireless link."""
        implant = RetinalImplant()
        # Without link, activation should fail
        assert not implant.activate()

    def test_activation_with_link(self):
        """Test successful activation with wireless link."""
        implant = RetinalImplant()
        implant.establish_wireless_link(distance_mm=15.0)
        assert implant.activate()
        assert implant.state == DeviceState.ACTIVE

    def test_frame_processing(self):
        """Test frame processing in active state."""
        implant = RetinalImplant(array_rows=3, array_cols=3)
        implant.establish_wireless_link(distance_mm=15.0)
        implant.activate()

        image = [[0.5] * 9 for _ in range(9)]
        result = implant.process_frame(image)

        # Result should contain processing metrics
        assert "active_electrodes" in result
        assert "phosphenes_generated" in result
        assert "frames_processed" in result

    def test_visual_field_coverage(self):
        """Test visual field coverage reporting."""
        implant = RetinalImplant(array_rows=10, array_cols=15)
        coverage = implant.get_visual_field_coverage()

        assert coverage["total_electrodes"] == 150
        assert "width_degrees" in coverage
        assert "height_degrees" in coverage

    def test_streaming_control(self):
        """Test streaming start/stop."""
        implant = RetinalImplant()
        implant.establish_wireless_link(15.0)
        implant.activate()

        assert implant.start_streaming()
        assert implant._is_streaming

        assert implant.stop_streaming()
        assert not implant._is_streaming

    def test_comprehensive_status(self):
        """Test comprehensive status report."""
        implant = RetinalImplant()
        implant.establish_wireless_link(15.0)
        implant.activate()

        status = implant.get_comprehensive_status()

        assert "device_state" in status
        assert "visual_field" in status
        assert "wireless_link" in status
        assert "biocompatibility" in status

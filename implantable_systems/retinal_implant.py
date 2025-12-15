"""
Retinal Implant Module

Implements retinal prosthesis systems for vision restoration including:
- Microelectrode arrays (epiretinal, subretinal, suprachoroidal)
- Image processing units
- Wireless power and data transfer
- Phosphene generation models
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable
import math

from .base import (
    ImplantableDevice,
    PowerSystem,
    Controller,
    Electrode,
    ElectrodeConfiguration,
    PhysiologicalState,
    MaterialProperties,
    DeviceState,
)


class ImplantLocation(Enum):
    """Location of retinal electrode array."""
    EPIRETINAL = "epiretinal"         # On inner retinal surface
    SUBRETINAL = "subretinal"         # Between retina and RPE
    SUPRACHOROIDAL = "suprachoroidal"  # Behind the choroid


class StimulationStrategy(Enum):
    """Stimulation encoding strategy."""
    DIRECT_INTENSITY = "direct_intensity"      # Current proportional to brightness
    PULSE_RATE_CODING = "pulse_rate"           # Frequency encodes brightness
    PULSE_WIDTH_CODING = "pulse_width"         # Duration encodes brightness
    CHARGE_BALANCED_BIPHASIC = "biphasic"      # Standard safe stimulation


@dataclass
class PhospheneModel:
    """
    Model of perceived phosphene (light perception) characteristics.

    Phosphenes are the visual perceptions produced by electrical stimulation.
    """
    brightness: float = 0.5         # 0-1 perceived brightness
    size_degrees: float = 1.0       # Apparent size in visual degrees
    color: str = "white"            # Perceived color (usually achromatic)
    persistence_ms: float = 50.0    # Duration after stimulation
    threshold_ua: float = 50.0      # Minimum current for perception

    @classmethod
    def from_stimulation(
        cls,
        current_ua: float,
        pulse_width_us: float,
        frequency_hz: float,
        electrode_size_um: float,
    ) -> "PhospheneModel":
        """
        Generate phosphene model from stimulation parameters.

        Based on psychophysical studies in retinal implant patients.
        """
        # Threshold depends on electrode size and location
        threshold = 30.0 + electrode_size_um * 0.1

        if current_ua < threshold:
            return cls(brightness=0.0, size_degrees=0.0)

        # Brightness saturates logarithmically
        brightness = min(1.0, math.log10(current_ua / threshold + 1) / 2)

        # Size increases with current (current spread)
        base_size = electrode_size_um / 100  # Approximate visual degrees
        size = base_size * (1 + (current_ua / threshold - 1) * 0.3)

        # Persistence depends on pulse parameters
        persistence = 30.0 + pulse_width_us * 0.05 + (1000 / frequency_hz) * 0.1

        return cls(
            brightness=brightness,
            size_degrees=size,
            color="white",
            persistence_ms=persistence,
            threshold_ua=threshold,
        )


@dataclass
class ImageProcessingParams:
    """Parameters for image to stimulation conversion."""
    resolution_x: int = 60          # Horizontal pixel count
    resolution_y: int = 60          # Vertical pixel count
    frame_rate_hz: float = 30.0     # Processing frame rate
    edge_enhancement: float = 1.0   # Edge enhancement factor
    contrast_enhancement: float = 1.5
    brightness_offset: float = 0.0
    downsample_method: str = "gaussian"  # gaussian, nearest, bilinear


class MicroelectrodeArray(Electrode):
    """
    Microelectrode array for retinal stimulation.

    High-density arrays with hundreds to thousands of electrodes
    for spatial resolution in visual prostheses.
    """

    def __init__(
        self,
        rows: int = 10,
        cols: int = 10,
        electrode_diameter_um: float = 200.0,
        pitch_um: float = 400.0,
        location: ImplantLocation = ImplantLocation.EPIRETINAL,
    ):
        config = ElectrodeConfiguration(
            num_electrodes=rows * cols,
            electrode_diameter_um=electrode_diameter_um,
            inter_electrode_spacing_um=pitch_um - electrode_diameter_um,
            material=MaterialProperties.platinum_iridium(),
            impedance_range_ohms=(1000, 50000),
        )
        super().__init__(config)

        self.rows = rows
        self.cols = cols
        self.pitch_um = pitch_um
        self.location = location
        self._stimulation_map: dict[int, tuple[float, float]] = {}  # electrode -> (current, width)

        # Calculate array dimensions
        self.array_width_mm = cols * pitch_um / 1000
        self.array_height_mm = rows * pitch_um / 1000

    def get_electrode_position(self, index: int) -> tuple[float, float]:
        """Get position of electrode in array coordinates (mm)."""
        if index >= self.config.num_electrodes:
            raise IndexError(f"Electrode {index} out of range")

        row = index // self.cols
        col = index % self.cols

        x = col * self.pitch_um / 1000
        y = row * self.pitch_um / 1000

        return (x, y)

    def map_to_visual_field(
        self, electrode_index: int, deg_per_mm: float = 3.5
    ) -> tuple[float, float]:
        """
        Map electrode position to visual field coordinates.

        Args:
            electrode_index: Index of electrode
            deg_per_mm: Visual degrees per mm on retina (varies with eccentricity)

        Returns:
            (x, y) position in visual degrees from fovea
        """
        x_mm, y_mm = self.get_electrode_position(electrode_index)

        # Center the array
        x_centered = x_mm - self.array_width_mm / 2
        y_centered = y_mm - self.array_height_mm / 2

        # Convert to visual degrees
        x_deg = x_centered * deg_per_mm
        y_deg = y_centered * deg_per_mm

        return (x_deg, y_deg)

    def set_stimulation_pattern(
        self, pattern: list[tuple[int, float, float]]
    ) -> None:
        """
        Set stimulation pattern for multiple electrodes.

        Args:
            pattern: List of (electrode_index, current_ua, pulse_width_us)
        """
        self._stimulation_map.clear()
        for electrode, current, width in pattern:
            if electrode < self.config.num_electrodes:
                if self.is_charge_density_safe(current, width):
                    self._stimulation_map[electrode] = (current, width)

    def generate_phosphene_map(self) -> list[PhospheneModel]:
        """Generate phosphene predictions for current stimulation pattern."""
        phosphenes = []

        for electrode, (current, width) in self._stimulation_map.items():
            if not self._is_active[electrode]:
                continue

            phosphene = PhospheneModel.from_stimulation(
                current_ua=current,
                pulse_width_us=width,
                frequency_hz=50.0,  # Default frequency
                electrode_size_um=self.config.electrode_diameter_um,
            )
            phosphenes.append(phosphene)

        return phosphenes

    def calibrate_electrode(
        self,
        electrode_index: int,
        threshold_ua: float,
        max_comfortable_ua: float,
    ) -> None:
        """
        Store calibration data for an electrode.

        Calibration determines perception threshold and comfort limits.
        """
        # This would be stored per-electrode in a real system
        pass

    def get_array_coverage_deg(self) -> tuple[float, float]:
        """Get visual field coverage of the array in degrees."""
        deg_per_mm = 3.5  # Approximate central retina
        width_deg = self.array_width_mm * deg_per_mm
        height_deg = self.array_height_mm * deg_per_mm
        return (width_deg, height_deg)


class ImageProcessor:
    """
    Image processing unit for retinal implant.

    Converts camera images to electrode stimulation patterns.
    """

    def __init__(
        self,
        electrode_array: MicroelectrodeArray,
        params: ImageProcessingParams | None = None,
    ):
        self.electrode_array = electrode_array
        self.params = params or ImageProcessingParams(
            resolution_x=electrode_array.cols,
            resolution_y=electrode_array.rows,
        )
        self._current_frame: list[list[float]] | None = None
        self._strategy = StimulationStrategy.CHARGE_BALANCED_BIPHASIC

    def process_frame(
        self, image: list[list[float]]
    ) -> list[tuple[int, float, float]]:
        """
        Process a grayscale image frame into stimulation pattern.

        Args:
            image: 2D array of pixel values (0-1)

        Returns:
            Stimulation pattern as list of (electrode, current_ua, pulse_width_us)
        """
        # Downsample to electrode resolution
        downsampled = self._downsample(image)

        # Apply image enhancements
        enhanced = self._enhance_image(downsampled)

        # Convert to stimulation parameters
        pattern = []
        for row in range(self.electrode_array.rows):
            for col in range(self.electrode_array.cols):
                electrode_idx = row * self.electrode_array.cols + col
                pixel_value = enhanced[row][col]

                if pixel_value > 0.05:  # Threshold for stimulation
                    current, width = self._pixel_to_stimulation(pixel_value)
                    pattern.append((electrode_idx, current, width))

        self._current_frame = enhanced
        return pattern

    def _downsample(self, image: list[list[float]]) -> list[list[float]]:
        """Downsample image to electrode array resolution."""
        input_rows = len(image)
        input_cols = len(image[0]) if image else 0

        if input_rows == 0 or input_cols == 0:
            return [[0.0] * self.electrode_array.cols
                    for _ in range(self.electrode_array.rows)]

        output = []
        for out_row in range(self.electrode_array.rows):
            row_data = []
            for out_col in range(self.electrode_array.cols):
                # Calculate corresponding input region
                in_row_start = int(out_row * input_rows / self.electrode_array.rows)
                in_row_end = int((out_row + 1) * input_rows / self.electrode_array.rows)
                in_col_start = int(out_col * input_cols / self.electrode_array.cols)
                in_col_end = int((out_col + 1) * input_cols / self.electrode_array.cols)

                # Average the region
                total = 0.0
                count = 0
                for r in range(in_row_start, in_row_end):
                    for c in range(in_col_start, in_col_end):
                        total += image[r][c]
                        count += 1

                row_data.append(total / max(count, 1))
            output.append(row_data)

        return output

    def _enhance_image(
        self, image: list[list[float]]
    ) -> list[list[float]]:
        """Apply contrast and edge enhancement."""
        rows = len(image)
        cols = len(image[0]) if image else 0

        # Apply contrast enhancement
        enhanced = []
        for row in range(rows):
            row_data = []
            for col in range(cols):
                value = image[row][col]

                # Contrast enhancement (S-curve)
                value = (value - 0.5) * self.params.contrast_enhancement + 0.5

                # Brightness adjustment
                value += self.params.brightness_offset

                # Clamp to valid range
                value = max(0.0, min(1.0, value))
                row_data.append(value)
            enhanced.append(row_data)

        # Edge enhancement using simple Laplacian
        if self.params.edge_enhancement > 0 and rows > 2 and cols > 2:
            edge_enhanced = [row[:] for row in enhanced]
            for row in range(1, rows - 1):
                for col in range(1, cols - 1):
                    laplacian = (
                        enhanced[row-1][col] + enhanced[row+1][col] +
                        enhanced[row][col-1] + enhanced[row][col+1] -
                        4 * enhanced[row][col]
                    )
                    edge_enhanced[row][col] = enhanced[row][col] - \
                        laplacian * self.params.edge_enhancement * 0.25
                    edge_enhanced[row][col] = max(0.0, min(1.0, edge_enhanced[row][col]))
            return edge_enhanced

        return enhanced

    def _pixel_to_stimulation(
        self, pixel_value: float
    ) -> tuple[float, float]:
        """
        Convert pixel brightness to stimulation parameters.

        Returns (current_ua, pulse_width_us)
        """
        # Base stimulation parameters
        min_current = 50.0    # uA
        max_current = 300.0   # uA
        min_width = 100.0     # us
        max_width = 500.0     # us

        if self._strategy == StimulationStrategy.DIRECT_INTENSITY:
            # Current proportional to brightness
            current = min_current + pixel_value * (max_current - min_current)
            width = 200.0  # Fixed pulse width

        elif self._strategy == StimulationStrategy.PULSE_WIDTH_CODING:
            # Width proportional to brightness
            current = 150.0  # Fixed current
            width = min_width + pixel_value * (max_width - min_width)

        elif self._strategy == StimulationStrategy.PULSE_RATE_CODING:
            # For rate coding, we return fixed parameters
            # Rate is handled by frame rate adjustment
            current = 150.0
            width = 200.0

        else:  # CHARGE_BALANCED_BIPHASIC
            # Balanced current/width for charge density control
            current = min_current + pixel_value * (max_current - min_current) * 0.7
            width = min_width + pixel_value * (max_width - min_width) * 0.5

        return (current, width)

    def set_strategy(self, strategy: StimulationStrategy) -> None:
        """Set the stimulation encoding strategy."""
        self._strategy = strategy


class WirelessPowerDataLink(PowerSystem):
    """
    Wireless power and data transfer system for retinal implants.

    Uses inductive coupling for power and RF for bidirectional data.
    """

    def __init__(
        self,
        max_power_mw: float = 50.0,
        data_rate_kbps: float = 2000.0,
        carrier_frequency_mhz: float = 13.56,
    ):
        super().__init__(
            capacity_wh=0.01,  # Small buffer capacitor
            nominal_voltage=3.3,
            max_discharge_rate=max_power_mw / 1000,
        )
        self.max_power_mw = max_power_mw
        self.data_rate_kbps = data_rate_kbps
        self.carrier_frequency_mhz = carrier_frequency_mhz
        self._link_quality = 1.0  # 0-1
        self._is_linked = False
        self._power_received_mw = 0.0

    def establish_link(self, distance_mm: float = 20.0) -> bool:
        """
        Establish wireless link with external unit.

        Args:
            distance_mm: Distance to external coil

        Returns:
            True if link established successfully
        """
        # Link quality decreases with distance
        # Typical inductive link works up to ~40mm
        if distance_mm > 40:
            self._is_linked = False
            self._link_quality = 0.0
            return False

        # Quality based on distance
        self._link_quality = max(0.0, 1.0 - (distance_mm / 50.0) ** 2)
        self._is_linked = self._link_quality > 0.3

        # Power received depends on link quality
        self._power_received_mw = self.max_power_mw * self._link_quality

        return self._is_linked

    def charge(self, power_watts: float, duration_hours: float) -> float:
        """Charge internal buffer (continuous wireless power)."""
        if not self._is_linked:
            return self.charge_level

        # Power limited by link quality
        effective_power = min(power_watts, self._power_received_mw / 1000)
        energy = effective_power * duration_hours

        self._current_charge_wh = min(self.capacity_wh, self._current_charge_wh + energy)
        return self.charge_level

    def discharge(self, power_watts: float, duration_hours: float) -> bool:
        """Discharge for stimulation power."""
        # For retinal implants, power is typically supplied continuously
        if not self._is_linked:
            # Use buffer
            energy = power_watts * duration_hours
            if energy > self._current_charge_wh:
                return False
            self._current_charge_wh -= energy
        return True

    def transmit_data(self, data_bits: int) -> float:
        """
        Transmit data and return transmission time.

        Args:
            data_bits: Number of bits to transmit

        Returns:
            Transmission time in milliseconds
        """
        if not self._is_linked:
            return -1.0  # Transmission failed

        effective_rate = self.data_rate_kbps * self._link_quality
        time_ms = data_bits / effective_rate

        return time_ms

    def get_status(self) -> dict:
        """Get link status."""
        return {
            "is_linked": self._is_linked,
            "link_quality": self._link_quality,
            "power_received_mw": self._power_received_mw,
            "data_rate_effective_kbps": self.data_rate_kbps * self._link_quality,
            "buffer_charge_percent": self.charge_level,
        }


class RetinalImplant(ImplantableDevice):
    """
    Complete retinal implant system for vision restoration.

    Integrates microelectrode array, image processor, and wireless systems.
    """

    def __init__(
        self,
        name: str = "Retinal Prosthesis",
        manufacturer: str = "Generic",
        model: str = "RET-100",
        array_rows: int = 10,
        array_cols: int = 10,
        electrode_size_um: float = 200.0,
        location: ImplantLocation = ImplantLocation.EPIRETINAL,
    ):
        super().__init__(
            name=name,
            manufacturer=manufacturer,
            model=model,
            materials=[
                MaterialProperties.platinum_iridium(),
                MaterialProperties.silicone_medical(),
            ]
        )

        # Initialize subsystems
        self._electrode_array = MicroelectrodeArray(
            rows=array_rows,
            cols=array_cols,
            electrode_diameter_um=electrode_size_um,
            location=location,
        )
        self._image_processor = ImageProcessor(self._electrode_array)
        self._power_system = WirelessPowerDataLink()

        # Operating parameters
        self._frame_rate_hz = 30.0
        self._is_streaming = False
        self._frames_processed = 0

    def activate(self) -> bool:
        """Activate the retinal implant system."""
        if self._state == DeviceState.OFF:
            self.state = DeviceState.STANDBY

        # Check wireless link
        if not self._power_system._is_linked:
            self.log_error("Cannot activate: wireless link not established")
            return False

        # Perform self-test
        tests = self.perform_self_test()
        if not all(tests.values()):
            self.log_error(f"Self-test failed: {tests}")
            return False

        self.state = DeviceState.ACTIVE
        return True

    def deactivate(self) -> bool:
        """Deactivate the retinal implant."""
        self._is_streaming = False
        self._electrode_array._stimulation_map.clear()
        self.state = DeviceState.STANDBY
        return True

    def perform_self_test(self) -> dict[str, bool]:
        """Perform system self-test."""
        tests = {
            "wireless_link": self._power_system._is_linked,
            "power_sufficient": self._power_system._power_received_mw > 10,
            "electrode_array": True,
            "image_processor": True,
        }

        # Check electrode impedances
        impedances = self._electrode_array.measure_all_impedances()
        low_impedance = sum(1 for z in impedances if z < self._electrode_array.config.impedance_range_ohms[0])
        high_impedance = sum(1 for z in impedances if z > self._electrode_array.config.impedance_range_ohms[1])

        tests["electrode_impedances"] = (low_impedance + high_impedance) < len(impedances) * 0.1

        return tests

    def establish_wireless_link(self, distance_mm: float = 20.0) -> bool:
        """Establish wireless connection with external camera unit."""
        return self._power_system.establish_link(distance_mm)

    def process_frame(self, image: list[list[float]]) -> dict:
        """
        Process a single image frame.

        Args:
            image: Grayscale image as 2D array (0-1 values)

        Returns:
            Processing results including predicted phosphenes
        """
        if self._state != DeviceState.ACTIVE:
            return {"status": "inactive"}

        # Process image to stimulation pattern
        pattern = self._image_processor.process_frame(image)

        # Apply to electrode array
        self._electrode_array.set_stimulation_pattern(pattern)

        # Generate phosphene predictions
        phosphenes = self._electrode_array.generate_phosphene_map()

        self._frames_processed += 1

        # Calculate power consumption
        total_charge = sum(
            current * width * 1e-6  # Convert to nC
            for _, current, width in pattern
        )
        power_uw = total_charge * self._frame_rate_hz * 3.3 * 1000  # Approximate power

        telemetry = {
            "active_electrodes": len(pattern),
            "total_electrodes": self._electrode_array.config.num_electrodes,
            "phosphenes_generated": len(phosphenes),
            "avg_brightness": sum(p.brightness for p in phosphenes) / max(len(phosphenes), 1),
            "power_consumption_uw": power_uw,
            "frames_processed": self._frames_processed,
        }
        self.record_telemetry(telemetry)

        return telemetry

    def start_streaming(self) -> bool:
        """Start continuous frame processing."""
        if self._state != DeviceState.ACTIVE:
            return False
        self._is_streaming = True
        return True

    def stop_streaming(self) -> bool:
        """Stop continuous frame processing."""
        self._is_streaming = False
        return True

    def calibrate_electrode(
        self,
        electrode_index: int,
        threshold_ua: float,
        max_ua: float,
    ) -> bool:
        """
        Calibrate a single electrode.

        Args:
            electrode_index: Index of electrode to calibrate
            threshold_ua: Perception threshold current
            max_ua: Maximum comfortable current

        Returns:
            True if calibration successful
        """
        if electrode_index >= self._electrode_array.config.num_electrodes:
            return False

        # Store calibration (simplified)
        self._electrode_array.calibrate_electrode(electrode_index, threshold_ua, max_ua)
        return True

    def get_visual_field_coverage(self) -> dict:
        """Get information about visual field coverage."""
        width_deg, height_deg = self._electrode_array.get_array_coverage_deg()
        total_electrodes = self._electrode_array.config.num_electrodes
        active_electrodes = self._electrode_array.active_electrode_count

        return {
            "width_degrees": width_deg,
            "height_degrees": height_deg,
            "total_electrodes": total_electrodes,
            "active_electrodes": active_electrodes,
            "electrode_resolution": f"{self._electrode_array.cols}x{self._electrode_array.rows}",
            "implant_location": self._electrode_array.location.value,
        }

    def get_comprehensive_status(self) -> dict:
        """Get comprehensive system status."""
        return {
            "device_state": self._state.value,
            "visual_field": self.get_visual_field_coverage(),
            "wireless_link": self._power_system.get_status(),
            "biocompatibility": self.get_biocompatibility_assessment(),
            "is_streaming": self._is_streaming,
            "frames_processed": self._frames_processed,
            "error_count": len(self._error_log),
        }

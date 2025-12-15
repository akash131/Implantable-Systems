"""
Data Management and Telemetry for Implantable Devices.

Provides:
- Telemetry data recording and storage
- Trend analysis and statistics
- Data export (CSV, JSON, HL7 FHIR)
- Remote monitoring integration
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Iterator, Callable
from pathlib import Path
import json
import csv
import io
import statistics


class DataResolution(Enum):
    """Time resolution for data storage."""
    RAW = "raw"              # Every sample
    SECOND = "second"        # 1 Hz
    MINUTE = "minute"        # Once per minute
    HOURLY = "hourly"        # Hourly aggregates
    DAILY = "daily"          # Daily summaries


class ExportFormat(Enum):
    """Supported export formats."""
    CSV = "csv"
    JSON = "json"
    HL7_FHIR = "fhir"


@dataclass
class TelemetryPoint:
    """Single telemetry data point."""
    timestamp: str
    parameter: str
    value: float
    unit: str
    device_serial: str
    quality: str = "good"  # good, interpolated, suspect, missing

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "parameter": self.parameter,
            "value": self.value,
            "unit": self.unit,
            "device_serial": self.device_serial,
            "quality": self.quality,
        }


@dataclass
class TelemetryWindow:
    """Aggregated telemetry over a time window."""
    start_time: str
    end_time: str
    parameter: str
    unit: str
    device_serial: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    std_dev: float
    median_value: float

    @classmethod
    def from_points(cls, points: list[TelemetryPoint]) -> "TelemetryWindow":
        """Create window summary from raw points."""
        if not points:
            raise ValueError("Cannot create window from empty points")

        values = [p.value for p in points]
        first = points[0]
        last = points[-1]

        return cls(
            start_time=first.timestamp,
            end_time=last.timestamp,
            parameter=first.parameter,
            unit=first.unit,
            device_serial=first.device_serial,
            count=len(values),
            min_value=min(values),
            max_value=max(values),
            mean_value=statistics.mean(values),
            std_dev=statistics.stdev(values) if len(values) > 1 else 0.0,
            median_value=statistics.median(values),
        )


class TelemetryBuffer:
    """
    Circular buffer for telemetry data.

    Efficiently stores recent telemetry with automatic aging.
    """

    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self._data: list[TelemetryPoint] = []
        self._index: dict[str, list[int]] = {}  # parameter -> indices

    def add(self, point: TelemetryPoint) -> None:
        """Add a telemetry point."""
        if len(self._data) >= self.max_size:
            # Remove oldest 10%
            remove_count = self.max_size // 10
            self._data = self._data[remove_count:]
            self._rebuild_index()

        idx = len(self._data)
        self._data.append(point)

        if point.parameter not in self._index:
            self._index[point.parameter] = []
        self._index[point.parameter].append(idx)

    def _rebuild_index(self) -> None:
        """Rebuild parameter index after cleanup."""
        self._index.clear()
        for i, point in enumerate(self._data):
            if point.parameter not in self._index:
                self._index[point.parameter] = []
            self._index[point.parameter].append(i)

    def get_parameter(
        self,
        parameter: str,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> list[TelemetryPoint]:
        """Get telemetry for a specific parameter."""
        if parameter not in self._index:
            return []

        points = [self._data[i] for i in self._index[parameter]]

        if start_time:
            points = [p for p in points if p.timestamp >= start_time]

        if end_time:
            points = [p for p in points if p.timestamp <= end_time]

        return points

    def get_latest(self, parameter: str) -> TelemetryPoint | None:
        """Get most recent value for a parameter."""
        if parameter not in self._index or not self._index[parameter]:
            return None
        return self._data[self._index[parameter][-1]]

    def get_all_parameters(self) -> list[str]:
        """Get list of all recorded parameters."""
        return list(self._index.keys())

    @property
    def size(self) -> int:
        """Current buffer size."""
        return len(self._data)


class TrendAnalyzer:
    """
    Analyzes telemetry trends over time.

    Detects patterns, anomalies, and significant changes.
    """

    def __init__(self, buffer: TelemetryBuffer):
        self._buffer = buffer

    def calculate_trend(
        self,
        parameter: str,
        window_hours: float = 24.0,
    ) -> dict:
        """
        Calculate trend for a parameter over a time window.

        Returns:
            Dictionary with trend direction, slope, and statistics.
        """
        end_time = datetime.now().isoformat()
        start_time = (datetime.now() - timedelta(hours=window_hours)).isoformat()

        points = self._buffer.get_parameter(parameter, start_time, end_time)

        if len(points) < 2:
            return {
                "trend": "insufficient_data",
                "points": len(points),
            }

        values = [p.value for p in points]

        # Calculate linear regression slope
        n = len(values)
        x = list(range(n))
        x_mean = sum(x) / n
        y_mean = sum(values) / n

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0
        else:
            slope = numerator / denominator

        # Determine trend direction
        if abs(slope) < 0.01 * y_mean:  # Less than 1% change
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"

        # Calculate percentage change
        if values[0] != 0:
            pct_change = ((values[-1] - values[0]) / values[0]) * 100
        else:
            pct_change = 0

        return {
            "trend": trend,
            "slope": slope,
            "start_value": values[0],
            "end_value": values[-1],
            "percent_change": pct_change,
            "min": min(values),
            "max": max(values),
            "mean": statistics.mean(values),
            "std_dev": statistics.stdev(values) if n > 1 else 0,
            "points": n,
            "window_hours": window_hours,
        }

    def detect_anomalies(
        self,
        parameter: str,
        sigma_threshold: float = 3.0,
    ) -> list[TelemetryPoint]:
        """
        Detect anomalous values using z-score method.

        Args:
            parameter: Parameter to analyze
            sigma_threshold: Number of standard deviations for anomaly

        Returns:
            List of anomalous telemetry points
        """
        points = self._buffer.get_parameter(parameter)

        if len(points) < 10:
            return []

        values = [p.value for p in points]
        mean = statistics.mean(values)
        std = statistics.stdev(values)

        if std == 0:
            return []

        anomalies = []
        for point in points:
            z_score = abs((point.value - mean) / std)
            if z_score > sigma_threshold:
                anomalies.append(point)

        return anomalies

    def compare_periods(
        self,
        parameter: str,
        period1_start: str,
        period1_end: str,
        period2_start: str,
        period2_end: str,
    ) -> dict:
        """Compare statistics between two time periods."""
        points1 = self._buffer.get_parameter(parameter, period1_start, period1_end)
        points2 = self._buffer.get_parameter(parameter, period2_start, period2_end)

        if not points1 or not points2:
            return {"error": "Insufficient data for comparison"}

        values1 = [p.value for p in points1]
        values2 = [p.value for p in points2]

        mean1, mean2 = statistics.mean(values1), statistics.mean(values2)
        std1 = statistics.stdev(values1) if len(values1) > 1 else 0
        std2 = statistics.stdev(values2) if len(values2) > 1 else 0

        return {
            "period1": {
                "start": period1_start,
                "end": period1_end,
                "count": len(values1),
                "mean": mean1,
                "std": std1,
            },
            "period2": {
                "start": period2_start,
                "end": period2_end,
                "count": len(values2),
                "mean": mean2,
                "std": std2,
            },
            "difference": {
                "mean_change": mean2 - mean1,
                "percent_change": ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0,
            },
        }


class DataExporter:
    """
    Exports telemetry data to various formats.

    Supports CSV, JSON, and HL7 FHIR for interoperability.
    """

    def __init__(self, buffer: TelemetryBuffer):
        self._buffer = buffer

    def export_csv(
        self,
        parameters: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> str:
        """Export data to CSV format."""
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "timestamp", "parameter", "value", "unit", "device_serial", "quality"
        ])

        params = parameters or self._buffer.get_all_parameters()

        for param in params:
            points = self._buffer.get_parameter(param, start_time, end_time)
            for point in points:
                writer.writerow([
                    point.timestamp,
                    point.parameter,
                    point.value,
                    point.unit,
                    point.device_serial,
                    point.quality,
                ])

        return output.getvalue()

    def export_json(
        self,
        parameters: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
    ) -> str:
        """Export data to JSON format."""
        params = parameters or self._buffer.get_all_parameters()

        data = {
            "export_time": datetime.now().isoformat(),
            "parameters": {},
        }

        for param in params:
            points = self._buffer.get_parameter(param, start_time, end_time)
            data["parameters"][param] = [p.to_dict() for p in points]

        return json.dumps(data, indent=2)

    def export_fhir_observation(
        self,
        parameter: str,
        patient_id: str,
    ) -> list[dict]:
        """
        Export telemetry as HL7 FHIR Observation resources.

        Follows FHIR R4 specification for device observations.
        """
        points = self._buffer.get_parameter(parameter)

        observations = []
        for point in points:
            obs = {
                "resourceType": "Observation",
                "status": "final",
                "category": [{
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "vital-signs",
                        "display": "Vital Signs"
                    }]
                }],
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": self._get_loinc_code(parameter),
                        "display": parameter
                    }],
                    "text": parameter
                },
                "subject": {
                    "reference": f"Patient/{patient_id}"
                },
                "device": {
                    "display": point.device_serial
                },
                "effectiveDateTime": point.timestamp,
                "valueQuantity": {
                    "value": point.value,
                    "unit": point.unit,
                    "system": "http://unitsofmeasure.org",
                    "code": self._get_ucum_code(point.unit)
                },
                "dataAbsentReason": None if point.quality == "good" else {
                    "coding": [{
                        "system": "http://terminology.hl7.org/CodeSystem/data-absent-reason",
                        "code": "unknown"
                    }]
                }
            }
            observations.append(obs)

        return observations

    def _get_loinc_code(self, parameter: str) -> str:
        """Map parameter to LOINC code."""
        loinc_map = {
            "heart_rate": "8867-4",
            "cardiac_output": "8741-1",
            "blood_pressure_systolic": "8480-6",
            "blood_pressure_diastolic": "8462-4",
            "pump_speed": "LP6847-6",
            "battery_level": "LP6847-6",
        }
        return loinc_map.get(parameter.lower(), "LP6847-6")

    def _get_ucum_code(self, unit: str) -> str:
        """Map unit to UCUM code."""
        ucum_map = {
            "bpm": "/min",
            "rpm": "/min",
            "L/min": "L/min",
            "mmHg": "mm[Hg]",
            "%": "%",
            "W": "W",
            "mA": "mA",
            "us": "us",
        }
        return ucum_map.get(unit, unit)


class RemoteMonitoringHub:
    """
    Hub for remote patient monitoring.

    Aggregates data from multiple devices and provides alerting.
    """

    def __init__(self):
        self._devices: dict[str, TelemetryBuffer] = {}
        self._alert_rules: list[dict] = []
        self._alert_callbacks: list[Callable] = []

    def register_device(self, device_serial: str) -> TelemetryBuffer:
        """Register a device for monitoring."""
        if device_serial not in self._devices:
            self._devices[device_serial] = TelemetryBuffer()
        return self._devices[device_serial]

    def record_telemetry(
        self,
        device_serial: str,
        parameter: str,
        value: float,
        unit: str,
    ) -> None:
        """Record telemetry from a device."""
        if device_serial not in self._devices:
            self.register_device(device_serial)

        point = TelemetryPoint(
            timestamp=datetime.now().isoformat(),
            parameter=parameter,
            value=value,
            unit=unit,
            device_serial=device_serial,
        )

        self._devices[device_serial].add(point)
        self._check_alerts(point)

    def add_alert_rule(
        self,
        parameter: str,
        condition: str,  # "above", "below", "outside_range"
        threshold: float | tuple[float, float],
        severity: str = "warning",
    ) -> None:
        """Add an alerting rule."""
        self._alert_rules.append({
            "parameter": parameter,
            "condition": condition,
            "threshold": threshold,
            "severity": severity,
        })

    def on_alert(self, callback: Callable) -> None:
        """Register alert callback."""
        self._alert_callbacks.append(callback)

    def _check_alerts(self, point: TelemetryPoint) -> None:
        """Check if point triggers any alerts."""
        for rule in self._alert_rules:
            if rule["parameter"] != point.parameter:
                continue

            triggered = False
            threshold = rule["threshold"]

            if rule["condition"] == "above" and point.value > threshold:
                triggered = True
            elif rule["condition"] == "below" and point.value < threshold:
                triggered = True
            elif rule["condition"] == "outside_range":
                if isinstance(threshold, tuple):
                    if point.value < threshold[0] or point.value > threshold[1]:
                        triggered = True

            if triggered:
                alert = {
                    "timestamp": datetime.now().isoformat(),
                    "device_serial": point.device_serial,
                    "parameter": point.parameter,
                    "value": point.value,
                    "rule": rule,
                    "severity": rule["severity"],
                }

                for callback in self._alert_callbacks:
                    callback(alert)

    def get_device_summary(self, device_serial: str) -> dict:
        """Get summary of device telemetry."""
        if device_serial not in self._devices:
            return {"error": "Device not registered"}

        buffer = self._devices[device_serial]
        params = buffer.get_all_parameters()

        summary = {
            "device_serial": device_serial,
            "parameters": {},
            "last_contact": None,
        }

        for param in params:
            latest = buffer.get_latest(param)
            if latest:
                summary["parameters"][param] = {
                    "latest_value": latest.value,
                    "unit": latest.unit,
                    "timestamp": latest.timestamp,
                }
                if not summary["last_contact"] or latest.timestamp > summary["last_contact"]:
                    summary["last_contact"] = latest.timestamp

        return summary

    def get_all_devices(self) -> list[str]:
        """Get list of all registered devices."""
        return list(self._devices.keys())


class DataLogger:
    """
    Persistent data logger for device telemetry.

    Writes data to files with rotation and compression.
    """

    def __init__(self, log_dir: str = "./device_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_files: dict[str, Path] = {}

    def log(
        self,
        device_serial: str,
        data: dict,
    ) -> None:
        """Log data for a device."""
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{device_serial}_{date_str}.jsonl"
        filepath = self.log_dir / filename

        record = {
            "timestamp": datetime.now().isoformat(),
            **data,
        }

        with open(filepath, "a") as f:
            f.write(json.dumps(record) + "\n")

        self._current_files[device_serial] = filepath

    def read_logs(
        self,
        device_serial: str,
        date: str | None = None,
    ) -> Iterator[dict]:
        """Read logs for a device."""
        pattern = f"{device_serial}_*.jsonl"
        if date:
            pattern = f"{device_serial}_{date}.jsonl"

        for filepath in self.log_dir.glob(pattern):
            with open(filepath) as f:
                for line in f:
                    yield json.loads(line)

    def get_log_files(self, device_serial: str | None = None) -> list[Path]:
        """Get list of log files."""
        if device_serial:
            return list(self.log_dir.glob(f"{device_serial}_*.jsonl"))
        return list(self.log_dir.glob("*.jsonl"))

    def cleanup_old_logs(self, days: int = 90) -> int:
        """Remove logs older than specified days."""
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0

        for filepath in self.log_dir.glob("*.jsonl"):
            # Extract date from filename
            parts = filepath.stem.split("_")
            if len(parts) >= 2:
                date_str = parts[-1]
                try:
                    file_date = datetime.strptime(date_str, "%Y%m%d")
                    if file_date < cutoff:
                        filepath.unlink()
                        removed += 1
                except ValueError:
                    pass

        return removed

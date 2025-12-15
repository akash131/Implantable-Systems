"""Tests for advanced features: vendors, clinical, data management, safety, simulation, communication."""

import pytest
from datetime import datetime


class TestVendorProfiles:
    """Tests for vendor device profiles."""

    def test_device_registry_list_all(self):
        """Test listing all available devices."""
        from implantable_systems.vendors import DeviceRegistry

        all_devices = DeviceRegistry.list_all_devices()
        assert "lvad" in all_devices
        assert "dbs" in all_devices
        assert len(all_devices["lvad"]) > 0

    def test_create_lvad_from_profile(self):
        """Test creating LVAD from profile."""
        from implantable_systems.vendors import (
            LVADProfiles,
            create_lvad_from_profile,
        )

        lvad = create_lvad_from_profile(LVADProfiles.HEARTMATE_3)
        assert lvad.name == "HeartMate 3"
        assert lvad.manufacturer == "Abbott"

    def test_create_dbs_from_profile(self):
        """Test creating DBS from profile."""
        from implantable_systems.vendors import (
            DBSProfiles,
            create_dbs_from_profile,
        )
        from implantable_systems.deep_brain_stimulator import BrainTarget

        dbs = create_dbs_from_profile(
            DBSProfiles.MEDTRONIC_PERCEPT,
            target=BrainTarget.STN,
        )
        assert "Percept" in dbs.name
        assert dbs.target == BrainTarget.STN

    def test_device_registry_create(self):
        """Test creating device by name."""
        from implantable_systems.vendors import DeviceRegistry

        device = DeviceRegistry.create_device("HeartMate 3")
        assert device is not None


class TestClinicalProgramming:
    """Tests for clinical programming interface."""

    def test_patient_creation(self):
        """Test patient data structure."""
        from implantable_systems.clinical import Patient

        patient = Patient(
            id="PT001",
            mrn="12345",
            name="Test Patient",
            date_of_birth="1970-01-01",
            implant_date="2024-01-01",
            device_serial="TEST-001",
            indication="Heart Failure",
        )
        assert patient.id == "PT001"
        assert patient.time_since_implant.days >= 0

    def test_clinician_permissions(self):
        """Test clinician permission checking."""
        from implantable_systems.clinical import Clinician

        clinician = Clinician(
            id="DR001",
            name="Dr. Test",
            credentials="MD",
            institution="Test Hospital",
            permissions=["read", "program"],
        )
        assert clinician.has_permission("read")
        assert clinician.has_permission("program")
        assert not clinician.has_permission("admin")

    def test_programming_session(self):
        """Test programming session workflow."""
        from implantable_systems.clinical import (
            Patient,
            Clinician,
            LVADProgrammer,
            ProgrammingMode,
        )
        from implantable_systems.lvad import LVAD

        lvad = LVAD()
        programmer = LVADProgrammer(lvad)

        patient = Patient(
            id="PT001", mrn="12345", name="Test",
            date_of_birth="1970-01-01", implant_date="2024-01-01",
            device_serial="TEST-001", indication="HF",
        )
        clinician = Clinician(
            id="DR001", name="Dr. Test", credentials="MD",
            institution="Hospital",
        )

        session = programmer.start_session(
            patient=patient,
            clinician=clinician,
            device_serial="TEST-001",
            mode=ProgrammingMode.READ_ONLY,
        )
        assert session is not None
        assert programmer.is_session_active

        report = programmer.end_session("Test complete")
        assert report["total_changes"] == 0
        assert not programmer.is_session_active

    def test_safety_validator(self):
        """Test parameter safety validation."""
        from implantable_systems.clinical import SafetyValidator

        validator = SafetyValidator()

        # Test within limits
        is_valid, msg = validator.validate_value("dbs_amplitude_ma", 3.0)
        assert is_valid

        # Test exceeds limits
        is_valid, msg = validator.validate_value("dbs_amplitude_ma", 15.0)
        assert not is_valid


class TestDataManagement:
    """Tests for data management and telemetry."""

    def test_telemetry_buffer(self):
        """Test telemetry buffer operations."""
        from implantable_systems.data_management import (
            TelemetryBuffer,
            TelemetryPoint,
        )

        buffer = TelemetryBuffer(max_size=100)

        point = TelemetryPoint(
            timestamp=datetime.now().isoformat(),
            parameter="flow_lpm",
            value=5.0,
            unit="L/min",
            device_serial="TEST-001",
        )
        buffer.add(point)

        assert buffer.size == 1
        assert "flow_lpm" in buffer.get_all_parameters()

    def test_telemetry_window(self):
        """Test telemetry window aggregation."""
        from implantable_systems.data_management import (
            TelemetryPoint,
            TelemetryWindow,
        )

        points = [
            TelemetryPoint(
                timestamp=datetime.now().isoformat(),
                parameter="flow",
                value=v,
                unit="L/min",
                device_serial="TEST",
            )
            for v in [4.0, 5.0, 6.0, 5.0, 5.5]
        ]

        window = TelemetryWindow.from_points(points)
        assert window.count == 5
        assert window.min_value == 4.0
        assert window.max_value == 6.0

    def test_remote_monitoring_hub(self):
        """Test remote monitoring hub."""
        from implantable_systems.data_management import RemoteMonitoringHub

        hub = RemoteMonitoringHub()
        hub.register_device("DEVICE-001")

        alerts = []
        hub.add_alert_rule("flow", "below", 2.0, "critical")
        hub.on_alert(lambda a: alerts.append(a))

        hub.record_telemetry("DEVICE-001", "flow", 5.0, "L/min")
        assert len(alerts) == 0

        hub.record_telemetry("DEVICE-001", "flow", 1.5, "L/min")
        assert len(alerts) == 1

    def test_data_exporter_json(self):
        """Test JSON data export."""
        from implantable_systems.data_management import (
            TelemetryBuffer,
            TelemetryPoint,
            DataExporter,
        )
        import json

        buffer = TelemetryBuffer()
        buffer.add(TelemetryPoint(
            timestamp=datetime.now().isoformat(),
            parameter="test",
            value=1.0,
            unit="unit",
            device_serial="DEV",
        ))

        exporter = DataExporter(buffer)
        json_str = exporter.export_json()
        data = json.loads(json_str)

        assert "parameters" in data
        assert "test" in data["parameters"]


class TestSafetySystem:
    """Tests for alarm and safety systems."""

    def test_alarm_creation(self):
        """Test alarm creation and states."""
        from implantable_systems.safety import (
            Alarm,
            AlarmPriority,
            AlarmCategory,
            AlarmState,
        )

        alarm = Alarm(
            alarm_id="ALM-001",
            code="TEST001",
            message="Test alarm",
            priority=AlarmPriority.HIGH,
            category=AlarmCategory.TECHNICAL,
            device_serial="DEV-001",
        )
        assert alarm.state == AlarmState.ACTIVE

        alarm.acknowledge("user")
        assert alarm.state == AlarmState.ACKNOWLEDGED

    def test_alarm_manager(self):
        """Test alarm manager functionality."""
        from implantable_systems.safety import (
            AlarmManager,
            AlarmDefinition,
            AlarmPriority,
            AlarmCategory,
        )

        manager = AlarmManager()

        definition = AlarmDefinition(
            code="TEST001",
            message_template="Value {value} exceeds {threshold}",
            priority=AlarmPriority.HIGH,
            category=AlarmCategory.TECHNICAL,
            parameter="test_param",
            condition="above",
            threshold=10.0,
        )
        manager.register_alarm(definition)

        # Check normal value - no alarm
        alarms = manager.check_condition("DEV-001", "test_param", 5.0)
        assert len(alarms) == 0

        # Check abnormal value - alarm triggered
        alarms = manager.check_condition("DEV-001", "test_param", 15.0)
        assert len(alarms) == 1
        assert alarms[0].priority == AlarmPriority.HIGH

    def test_safety_interlock(self):
        """Test safety interlock system."""
        from implantable_systems.safety import SafetyInterlock

        interlock = SafetyInterlock("TestInterlock")

        # Add condition that's always safe
        interlock.add_condition(lambda: True, "Always safe")

        is_safe, reason = interlock.check()
        assert is_safe

        # Add condition that fails
        interlock.add_condition(lambda: False, "Test failure")

        is_safe, reason = interlock.check()
        assert not is_safe
        assert reason == "Test failure"


class TestSimulation:
    """Tests for simulation framework."""

    def test_cardiovascular_model(self):
        """Test cardiovascular patient model."""
        from implantable_systems.simulation import CardiovascularModel

        model = CardiovascularModel(
            native_heart_function=0.2,
            body_surface_area=1.9,
        )

        state = model.update(dt=0.1)
        assert "heart_rate_bpm" in state
        assert "cardiac_output_lpm" in state
        assert state["cardiac_output_lpm"] > 0

    def test_neurological_model(self):
        """Test neurological patient model."""
        from implantable_systems.simulation import NeurologicalModel

        model = NeurologicalModel(
            condition="parkinsons",
            baseline_symptoms=0.7,
        )

        state = model.update(dt=0.1)
        assert "symptom_severity" in state
        assert "beta_power" in state

    def test_scenario_creation(self):
        """Test scenario creation."""
        from implantable_systems.simulation import Scenario

        scenario = Scenario.exercise_test(duration_minutes=5.0)
        assert scenario.name == "Exercise Test"

        events = scenario.get_events_at_time(0.0, tolerance=1.0)
        assert len(events) > 0

    def test_simulation_config(self):
        """Test simulation configuration."""
        from implantable_systems.simulation import SimulationConfig

        config = SimulationConfig(
            time_step_seconds=0.01,
            duration_hours=1.0,
            enable_noise=True,
        )
        assert config.time_step_seconds == 0.01
        assert config.duration_hours == 1.0


class TestCommunication:
    """Tests for communication protocols."""

    def test_ble_channel(self):
        """Test BLE communication channel."""
        from implantable_systems.communication import BLEChannel, ConnectionState

        channel = BLEChannel("DEV-001")
        assert channel.state == ConnectionState.DISCONNECTED

        assert channel.connect()
        assert channel.state == ConnectionState.CONNECTED

        channel.disconnect()
        assert channel.state == ConnectionState.DISCONNECTED

    def test_message_serialization(self):
        """Test message serialization."""
        from implantable_systems.communication import Message

        original = Message(
            message_id=1,
            command="TEST",
            payload=b"Hello",
        )
        data = original.to_bytes()
        restored = Message.from_bytes(data)

        assert restored.message_id == original.message_id
        assert restored.command == original.command

    def test_telemetry_packet(self):
        """Test telemetry packet."""
        from implantable_systems.communication import TelemetryPacket

        packet = TelemetryPacket(
            sequence_number=1,
            timestamp=123456,
            parameters={"flow": 5.0, "power": 6.0},
            battery_level=80,
            alarm_flags=0,
        )
        packet.checksum = packet.calculate_checksum()

        assert packet.verify_checksum()

        data = packet.to_bytes()
        assert len(data) > 0

    def test_device_communicator(self):
        """Test device communicator."""
        from implantable_systems.communication import (
            DeviceCommunicator,
            BLEChannel,
            ProtocolType,
        )
        from implantable_systems.lvad import LVAD

        lvad = LVAD()
        lvad.activate()

        comm = DeviceCommunicator(lvad)
        comm.add_channel(ProtocolType.BLE, BLEChannel("TEST"))

        assert comm.connect(ProtocolType.BLE)
        status = comm.get_connection_status()
        assert status["connected"]

        comm.disconnect()

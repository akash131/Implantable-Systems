#!/usr/bin/env python3
"""
Remote Monitoring Example

Demonstrates:
- Setting up remote monitoring hub
- Multiple device telemetry collection
- Alert rules and notifications
- Data export and analysis
"""

import sys
sys.path.insert(0, '..')

from implantable_systems.vendors import (
    create_lvad_from_profile,
    create_dbs_from_profile,
    LVADProfiles,
    DBSProfiles,
)
from implantable_systems.data_management import (
    RemoteMonitoringHub,
    TelemetryBuffer,
    TrendAnalyzer,
    DataExporter,
    TelemetryPoint,
)
from implantable_systems.safety import (
    AlarmManager,
    AlarmDefinition,
    AlarmPriority,
    AlarmCategory,
)
from implantable_systems.communication import (
    DeviceCommunicator,
    BLEChannel,
    ProtocolType,
)
from implantable_systems.base import PhysiologicalState


def main():
    print("=" * 60)
    print("Remote Monitoring Hub Example")
    print("=" * 60)

    # 1. Create remote monitoring hub
    print("\n1. Setting up Remote Monitoring Hub...")
    hub = RemoteMonitoringHub()

    # 2. Register multiple devices
    print("\n2. Registering devices...")

    # LVAD patient
    lvad_serial = "HM3-2024-001"
    hub.register_device(lvad_serial)
    print(f"   Registered LVAD: {lvad_serial}")

    # DBS patient
    dbs_serial = "PERCEPT-2024-001"
    hub.register_device(dbs_serial)
    print(f"   Registered DBS: {dbs_serial}")

    # 3. Set up alert rules
    print("\n3. Configuring alert rules...")

    # LVAD alerts
    hub.add_alert_rule(
        parameter="flow_lpm",
        condition="below",
        threshold=2.5,
        severity="critical",
    )
    hub.add_alert_rule(
        parameter="battery_percent",
        condition="below",
        threshold=25,
        severity="warning",
    )

    # DBS alerts
    hub.add_alert_rule(
        parameter="impedance_ohms",
        condition="above",
        threshold=3000,
        severity="warning",
    )

    alerts_received = []

    def alert_handler(alert):
        alerts_received.append(alert)
        print(f"\n   *** ALERT: {alert['severity'].upper()} ***")
        print(f"       Device: {alert['device_serial']}")
        print(f"       Parameter: {alert['parameter']} = {alert['value']}")

    hub.on_alert(alert_handler)
    print("   Alert rules configured")

    # 4. Simulate telemetry data reception
    print("\n4. Simulating telemetry reception...")

    # Create actual devices for simulation
    lvad = create_lvad_from_profile(LVADProfiles.HEARTMATE_3)
    lvad.activate()

    dbs = create_dbs_from_profile(DBSProfiles.MEDTRONIC_PERCEPT)
    dbs.activate()

    # Simulate several telemetry updates
    print("\n   Receiving telemetry data...")

    physio_states = [
        PhysiologicalState(activity_level=0.0),  # Rest
        PhysiologicalState(activity_level=0.3),  # Light activity
        PhysiologicalState(activity_level=0.6),  # Moderate activity
        PhysiologicalState(activity_level=0.2),  # Cooling down
    ]

    for i, physio in enumerate(physio_states):
        # Update LVAD
        lvad_status = lvad.update(physio, dt=0.1)

        # Record telemetry to hub
        hub.record_telemetry(
            lvad_serial,
            "flow_lpm",
            lvad_status.get("flow_lpm", 5.0),
            "L/min"
        )
        hub.record_telemetry(
            lvad_serial,
            "power_watts",
            lvad_status.get("power_watts", 6.0),
            "W"
        )
        hub.record_telemetry(
            lvad_serial,
            "battery_percent",
            lvad_status.get("battery_level", 80),
            "%"
        )

        # Update DBS
        dbs_status = dbs.update()

        hub.record_telemetry(
            dbs_serial,
            "amplitude_ma",
            dbs_status.get("right_amplitude_ma", 2.0),
            "mA"
        )
        hub.record_telemetry(
            dbs_serial,
            "battery_percent",
            dbs_status.get("battery_level", 90),
            "%"
        )

        print(f"   [{i+1}/4] Telemetry received from both devices")

    # 5. Trigger an alert condition
    print("\n5. Testing alert system...")
    # Simulate low flow condition
    hub.record_telemetry(lvad_serial, "flow_lpm", 2.0, "L/min")

    print(f"\n   Total alerts generated: {len(alerts_received)}")

    # 6. Get device summaries
    print("\n6. Device Summaries:")

    lvad_summary = hub.get_device_summary(lvad_serial)
    print(f"\n   LVAD ({lvad_serial}):")
    print(f"     Last contact: {lvad_summary.get('last_contact', 'Unknown')[:19]}")
    for param, data in lvad_summary.get("parameters", {}).items():
        print(f"     {param}: {data['latest_value']:.2f} {data['unit']}")

    dbs_summary = hub.get_device_summary(dbs_serial)
    print(f"\n   DBS ({dbs_serial}):")
    print(f"     Last contact: {dbs_summary.get('last_contact', 'Unknown')[:19]}")
    for param, data in dbs_summary.get("parameters", {}).items():
        print(f"     {param}: {data['latest_value']:.2f} {data['unit']}")

    # 7. Data export
    print("\n7. Data Export:")

    # Get telemetry buffer for LVAD
    lvad_buffer = hub._devices[lvad_serial]
    exporter = DataExporter(lvad_buffer)

    # Export to JSON
    json_data = exporter.export_json(parameters=["flow_lpm"])
    print(f"   JSON export: {len(json_data)} bytes")

    # Export to CSV
    csv_data = exporter.export_csv(parameters=["flow_lpm", "power_watts"])
    print(f"   CSV export: {len(csv_data)} bytes")

    # Export to FHIR
    fhir_observations = exporter.export_fhir_observation("flow_lpm", "PT001")
    print(f"   FHIR observations: {len(fhir_observations)} resources")

    # 8. Trend analysis
    print("\n8. Trend Analysis:")
    analyzer = TrendAnalyzer(lvad_buffer)

    flow_trend = analyzer.calculate_trend("flow_lpm", window_hours=24)
    print(f"   Flow trend: {flow_trend.get('trend', 'insufficient_data')}")

    if flow_trend.get("points", 0) >= 2:
        print(f"   Change: {flow_trend.get('percent_change', 0):.1f}%")
        print(f"   Range: {flow_trend.get('min', 0):.1f} - {flow_trend.get('max', 0):.1f} L/min")

    # 9. Communication setup demo
    print("\n9. Communication Channel Setup:")

    # Set up BLE communication
    ble_channel = BLEChannel(lvad_serial)
    communicator = DeviceCommunicator(lvad)
    communicator.add_channel(ProtocolType.BLE, ble_channel)

    print("   Connecting via Bluetooth LE...")
    if communicator.connect(ProtocolType.BLE):
        print("   Connected successfully")

        conn_status = communicator.get_connection_status()
        print(f"   Protocol: {conn_status.get('protocol')}")

        # Request telemetry
        packet = communicator.request_telemetry()
        if packet:
            print(f"   Telemetry packet received:")
            print(f"     Sequence: {packet.sequence_number}")
            print(f"     Battery: {packet.battery_level}%")
            print(f"     Parameters: {len(packet.parameters)}")

        communicator.disconnect()
        print("   Disconnected")

    # 10. List all monitored devices
    print("\n10. Monitoring Summary:")
    all_devices = hub.get_all_devices()
    print(f"   Total devices monitored: {len(all_devices)}")
    for device in all_devices:
        print(f"     - {device}")

    print("\n" + "=" * 60)
    print("Remote monitoring demonstration complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

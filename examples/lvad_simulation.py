#!/usr/bin/env python3
"""
LVAD Simulation Example

Demonstrates:
- Creating an LVAD from vendor profile
- Setting up patient cardiovascular model
- Running exercise stress test simulation
- Monitoring telemetry and alarms
"""

import sys
sys.path.insert(0, '..')

from implantable_systems.vendors import (
    LVADProfiles,
    create_lvad_from_profile,
    DeviceRegistry,
)
from implantable_systems.simulation import (
    DeviceSimulator,
    CardiovascularModel,
    SimulationConfig,
    Scenario,
)
from implantable_systems.safety import (
    AlarmManager,
    StandardAlarms,
    AlarmPriority,
)
from implantable_systems.data_management import (
    TelemetryBuffer,
    TrendAnalyzer,
    DataExporter,
)
from implantable_systems.clinical import (
    Patient,
    Clinician,
    LVADProgrammer,
    ProgrammingMode,
)


def main():
    print("=" * 60)
    print("LVAD Simulation Example")
    print("=" * 60)

    # 1. List available LVAD devices
    print("\nAvailable LVAD devices:")
    for device_name in DeviceRegistry.list_lvads():
        print(f"  - {device_name}")

    # 2. Create a HeartMate 3 LVAD
    print("\n1. Creating HeartMate 3 LVAD...")
    lvad = create_lvad_from_profile(LVADProfiles.HEARTMATE_3)
    print(f"   Device: {lvad.name}")
    print(f"   Manufacturer: {lvad.manufacturer}")
    print(f"   Flow type: {lvad.flow_type.value}")

    # 3. Activate the device
    print("\n2. Activating device...")
    lvad.activate()
    print(f"   State: {lvad.state.value}")

    # 4. Set up patient model
    print("\n3. Creating patient cardiovascular model...")
    patient = CardiovascularModel(
        native_heart_function=0.15,  # Severe heart failure
        body_surface_area=1.9,
        baseline_heart_rate=85,
    )
    print(f"   Native heart function: {patient.native_heart_function * 100:.0f}%")

    # 5. Set up simulation
    print("\n4. Configuring simulation...")
    config = SimulationConfig(
        time_step_seconds=0.1,
        duration_hours=0.05,  # 3 minutes for demo
        log_interval_seconds=10.0,
        enable_noise=True,
    )

    simulator = DeviceSimulator(lvad, patient, config)

    # Load exercise scenario
    scenario = Scenario.exercise_test(duration_minutes=3.0)
    simulator.load_scenario(scenario)
    print(f"   Scenario: {scenario.name}")
    print(f"   Duration: {config.duration_hours * 60:.1f} minutes")

    # 6. Set up telemetry monitoring
    telemetry_buffer = TelemetryBuffer()

    def telemetry_callback(time_sec, data):
        if "device" in data and "flow_lpm" in data["device"]:
            from implantable_systems.data_management import TelemetryPoint
            point = TelemetryPoint(
                timestamp=data.get("timestamp", ""),
                parameter="flow_lpm",
                value=data["device"]["flow_lpm"],
                unit="L/min",
                device_serial=lvad.model,
            )
            telemetry_buffer.add(point)

    simulator.add_callback(telemetry_callback)

    # 7. Set up alarm monitoring
    alarm_manager = AlarmManager()
    for alarm_def in StandardAlarms.get_lvad_alarms():
        alarm_manager.register_alarm(alarm_def)

    def alarm_handler(alarm):
        print(f"\n   *** ALARM: {alarm.message} ***")

    alarm_manager.register_handler(AlarmPriority.HIGH, alarm_handler)

    # 8. Run simulation
    print("\n5. Running simulation...")
    print("   " + "-" * 50)

    for result in simulator.run():
        device = result.get("device", {})
        patient_state = result.get("patient", {})

        flow = device.get("flow_lpm", 0)
        power = device.get("power_watts", 0)
        hr = patient_state.get("heart_rate_bpm", 0)
        map_val = patient_state.get("mean_arterial_pressure_mmhg", 0)
        activity = patient_state.get("activity_level", 0)

        print(f"   t={result['time']:5.0f}s | Flow: {flow:.1f} L/min | "
              f"Power: {power:.1f}W | HR: {hr:.0f} | MAP: {map_val:.0f} | "
              f"Activity: {activity*100:.0f}%")

        # Check for alarms
        if flow > 0:
            alarm_manager.check_condition(lvad.model, "flow_lpm", flow)

    # 9. Analyze results
    print("\n6. Simulation Results:")
    results = simulator.get_results()
    print(f"   Duration: {results['duration_seconds']:.1f} seconds")
    print(f"   Samples: {results['samples']}")

    if "flow_lpm" in results.get("device_statistics", {}):
        flow_stats = results["device_statistics"]["flow_lpm"]
        print(f"   Flow range: {flow_stats['min']:.1f} - {flow_stats['max']:.1f} L/min")
        print(f"   Mean flow: {flow_stats['mean']:.1f} L/min")

    # 10. Trend analysis
    print("\n7. Trend Analysis:")
    analyzer = TrendAnalyzer(telemetry_buffer)
    trend = analyzer.calculate_trend("flow_lpm", window_hours=1.0)
    print(f"   Flow trend: {trend.get('trend', 'N/A')}")
    print(f"   Samples analyzed: {trend.get('points', 0)}")

    # 11. Clinical programming demo
    print("\n8. Clinical Programming Session:")

    patient_info = Patient(
        id="PT001",
        mrn="12345678",
        name="John Doe",
        date_of_birth="1960-05-15",
        implant_date="2024-01-15",
        device_serial=lvad.model,
        indication="End-stage heart failure",
    )

    clinician = Clinician(
        id="DR001",
        name="Dr. Jane Smith",
        credentials="MD",
        institution="University Hospital",
        specialization="Heart Failure",
    )

    programmer = LVADProgrammer(lvad)
    session = programmer.start_session(
        patient=patient_info,
        clinician=clinician,
        device_serial=lvad.model,
        mode=ProgrammingMode.PROGRAMMING,
    )

    print(f"   Session ID: {session.session_id}")
    print(f"   Patient: {patient_info.name}")
    print(f"   Clinician: {clinician.name}")

    # Read current parameters
    params = programmer.read_device_parameters()
    print("\n   Current parameters:")
    for key, value in params.items():
        if value is not None:
            print(f"     {key}: {value}")

    # Make an adjustment
    programmer.apply_parameter(
        "target_flow_lpm",
        5.5,
        rationale="Optimize for patient activity level"
    )

    # End session
    report = programmer.end_session("Routine follow-up, parameters optimized")
    print(f"\n   Session completed. Changes made: {report['total_changes']}")

    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

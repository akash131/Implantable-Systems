#!/usr/bin/env python3
"""
Deep Brain Stimulator Closed-Loop Example

Demonstrates:
- Creating a DBS system from vendor profile
- Setting up closed-loop adaptive stimulation
- Patient neurological model simulation
- MRI safety assessment
"""

import sys
sys.path.insert(0, '..')

from implantable_systems.vendors import (
    DBSProfiles,
    create_dbs_from_profile,
)
from implantable_systems.deep_brain_stimulator import (
    StimulationParameters,
    StimulationMode,
    NeuralSignal,
    BrainTarget,
)
from implantable_systems.simulation import (
    DeviceSimulator,
    NeurologicalModel,
    SimulationConfig,
)
from implantable_systems.clinical import (
    Patient,
    Clinician,
    DBSProgrammer,
    ProgrammingMode,
)


def main():
    print("=" * 60)
    print("Deep Brain Stimulator Closed-Loop Example")
    print("=" * 60)

    # 1. Create Medtronic Percept DBS (with sensing capability)
    print("\n1. Creating Medtronic Percept DBS system...")
    dbs = create_dbs_from_profile(
        DBSProfiles.MEDTRONIC_PERCEPT,
        target=BrainTarget.STN,
        bilateral=True,
    )
    print(f"   Device: {dbs.name}")
    print(f"   Target: {dbs.target.value}")
    print(f"   Bilateral: {dbs.bilateral}")
    print(f"   Sensing capability: {DBSProfiles.MEDTRONIC_PERCEPT.has_sensing}")

    # 2. Activate and configure
    print("\n2. Activating device...")
    dbs.activate()

    # Set initial stimulation parameters
    initial_params = StimulationParameters(
        amplitude_ma=2.0,
        pulse_width_us=60,
        frequency_hz=130,
        active_contacts=[1, 2],
    )
    dbs.set_stimulation_parameters(initial_params, side="right")
    dbs.set_stimulation_parameters(initial_params, side="left")

    # Enable closed-loop mode
    dbs.set_mode(StimulationMode.CLOSED_LOOP)
    dbs.start_stimulation()
    print(f"   Mode: {dbs._mode.value}")
    print(f"   Stimulation active: {dbs._stimulation_active}")

    # 3. Create patient neurological model
    print("\n3. Creating patient neurological model...")
    patient = NeurologicalModel(
        condition="parkinsons",
        baseline_symptoms=0.7,
    )
    print(f"   Condition: {patient.condition}")
    print(f"   Baseline symptom severity: {patient.baseline_symptoms}")

    # 4. Run closed-loop simulation
    print("\n4. Running closed-loop stimulation simulation...")
    print("   " + "-" * 50)

    config = SimulationConfig(
        time_step_seconds=0.1,
        duration_hours=0.01,  # 36 seconds for demo
        log_interval_seconds=4.0,
    )

    simulator = DeviceSimulator(dbs, patient, config)

    print("   Time | Beta | Symptoms | Amplitude | Tremor")
    print("   " + "-" * 50)

    for result in simulator.run():
        patient_state = result.get("patient", {})
        device_state = result.get("device", {})

        beta = patient_state.get("beta_power", 0)
        symptoms = patient_state.get("symptom_severity", 0)
        tremor = patient_state.get("tremor_amplitude", 0)

        # Get controller metrics if available
        controller_metrics = device_state.get("controller_metrics", {})
        amplitude = controller_metrics.get("current_amplitude_ma",
                    device_state.get("right_amplitude_ma", 2.0))

        print(f"   {result['time']:5.1f}s | {beta:.2f} | {symptoms:.2f}     | "
              f"{amplitude:.2f} mA   | {tremor:.2f}")

    # 5. Show adaptation results
    print("\n5. Closed-loop adaptation results:")
    results = simulator.get_results()

    if "beta_power" in results.get("patient_statistics", {}):
        beta_stats = results["patient_statistics"]["beta_power"]
        print(f"   Beta power: {beta_stats['mean']:.3f} (target: 0.3)")

    if "symptom_severity" in results.get("patient_statistics", {}):
        symptom_stats = results["patient_statistics"]["symptom_severity"]
        print(f"   Symptom improvement: {(0.7 - symptom_stats['final']) / 0.7 * 100:.1f}%")

    # 6. MRI safety assessment
    print("\n6. MRI Safety Assessment:")
    print("   Checking compatibility for 1.5T MRI scan...")

    assessment = dbs.assess_mri_safety(
        field_strength_t=1.5,
        sar_w_kg=0.08,
        duration_min=30,
    )

    print(f"   Safe to scan: {assessment.get('is_safe', 'Unknown')}")
    print(f"   Estimated temperature rise: {assessment.get('estimated_temp_rise_c', 0):.2f}°C")

    if assessment.get("recommendations"):
        print("   Recommendations:")
        for rec in assessment["recommendations"][:3]:
            print(f"     - {rec}")

    # Enter MRI-safe mode
    print("\n   Entering MRI-safe mode...")
    dbs.enter_mri_safe_mode()
    print(f"   Device state: {dbs.state.value}")
    print(f"   Stimulation active: {dbs._stimulation_active}")

    # Exit MRI mode
    dbs.exit_mri_safe_mode()
    print("   Exited MRI-safe mode, stimulation can resume")

    # 7. Clinical programming
    print("\n7. Clinical Programming Session:")

    patient_info = Patient(
        id="PT002",
        mrn="87654321",
        name="Jane Smith",
        date_of_birth="1955-08-20",
        implant_date="2023-06-01",
        device_serial=dbs.model,
        indication="Parkinson's Disease",
        medications=["Levodopa/Carbidopa", "Pramipexole"],
    )

    clinician = Clinician(
        id="DR002",
        name="Dr. Robert Johnson",
        credentials="MD, PhD",
        institution="Movement Disorders Center",
        specialization="Movement Disorders",
    )

    programmer = DBSProgrammer(dbs)
    session = programmer.start_session(
        patient=patient_info,
        clinician=clinician,
        device_serial=dbs.model,
        mode=ProgrammingMode.PROGRAMMING,
    )

    # Run impedance check
    print("\n   Running impedance check...")
    impedances = programmer.run_impedance_check()
    print(f"   Right lead contacts: {len(impedances['right_lead'])}")

    # Adjust parameters
    print("\n   Adjusting stimulation amplitude...")
    programmer.apply_parameter(
        "right_amplitude_ma",
        2.5,
        rationale="Increase for better symptom control"
    )

    report = programmer.end_session("Parameter optimization complete")
    print(f"   Session duration: {report['duration_minutes']:.1f} minutes")

    print("\n" + "=" * 60)
    print("DBS closed-loop simulation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

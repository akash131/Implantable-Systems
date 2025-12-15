"""
Implantable Medical Device Systems Library

A comprehensive simulation and modeling library for implantable medical devices including:
- Ventricular Assist Devices (LVADs)
- Total Artificial Hearts
- Retinal Implants
- Deep Brain Stimulators

Features:
- Vendor-specific device profiles (HeartMate, Medtronic, Boston Scientific, etc.)
- Clinical programming interfaces with audit trails
- Remote monitoring and telemetry management
- Alarm and safety interlock systems
- Time-based simulation with patient physiological models
- Communication protocols (BLE, NFC, inductive telemetry)
"""

# Base classes
from .base import (
    ImplantableDevice,
    PowerSystem,
    Controller,
    PIDController,
    Electrode,
    ElectrodeConfiguration,
    MaterialProperties,
    PhysiologicalState,
    DeviceState,
)

# Device modules
from .lvad import (
    LVAD,
    ContinuousFlowPump,
    PulsatileFlowPump,
    TranscutaneousEnergyTransfer,
    PhysiologicalController,
    FlowType,
    PumpType,
)
from .artificial_heart import (
    TotalArtificialHeart,
    DualVentricle,
    HeartPowerSystem,
    TAHController,
    VentricleType,
)
from .retinal_implant import (
    RetinalImplant,
    MicroelectrodeArray,
    ImageProcessor,
    WirelessPowerDataLink,
    ImplantLocation,
)
from .deep_brain_stimulator import (
    DeepBrainStimulator,
    ClosedLoopController,
    MRIConditionalDesign,
    DBSElectrodeArray,
    StimulationParameters,
    NeuralSignal,
    BrainTarget,
    StimulationMode,
)

# Vendor profiles
from .vendors import (
    DeviceRegistry,
    LVADProfiles,
    TAHProfiles,
    RetinalImplantProfiles,
    DBSProfiles,
    create_lvad_from_profile,
    create_tah_from_profile,
    create_retinal_implant_from_profile,
    create_dbs_from_profile,
    Manufacturer,
)

# Clinical programming
from .clinical import (
    ClinicalProgrammer,
    LVADProgrammer,
    DBSProgrammer,
    ProgrammingSession,
    Patient,
    Clinician,
    ProgrammingMode,
    SafetyValidator,
    ClinicWorkflow,
    FollowUpSchedule,
)

# Data management
from .data_management import (
    TelemetryBuffer,
    TelemetryPoint,
    TelemetryWindow,
    TrendAnalyzer,
    DataExporter,
    RemoteMonitoringHub,
    DataLogger,
    DataResolution,
    ExportFormat,
)

# Safety systems
from .safety import (
    AlarmManager,
    Alarm,
    AlarmDefinition,
    AlarmPriority,
    AlarmCategory,
    AlarmState,
    SafetyInterlock,
    SafetyMonitor,
    HazardAnalysis,
    HazardEntry,
    StandardAlarms,
)

# Simulation
from .simulation import (
    DeviceSimulator,
    SimulationConfig,
    SimulationState,
    PatientModel,
    CardiovascularModel,
    NeurologicalModel,
    VisualModel,
    Scenario,
    SimulationEvent,
)

# Communication
from .communication import (
    CommunicationChannel,
    BLEChannel,
    NFCChannel,
    InductiveTelemetry,
    DeviceCommunicator,
    SecureSession,
    TelemetryPacket,
    Message,
    ConnectionState,
    ProtocolType,
)

__version__ = "2.0.0"
__all__ = [
    # Base classes
    "ImplantableDevice",
    "PowerSystem",
    "Controller",
    "PIDController",
    "Electrode",
    "ElectrodeConfiguration",
    "MaterialProperties",
    "PhysiologicalState",
    "DeviceState",
    # LVAD
    "LVAD",
    "ContinuousFlowPump",
    "PulsatileFlowPump",
    "TranscutaneousEnergyTransfer",
    "PhysiologicalController",
    "FlowType",
    "PumpType",
    # Total Artificial Heart
    "TotalArtificialHeart",
    "DualVentricle",
    "HeartPowerSystem",
    "TAHController",
    "VentricleType",
    # Retinal Implant
    "RetinalImplant",
    "MicroelectrodeArray",
    "ImageProcessor",
    "WirelessPowerDataLink",
    "ImplantLocation",
    # Deep Brain Stimulator
    "DeepBrainStimulator",
    "ClosedLoopController",
    "MRIConditionalDesign",
    "DBSElectrodeArray",
    "StimulationParameters",
    "NeuralSignal",
    "BrainTarget",
    "StimulationMode",
    # Vendors
    "DeviceRegistry",
    "LVADProfiles",
    "TAHProfiles",
    "RetinalImplantProfiles",
    "DBSProfiles",
    "create_lvad_from_profile",
    "create_tah_from_profile",
    "create_retinal_implant_from_profile",
    "create_dbs_from_profile",
    "Manufacturer",
    # Clinical
    "ClinicalProgrammer",
    "LVADProgrammer",
    "DBSProgrammer",
    "ProgrammingSession",
    "Patient",
    "Clinician",
    "ProgrammingMode",
    "SafetyValidator",
    "ClinicWorkflow",
    "FollowUpSchedule",
    # Data Management
    "TelemetryBuffer",
    "TelemetryPoint",
    "TelemetryWindow",
    "TrendAnalyzer",
    "DataExporter",
    "RemoteMonitoringHub",
    "DataLogger",
    "DataResolution",
    "ExportFormat",
    # Safety
    "AlarmManager",
    "Alarm",
    "AlarmDefinition",
    "AlarmPriority",
    "AlarmCategory",
    "AlarmState",
    "SafetyInterlock",
    "SafetyMonitor",
    "HazardAnalysis",
    "HazardEntry",
    "StandardAlarms",
    # Simulation
    "DeviceSimulator",
    "SimulationConfig",
    "SimulationState",
    "PatientModel",
    "CardiovascularModel",
    "NeurologicalModel",
    "VisualModel",
    "Scenario",
    "SimulationEvent",
    # Communication
    "CommunicationChannel",
    "BLEChannel",
    "NFCChannel",
    "InductiveTelemetry",
    "DeviceCommunicator",
    "SecureSession",
    "TelemetryPacket",
    "Message",
    "ConnectionState",
    "ProtocolType",
]

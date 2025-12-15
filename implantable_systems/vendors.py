"""
Vendor-specific device profiles for implantable systems.

Provides pre-configured device profiles matching commercial implantable devices
from major manufacturers including Abbott, Medtronic, Boston Scientific, etc.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .lvad import (
    LVAD,
    ContinuousFlowPump,
    TranscutaneousEnergyTransfer,
    PhysiologicalController,
    BiocompatibleCoating,
    FlowType,
    PumpType,
)
from .artificial_heart import (
    TotalArtificialHeart,
    DualVentricle,
    HeartPowerSystem,
    VentricleParameters,
)
from .retinal_implant import (
    RetinalImplant,
    MicroelectrodeArray,
    ImplantLocation,
)
from .deep_brain_stimulator import (
    DeepBrainStimulator,
    DBSElectrodeArray,
    MRIConditionalDesign,
    StimulationParameters,
    BrainTarget,
)
from .base import MaterialProperties


class Manufacturer(Enum):
    """Medical device manufacturers."""
    ABBOTT = "Abbott"
    MEDTRONIC = "Medtronic"
    BOSTON_SCIENTIFIC = "Boston Scientific"
    SECOND_SIGHT = "Second Sight"
    SYNCARDIA = "SynCardia"
    ABIOMED = "Abiomed"
    PIXIUM_VISION = "Pixium Vision"
    RETINA_IMPLANT_AG = "Retina Implant AG"


# =============================================================================
# LVAD Vendor Profiles
# =============================================================================

@dataclass
class LVADProfile:
    """Configuration profile for vendor-specific LVADs."""
    name: str
    manufacturer: Manufacturer
    model: str
    pump_type: PumpType
    max_speed_rpm: float
    min_speed_rpm: float
    max_flow_lpm: float
    weight_grams: float
    displacement_cc: float
    bearing_type: str
    has_pulse_mode: bool
    battery_life_hours: float
    fda_approval_year: Optional[int] = None


class LVADProfiles:
    """Pre-defined LVAD profiles matching commercial devices."""

    HEARTMATE_3 = LVADProfile(
        name="HeartMate 3",
        manufacturer=Manufacturer.ABBOTT,
        model="HM3",
        pump_type=PumpType.CENTRIFUGAL,
        max_speed_rpm=9000,
        min_speed_rpm=3000,
        max_flow_lpm=10.0,
        weight_grams=200,
        displacement_cc=80,
        bearing_type="full_magnetic_levitation",
        has_pulse_mode=True,  # Artificial pulse feature
        battery_life_hours=17,
        fda_approval_year=2017,
    )

    HEARTMATE_II = LVADProfile(
        name="HeartMate II",
        manufacturer=Manufacturer.ABBOTT,
        model="HMII",
        pump_type=PumpType.AXIAL,
        max_speed_rpm=15000,
        min_speed_rpm=6000,
        max_flow_lpm=10.0,
        weight_grams=375,
        displacement_cc=124,
        bearing_type="pivot_bearing",
        has_pulse_mode=False,
        battery_life_hours=10,
        fda_approval_year=2008,
    )

    HEARTWARE_HVAD = LVADProfile(
        name="HeartWare HVAD",
        manufacturer=Manufacturer.MEDTRONIC,
        model="HVAD",
        pump_type=PumpType.CENTRIFUGAL,
        max_speed_rpm=4000,
        min_speed_rpm=1800,
        max_flow_lpm=10.0,
        weight_grams=145,
        displacement_cc=50,
        bearing_type="hybrid_magnetic_hydrodynamic",
        has_pulse_mode=False,
        battery_life_hours=12,
        fda_approval_year=2012,
    )

    IMPELLA_5_5 = LVADProfile(
        name="Impella 5.5",
        manufacturer=Manufacturer.ABIOMED,
        model="IMPELLA-5.5",
        pump_type=PumpType.AXIAL,
        max_speed_rpm=33000,
        min_speed_rpm=12000,
        max_flow_lpm=6.2,
        weight_grams=25,  # Catheter-based
        displacement_cc=5,
        bearing_type="motor_purge",
        has_pulse_mode=False,
        battery_life_hours=0,  # External power
        fda_approval_year=2019,
    )

    @classmethod
    def get_all_profiles(cls) -> list[LVADProfile]:
        """Get all available LVAD profiles."""
        return [cls.HEARTMATE_3, cls.HEARTMATE_II, cls.HEARTWARE_HVAD, cls.IMPELLA_5_5]


def create_lvad_from_profile(profile: LVADProfile) -> LVAD:
    """Create an LVAD instance configured according to a vendor profile."""
    lvad = LVAD(
        name=profile.name,
        manufacturer=profile.manufacturer.value,
        model=profile.model,
        flow_type=FlowType.HYBRID if profile.has_pulse_mode else FlowType.CONTINUOUS,
        pump_type=profile.pump_type,
    )

    # Configure pump parameters
    if hasattr(lvad._pump, 'max_speed_rpm'):
        lvad._pump.max_speed_rpm = profile.max_speed_rpm

    # Configure controller
    lvad._controller.min_speed_rpm = profile.min_speed_rpm
    lvad._controller.max_speed_rpm = profile.max_speed_rpm

    return lvad


# =============================================================================
# Total Artificial Heart Vendor Profiles
# =============================================================================

@dataclass
class TAHProfile:
    """Configuration profile for vendor-specific TAH systems."""
    name: str
    manufacturer: Manufacturer
    model: str
    max_cardiac_output_lpm: float
    stroke_volume_ml: float
    max_rate_bpm: int
    weight_grams: float
    power_consumption_watts: float
    battery_life_hours: float
    fda_status: str


class TAHProfiles:
    """Pre-defined TAH profiles matching commercial devices."""

    SYNCARDIA_70CC = TAHProfile(
        name="SynCardia 70cc TAH",
        manufacturer=Manufacturer.SYNCARDIA,
        model="TAH-70",
        max_cardiac_output_lpm=9.5,
        stroke_volume_ml=70,
        max_rate_bpm=130,
        weight_grams=160,
        power_consumption_watts=13,
        battery_life_hours=4,
        fda_status="approved",
    )

    SYNCARDIA_50CC = TAHProfile(
        name="SynCardia 50cc TAH",
        manufacturer=Manufacturer.SYNCARDIA,
        model="TAH-50",
        max_cardiac_output_lpm=7.5,
        stroke_volume_ml=50,
        max_rate_bpm=130,
        weight_grams=140,
        power_consumption_watts=12,
        battery_life_hours=4.5,
        fda_status="approved",
    )

    @classmethod
    def get_all_profiles(cls) -> list[TAHProfile]:
        """Get all available TAH profiles."""
        return [cls.SYNCARDIA_70CC, cls.SYNCARDIA_50CC]


def create_tah_from_profile(profile: TAHProfile) -> TotalArtificialHeart:
    """Create a TAH instance configured according to a vendor profile."""
    tah = TotalArtificialHeart(
        name=profile.name,
        manufacturer=profile.manufacturer.value,
        model=profile.model,
    )

    # Configure ventricle parameters
    tah._ventricles.left_params.max_stroke_volume_ml = profile.stroke_volume_ml
    tah._ventricles.right_params.max_stroke_volume_ml = profile.stroke_volume_ml
    tah._ventricles.left_params.max_rate_bpm = profile.max_rate_bpm
    tah._ventricles.right_params.max_rate_bpm = profile.max_rate_bpm

    return tah


# =============================================================================
# Retinal Implant Vendor Profiles
# =============================================================================

@dataclass
class RetinalImplantProfile:
    """Configuration profile for vendor-specific retinal implants."""
    name: str
    manufacturer: Manufacturer
    model: str
    electrode_count: int
    array_rows: int
    array_cols: int
    electrode_diameter_um: float
    pitch_um: float
    location: ImplantLocation
    visual_field_degrees: float
    wireless_frequency_mhz: float
    fda_status: str


class RetinalImplantProfiles:
    """Pre-defined retinal implant profiles matching commercial devices."""

    ARGUS_II = RetinalImplantProfile(
        name="Argus II",
        manufacturer=Manufacturer.SECOND_SIGHT,
        model="ARGUS-II",
        electrode_count=60,
        array_rows=6,
        array_cols=10,
        electrode_diameter_um=200,
        pitch_um=575,
        location=ImplantLocation.EPIRETINAL,
        visual_field_degrees=20,
        wireless_frequency_mhz=3.0,
        fda_status="approved_2013",
    )

    PRIMA = RetinalImplantProfile(
        name="PRIMA System",
        manufacturer=Manufacturer.PIXIUM_VISION,
        model="PRIMA",
        electrode_count=378,
        array_rows=19,  # Hexagonal arrangement approximated
        array_cols=20,
        electrode_diameter_um=100,
        pitch_um=140,
        location=ImplantLocation.SUBRETINAL,
        visual_field_degrees=5,  # Central vision
        wireless_frequency_mhz=880,  # IR light-based
        fda_status="investigational",
    )

    ALPHA_AMS = RetinalImplantProfile(
        name="Alpha AMS",
        manufacturer=Manufacturer.RETINA_IMPLANT_AG,
        model="ALPHA-AMS",
        electrode_count=1600,
        array_rows=40,
        array_cols=40,
        electrode_diameter_um=50,
        pitch_um=70,
        location=ImplantLocation.SUBRETINAL,
        visual_field_degrees=11,
        wireless_frequency_mhz=13.56,  # NFC
        fda_status="ce_marked",
    )

    @classmethod
    def get_all_profiles(cls) -> list[RetinalImplantProfile]:
        """Get all available retinal implant profiles."""
        return [cls.ARGUS_II, cls.PRIMA, cls.ALPHA_AMS]


def create_retinal_implant_from_profile(profile: RetinalImplantProfile) -> RetinalImplant:
    """Create a retinal implant instance configured according to a vendor profile."""
    return RetinalImplant(
        name=profile.name,
        manufacturer=profile.manufacturer.value,
        model=profile.model,
        array_rows=profile.array_rows,
        array_cols=profile.array_cols,
        electrode_size_um=profile.electrode_diameter_um,
        location=profile.location,
    )


# =============================================================================
# Deep Brain Stimulator Vendor Profiles
# =============================================================================

@dataclass
class DBSProfile:
    """Configuration profile for vendor-specific DBS systems."""
    name: str
    manufacturer: Manufacturer
    model: str
    ipg_model: str
    channels: int
    has_sensing: bool
    has_directional: bool
    mri_conditional: bool
    max_amplitude_ma: float
    max_frequency_hz: float
    battery_type: str
    battery_longevity_years: float
    fda_approval_year: Optional[int] = None


class DBSProfiles:
    """Pre-defined DBS profiles matching commercial devices."""

    MEDTRONIC_PERCEPT = DBSProfile(
        name="Percept PC",
        manufacturer=Manufacturer.MEDTRONIC,
        model="PERCEPT-PC",
        ipg_model="B35300",
        channels=8,
        has_sensing=True,  # BrainSense technology
        has_directional=True,
        mri_conditional=True,
        max_amplitude_ma=25.5,
        max_frequency_hz=250,
        battery_type="primary",
        battery_longevity_years=5,
        fda_approval_year=2020,
    )

    MEDTRONIC_ACTIVA = DBSProfile(
        name="Activa RC",
        manufacturer=Manufacturer.MEDTRONIC,
        model="ACTIVA-RC",
        ipg_model="37612",
        channels=4,
        has_sensing=False,
        has_directional=False,
        mri_conditional=True,
        max_amplitude_ma=25.5,
        max_frequency_hz=250,
        battery_type="rechargeable",
        battery_longevity_years=15,
        fda_approval_year=2009,
    )

    BOSTON_VERCISE = DBSProfile(
        name="Vercise Genus",
        manufacturer=Manufacturer.BOSTON_SCIENTIFIC,
        model="VERCISE-GENUS",
        ipg_model="DB-2221",
        channels=16,
        has_sensing=False,
        has_directional=True,  # MICC - Multiple Independent Current Control
        mri_conditional=True,
        max_amplitude_ma=20.0,
        max_frequency_hz=500,
        battery_type="rechargeable",
        battery_longevity_years=25,
        fda_approval_year=2019,
    )

    ABBOTT_INFINITY = DBSProfile(
        name="Infinity DBS",
        manufacturer=Manufacturer.ABBOTT,
        model="INFINITY",
        ipg_model="6660",
        channels=8,
        has_sensing=False,
        has_directional=True,
        mri_conditional=True,
        max_amplitude_ma=12.75,
        max_frequency_hz=250,
        battery_type="primary",
        battery_longevity_years=5,
        fda_approval_year=2016,
    )

    @classmethod
    def get_all_profiles(cls) -> list[DBSProfile]:
        """Get all available DBS profiles."""
        return [
            cls.MEDTRONIC_PERCEPT,
            cls.MEDTRONIC_ACTIVA,
            cls.BOSTON_VERCISE,
            cls.ABBOTT_INFINITY,
        ]


def create_dbs_from_profile(
    profile: DBSProfile,
    target: BrainTarget = BrainTarget.STN,
    bilateral: bool = True,
) -> DeepBrainStimulator:
    """Create a DBS instance configured according to a vendor profile."""
    dbs = DeepBrainStimulator(
        name=profile.name,
        manufacturer=profile.manufacturer.value,
        model=profile.model,
        target=target,
        bilateral=bilateral,
        mri_conditional=profile.mri_conditional,
    )

    # Configure electrodes for directional if supported
    if profile.has_directional:
        dbs._right_lead = DBSElectrodeArray(
            num_contacts=profile.channels // (2 if bilateral else 1),
            is_directional=True,
        )
        if bilateral:
            dbs._left_lead = DBSElectrodeArray(
                num_contacts=profile.channels // 2,
                is_directional=True,
            )

    # Configure battery
    dbs._power_system.is_rechargeable = (profile.battery_type == "rechargeable")
    dbs._power_system.longevity_years = profile.battery_longevity_years

    return dbs


# =============================================================================
# Device Registry
# =============================================================================

class DeviceRegistry:
    """
    Central registry for all vendor device profiles.

    Provides easy access to device profiles and factory methods.
    """

    @staticmethod
    def list_lvads() -> list[str]:
        """List all available LVAD models."""
        return [p.name for p in LVADProfiles.get_all_profiles()]

    @staticmethod
    def list_tahs() -> list[str]:
        """List all available TAH models."""
        return [p.name for p in TAHProfiles.get_all_profiles()]

    @staticmethod
    def list_retinal_implants() -> list[str]:
        """List all available retinal implant models."""
        return [p.name for p in RetinalImplantProfiles.get_all_profiles()]

    @staticmethod
    def list_dbs_systems() -> list[str]:
        """List all available DBS models."""
        return [p.name for p in DBSProfiles.get_all_profiles()]

    @staticmethod
    def list_all_devices() -> dict[str, list[str]]:
        """List all available devices by category."""
        return {
            "lvad": DeviceRegistry.list_lvads(),
            "tah": DeviceRegistry.list_tahs(),
            "retinal": DeviceRegistry.list_retinal_implants(),
            "dbs": DeviceRegistry.list_dbs_systems(),
        }

    @staticmethod
    def create_device(device_name: str, **kwargs):
        """
        Create a device instance by name.

        Args:
            device_name: Name of the device to create
            **kwargs: Additional arguments passed to the factory

        Returns:
            Configured device instance
        """
        # Search LVAD profiles
        for profile in LVADProfiles.get_all_profiles():
            if profile.name.lower() == device_name.lower():
                return create_lvad_from_profile(profile)

        # Search TAH profiles
        for profile in TAHProfiles.get_all_profiles():
            if profile.name.lower() == device_name.lower():
                return create_tah_from_profile(profile)

        # Search retinal implant profiles
        for profile in RetinalImplantProfiles.get_all_profiles():
            if profile.name.lower() == device_name.lower():
                return create_retinal_implant_from_profile(profile)

        # Search DBS profiles
        for profile in DBSProfiles.get_all_profiles():
            if profile.name.lower() == device_name.lower():
                return create_dbs_from_profile(profile, **kwargs)

        raise ValueError(f"Unknown device: {device_name}")

    @staticmethod
    def get_device_info(device_name: str) -> dict:
        """Get detailed information about a device."""
        # Search all profiles
        all_profiles = (
            [(p, "lvad") for p in LVADProfiles.get_all_profiles()] +
            [(p, "tah") for p in TAHProfiles.get_all_profiles()] +
            [(p, "retinal") for p in RetinalImplantProfiles.get_all_profiles()] +
            [(p, "dbs") for p in DBSProfiles.get_all_profiles()]
        )

        for profile, category in all_profiles:
            if profile.name.lower() == device_name.lower():
                return {
                    "category": category,
                    "profile": profile.__dict__,
                }

        raise ValueError(f"Unknown device: {device_name}")

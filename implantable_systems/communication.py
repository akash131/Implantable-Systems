"""
Communication Protocols for Implantable Devices.

Provides:
- Bluetooth Low Energy (BLE) communication
- Near Field Communication (NFC) interfaces
- Proprietary telemetry protocols
- Secure data transmission
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Any, Optional
from abc import ABC, abstractmethod
import hashlib
import hmac
import struct


class ConnectionState(Enum):
    """Communication connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"


class ProtocolType(Enum):
    """Communication protocol types."""
    BLE = "bluetooth_low_energy"
    NFC = "near_field_communication"
    INDUCTIVE = "inductive_telemetry"
    PROPRIETARY = "proprietary"


@dataclass
class Message:
    """Communication message."""
    message_id: int
    command: str
    payload: bytes
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    requires_ack: bool = True
    priority: int = 0  # 0 = normal, higher = more urgent

    def to_bytes(self) -> bytes:
        """Serialize message to bytes."""
        header = struct.pack(
            ">HBB",
            self.message_id,
            len(self.command),
            self.priority,
        )
        command_bytes = self.command.encode("utf-8")
        length = struct.pack(">H", len(self.payload))
        return header + command_bytes + length + self.payload

    @classmethod
    def from_bytes(cls, data: bytes) -> "Message":
        """Deserialize message from bytes."""
        msg_id, cmd_len, priority = struct.unpack(">HBB", data[:4])
        command = data[4:4 + cmd_len].decode("utf-8")
        payload_len = struct.unpack(">H", data[4 + cmd_len:6 + cmd_len])[0]
        payload = data[6 + cmd_len:6 + cmd_len + payload_len]

        return cls(
            message_id=msg_id,
            command=command,
            payload=payload,
            priority=priority,
        )


class CommunicationChannel(ABC):
    """
    Abstract base class for device communication channels.

    Provides common interface for different protocols.
    """

    def __init__(self, device_id: str):
        self.device_id = device_id
        self._state = ConnectionState.DISCONNECTED
        self._message_counter = 0
        self._handlers: dict[str, Callable] = {}
        self._pending_acks: dict[int, Message] = {}
        self._encryption_key: Optional[bytes] = None

    @property
    def state(self) -> ConnectionState:
        """Current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if channel is connected."""
        return self._state in (ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED)

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection."""
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Close connection."""
        pass

    @abstractmethod
    def send(self, message: Message) -> bool:
        """Send a message."""
        pass

    @abstractmethod
    def receive(self) -> Message | None:
        """Receive a message (non-blocking)."""
        pass

    def register_handler(self, command: str, handler: Callable[[Message], Any]) -> None:
        """Register a handler for a specific command."""
        self._handlers[command] = handler

    def create_message(self, command: str, payload: bytes, priority: int = 0) -> Message:
        """Create a new message with auto-incrementing ID."""
        self._message_counter += 1
        return Message(
            message_id=self._message_counter,
            command=command,
            payload=payload,
            priority=priority,
        )

    def set_encryption_key(self, key: bytes) -> None:
        """Set encryption key for secure communication."""
        self._encryption_key = key

    def _encrypt(self, data: bytes) -> bytes:
        """Encrypt data using AES-like simulation."""
        if self._encryption_key is None:
            return data
        # Simple XOR encryption for simulation (real implementation would use AES)
        key = (self._encryption_key * (len(data) // len(self._encryption_key) + 1))[:len(data)]
        return bytes(a ^ b for a, b in zip(data, key))

    def _decrypt(self, data: bytes) -> bytes:
        """Decrypt data (XOR is symmetric)."""
        return self._encrypt(data)


class BLEChannel(CommunicationChannel):
    """
    Bluetooth Low Energy communication channel.

    Simulates BLE GATT-based communication for medical devices.
    """

    def __init__(
        self,
        device_id: str,
        service_uuid: str = "0000180D-0000-1000-8000-00805f9b34fb",
        mtu_size: int = 247,
    ):
        super().__init__(device_id)
        self.service_uuid = service_uuid
        self.mtu_size = mtu_size
        self._rssi = -60  # Signal strength simulation
        self._characteristics: dict[str, bytes] = {}
        self._notifications_enabled: set[str] = set()
        self._tx_buffer: list[Message] = []
        self._rx_buffer: list[Message] = []

    def connect(self) -> bool:
        """Simulate BLE connection."""
        self._state = ConnectionState.CONNECTING

        # Simulate connection process
        # In real implementation, this would scan and connect
        self._state = ConnectionState.CONNECTED

        return True

    def disconnect(self) -> None:
        """Disconnect BLE channel."""
        self._state = ConnectionState.DISCONNECTED
        self._notifications_enabled.clear()
        self._tx_buffer.clear()

    def authenticate(self, pin: str) -> bool:
        """Authenticate with device using PIN."""
        if self._state != ConnectionState.CONNECTED:
            return False

        # Simulate PIN verification
        expected_hash = hashlib.sha256(f"{self.device_id}{pin}".encode()).hexdigest()

        # For simulation, any 6-digit PIN works
        if len(pin) == 6 and pin.isdigit():
            self._state = ConnectionState.AUTHENTICATED
            return True

        return False

    def send(self, message: Message) -> bool:
        """Send message over BLE."""
        if not self.is_connected:
            return False

        data = message.to_bytes()

        # Check MTU size
        if len(data) > self.mtu_size:
            # Fragment message
            fragments = [
                data[i:i + self.mtu_size - 3]
                for i in range(0, len(data), self.mtu_size - 3)
            ]
            for i, frag in enumerate(fragments):
                # Add fragment header
                header = struct.pack(">BH", i == len(fragments) - 1, len(frag))
                self._tx_buffer.append(header + frag)
        else:
            self._tx_buffer.append(message)

        if message.requires_ack:
            self._pending_acks[message.message_id] = message

        return True

    def receive(self) -> Message | None:
        """Receive message from BLE."""
        if self._rx_buffer:
            return self._rx_buffer.pop(0)
        return None

    def enable_notifications(self, characteristic_uuid: str) -> bool:
        """Enable notifications for a characteristic."""
        if not self.is_connected:
            return False
        self._notifications_enabled.add(characteristic_uuid)
        return True

    def write_characteristic(self, uuid: str, data: bytes) -> bool:
        """Write to a GATT characteristic."""
        if not self.is_connected:
            return False
        self._characteristics[uuid] = data
        return True

    def read_characteristic(self, uuid: str) -> bytes | None:
        """Read from a GATT characteristic."""
        if not self.is_connected:
            return None
        return self._characteristics.get(uuid)

    def get_connection_info(self) -> dict:
        """Get BLE connection information."""
        return {
            "state": self._state.value,
            "device_id": self.device_id,
            "service_uuid": self.service_uuid,
            "mtu_size": self.mtu_size,
            "rssi": self._rssi,
            "notifications_enabled": list(self._notifications_enabled),
        }


class NFCChannel(CommunicationChannel):
    """
    Near Field Communication channel.

    Used for short-range device programming and data transfer.
    """

    def __init__(self, device_id: str):
        super().__init__(device_id)
        self._in_field = False
        self._data_buffer: bytes = b""

    def connect(self) -> bool:
        """Simulate NFC field detection."""
        self._in_field = True
        self._state = ConnectionState.CONNECTED
        return True

    def disconnect(self) -> None:
        """Remove from NFC field."""
        self._in_field = False
        self._state = ConnectionState.DISCONNECTED

    def send(self, message: Message) -> bool:
        """Send via NFC."""
        if not self._in_field:
            return False

        data = message.to_bytes()
        # NFC typically has ~424 bytes max per transmission
        if len(data) > 424:
            return False

        self._data_buffer = data
        return True

    def receive(self) -> Message | None:
        """Receive via NFC."""
        if not self._in_field or not self._data_buffer:
            return None

        msg = Message.from_bytes(self._data_buffer)
        self._data_buffer = b""
        return msg

    def read_ndef(self) -> dict | None:
        """Read NDEF records from NFC tag."""
        if not self._in_field:
            return None

        # Simulate NDEF record structure
        return {
            "type": "text/plain",
            "payload": f"Device: {self.device_id}",
            "language": "en",
        }


class InductiveTelemetry(CommunicationChannel):
    """
    Inductive telemetry for fully implanted devices.

    Used for devices without transcutaneous connections.
    """

    def __init__(
        self,
        device_id: str,
        carrier_frequency_khz: float = 175.0,
        data_rate_kbps: float = 100.0,
    ):
        super().__init__(device_id)
        self.carrier_frequency_khz = carrier_frequency_khz
        self.data_rate_kbps = data_rate_kbps
        self._head_position_ok = False
        self._signal_strength = 0.0
        self._tx_queue: list[Message] = []
        self._rx_queue: list[Message] = []

    def connect(self) -> bool:
        """Establish telemetry link."""
        if not self._head_position_ok:
            self._state = ConnectionState.ERROR
            return False

        if self._signal_strength < 0.3:
            self._state = ConnectionState.ERROR
            return False

        self._state = ConnectionState.CONNECTED
        return True

    def disconnect(self) -> None:
        """Disconnect telemetry."""
        self._state = ConnectionState.DISCONNECTED

    def set_head_position(self, position_ok: bool, signal_strength: float) -> None:
        """Set telemetry head position status."""
        self._head_position_ok = position_ok
        self._signal_strength = max(0.0, min(1.0, signal_strength))

    def send(self, message: Message) -> bool:
        """Send via inductive telemetry."""
        if not self.is_connected:
            return False

        # Simulate transmission time
        data_size_bits = len(message.to_bytes()) * 8
        tx_time_ms = data_size_bits / self.data_rate_kbps

        self._tx_queue.append(message)
        return True

    def receive(self) -> Message | None:
        """Receive from inductive telemetry."""
        if not self.is_connected or not self._rx_queue:
            return None
        return self._rx_queue.pop(0)

    def get_link_quality(self) -> dict:
        """Get telemetry link quality metrics."""
        return {
            "signal_strength": self._signal_strength,
            "head_position_ok": self._head_position_ok,
            "carrier_frequency_khz": self.carrier_frequency_khz,
            "effective_data_rate_kbps": self.data_rate_kbps * self._signal_strength,
        }


@dataclass
class TelemetryPacket:
    """Standard telemetry data packet."""
    sequence_number: int
    timestamp: int  # Milliseconds since device boot
    parameters: dict[str, float]
    battery_level: int  # 0-100
    alarm_flags: int  # Bit field
    checksum: int = 0

    def calculate_checksum(self) -> int:
        """Calculate packet checksum."""
        data = struct.pack(">IIB", self.sequence_number, self.timestamp, self.battery_level)
        for value in self.parameters.values():
            data += struct.pack(">f", value)
        return sum(data) & 0xFFFF

    def verify_checksum(self) -> bool:
        """Verify packet checksum."""
        return self.checksum == self.calculate_checksum()

    def to_bytes(self) -> bytes:
        """Serialize packet."""
        self.checksum = self.calculate_checksum()
        data = struct.pack(
            ">IIBH",
            self.sequence_number,
            self.timestamp,
            self.battery_level,
            len(self.parameters),
        )
        for name, value in self.parameters.items():
            name_bytes = name.encode("utf-8")[:16].ljust(16, b"\x00")
            data += name_bytes + struct.pack(">f", value)
        data += struct.pack(">IH", self.alarm_flags, self.checksum)
        return data


class DeviceCommunicator:
    """
    High-level device communication manager.

    Manages connections, protocols, and data synchronization.
    """

    def __init__(self, device):
        self._device = device
        self._channels: dict[ProtocolType, CommunicationChannel] = {}
        self._active_channel: CommunicationChannel | None = None
        self._telemetry_callback: Callable | None = None
        self._sequence_number = 0

    def add_channel(self, protocol: ProtocolType, channel: CommunicationChannel) -> None:
        """Add a communication channel."""
        self._channels[protocol] = channel

    def connect(self, protocol: ProtocolType = ProtocolType.BLE) -> bool:
        """Connect using specified protocol."""
        if protocol not in self._channels:
            return False

        channel = self._channels[protocol]
        if channel.connect():
            self._active_channel = channel
            return True
        return False

    def disconnect(self) -> None:
        """Disconnect active channel."""
        if self._active_channel:
            self._active_channel.disconnect()
            self._active_channel = None

    def send_command(self, command: str, params: dict | None = None) -> bool:
        """Send a command to the device."""
        if not self._active_channel or not self._active_channel.is_connected:
            return False

        payload = b""
        if params:
            import json
            payload = json.dumps(params).encode("utf-8")

        message = self._active_channel.create_message(command, payload)
        return self._active_channel.send(message)

    def request_telemetry(self) -> TelemetryPacket | None:
        """Request and receive telemetry packet."""
        if not self.send_command("GET_TELEMETRY"):
            return None

        # Simulate device response
        self._sequence_number += 1

        # Get device status
        status = {}
        if hasattr(self._device, 'get_comprehensive_status'):
            status = self._device.get_comprehensive_status()

        # Build telemetry packet
        parameters = {}
        if "current_flow_lpm" in status:
            parameters["flow"] = status["current_flow_lpm"]
        if "power_watts" in status:
            parameters["power"] = status["power_watts"]

        battery = 100
        if "power_status" in status:
            battery = int(status["power_status"].get("charge_level_percent", 100))

        packet = TelemetryPacket(
            sequence_number=self._sequence_number,
            timestamp=int(datetime.now().timestamp() * 1000) % (2**32),
            parameters=parameters,
            battery_level=battery,
            alarm_flags=0,
        )
        packet.checksum = packet.calculate_checksum()

        return packet

    def set_parameter(self, name: str, value: float) -> bool:
        """Set a device parameter remotely."""
        return self.send_command("SET_PARAM", {"name": name, "value": value})

    def on_telemetry(self, callback: Callable[[TelemetryPacket], None]) -> None:
        """Register callback for telemetry updates."""
        self._telemetry_callback = callback

    def get_connection_status(self) -> dict:
        """Get current connection status."""
        return {
            "connected": self._active_channel is not None and self._active_channel.is_connected,
            "protocol": next(
                (p.value for p, c in self._channels.items() if c == self._active_channel),
                None
            ),
            "channels_available": [p.value for p in self._channels.keys()],
        }


class SecureSession:
    """
    Secure communication session with authentication and encryption.

    Implements secure pairing and encrypted data transfer.
    """

    def __init__(self, channel: CommunicationChannel):
        self._channel = channel
        self._session_key: bytes | None = None
        self._is_authenticated = False
        self._nonce_counter = 0

    def pair(self, device_secret: str) -> bool:
        """
        Establish secure pairing with device.

        Uses challenge-response authentication.
        """
        if not self._channel.is_connected:
            return False

        # Generate challenge
        import secrets
        challenge = secrets.token_bytes(16)

        # Send challenge
        challenge_msg = self._channel.create_message("AUTH_CHALLENGE", challenge)
        self._channel.send(challenge_msg)

        # Compute expected response
        expected = hmac.new(
            device_secret.encode(),
            challenge,
            hashlib.sha256
        ).digest()

        # In real implementation, would receive response from device
        # For simulation, assume success
        self._session_key = hashlib.sha256(
            expected + challenge
        ).digest()[:16]

        self._channel.set_encryption_key(self._session_key)
        self._is_authenticated = True

        return True

    def send_secure(self, command: str, payload: dict) -> bool:
        """Send encrypted command."""
        if not self._is_authenticated:
            return False

        import json
        data = json.dumps(payload).encode("utf-8")

        # Add nonce for replay protection
        self._nonce_counter += 1
        nonce = struct.pack(">Q", self._nonce_counter)

        # MAC for integrity
        mac = hmac.new(self._session_key, nonce + data, hashlib.sha256).digest()[:8]

        message = self._channel.create_message(
            command,
            nonce + data + mac,
            priority=1
        )

        return self._channel.send(message)

    def verify_message(self, message: Message) -> tuple[bool, bytes]:
        """Verify and decrypt received message."""
        if not self._is_authenticated:
            return False, b""

        payload = message.payload
        if len(payload) < 16:  # nonce (8) + MAC (8)
            return False, b""

        nonce = payload[:8]
        mac_received = payload[-8:]
        data = payload[8:-8]

        # Verify MAC
        mac_expected = hmac.new(self._session_key, nonce + data, hashlib.sha256).digest()[:8]

        if not hmac.compare_digest(mac_received, mac_expected):
            return False, b""

        return True, data

    @property
    def is_authenticated(self) -> bool:
        """Check if session is authenticated."""
        return self._is_authenticated

"""In-memory state management for the simulated imaging device."""
import time
import threading


class DeviceState:
    """Thread-safe mutable state for the device simulator."""

    def __init__(self):
        self._lock = threading.Lock()
        self._status = "online"
        self._failure_mode = None
        self._capture_count = 0
        self._start_time = time.time()
        self._firmware_version = "SIM-FW-3.2.1"
        self._last_calibration = "2026-01-15T08:00:00Z"

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @status.setter
    def status(self, value: str):
        with self._lock:
            self._status = value

    @property
    def failure_mode(self):
        with self._lock:
            return self._failure_mode

    @failure_mode.setter
    def failure_mode(self, value):
        with self._lock:
            self._failure_mode = value

    @property
    def capture_count(self) -> int:
        with self._lock:
            return self._capture_count

    def increment_capture(self):
        with self._lock:
            self._capture_count += 1

    @property
    def uptime_seconds(self) -> float:
        return round(time.time() - self._start_time, 2)

    @property
    def firmware_version(self) -> str:
        return self._firmware_version

    @property
    def last_calibration(self) -> str:
        return self._last_calibration

    def reset(self):
        """Reset device to default online state with no failure modes."""
        with self._lock:
            self._status = "online"
            self._failure_mode = None
            self._capture_count = 0
            self._start_time = time.time()

    def to_dict(self) -> dict:
        return {
            "device_name": "SIMULATED_SCANNER_01",
            "status": self.status,
            "uptime_seconds": self.uptime_seconds,
            "firmware_version": self.firmware_version,
            "last_calibration": self.last_calibration,
            "capture_count": self.capture_count,
            "failure_mode": self.failure_mode,
        }


device_state = DeviceState()

"""G2gProbe — glass-to-glass (camera shutter → display) latency probe.

For any video pipeline with a remote consumer (FPV pilot, security operator,
remote teleoperator), the binding constraint is not any single subsystem's
latency — it's the *sum* of capture, ISP, encode, TX, decode, display.

This probe stamps a frame at every stage of the pipeline. Each stage calls
its own method; the final stage emits a record with all stamps and a computed
total. The report aggregator then produces a stacked-bar of per-stage budget
and flags any stage that's eating more than its allocation.

Usage pattern (each pipeline stage calls one method):

    sensor_ingest:   probe.stamp_capture(frame_id, t_capture_ns)
    isp:             probe.stamp_isp_done(frame_id)
    encode:          probe.stamp_encode_done(frame_id)
    comms (tx):      probe.stamp_tx_done(frame_id)
    comms (rx):      probe.stamp_rx_done(frame_id)
    decode:          probe.stamp_decode_done(frame_id)
    display:         probe.stamp_display(frame_id)        # ← this one emits the record

Frame IDs are propagated through pipeline message metadata. For SITL or
other co-located pipelines, all stages share a process tree so we stash
the stamps in a shared dict.

Reference budget for FPV tree-dodging (drone use case, see nightjar ADRs):
    capture + ISP        ~10-20ms
    encode               ~5-15ms     ← VPU spec driver
    tx                   ~10-30ms (good link) / 30-80ms (degraded)
    decode               ~5-15ms
    display + pilot      ~30-50ms
"""

from __future__ import annotations
import threading
import time
from typing import Dict, Optional

from ratchet.schemas import WorkloadRecord
from ratchet.probes.probe_writer import ProbeWriter


class G2gProbe:
    """Tracks per-frame timestamps across the full pipeline."""

    def __init__(
        self,
        writer: ProbeWriter,
        run_id: str,
        phase_provider=None,
        max_inflight: int = 64,
        subsystem: str = "video_encode",
    ) -> None:
        self.writer = writer
        self.run_id = run_id
        self.phase_provider = phase_provider or (lambda: "idle")
        self.max_inflight = max_inflight
        self.subsystem = subsystem

        self._stamps: Dict[int, Dict[str, int]] = {}
        self._lock = threading.Lock()

    def _set(self, frame_id: int, key: str, t_ns: Optional[int] = None) -> None:
        if t_ns is None:
            t_ns = time.perf_counter_ns()
        with self._lock:
            self._stamps.setdefault(frame_id, {})[key] = t_ns
            # Bound memory: drop oldest if too many in flight (e.g. dropped frames)
            if len(self._stamps) > self.max_inflight:
                oldest = min(self._stamps.keys())
                self._stamps.pop(oldest, None)

    def stamp_capture(self, frame_id: int, t_ns: Optional[int] = None) -> None:
        self._set(frame_id, "capture", t_ns)

    def stamp_isp_done(self, frame_id: int, t_ns: Optional[int] = None) -> None:
        self._set(frame_id, "isp_done", t_ns)

    def stamp_encode_done(self, frame_id: int, t_ns: Optional[int] = None) -> None:
        self._set(frame_id, "encode_done", t_ns)

    def stamp_tx_done(self, frame_id: int, t_ns: Optional[int] = None) -> None:
        self._set(frame_id, "tx_done", t_ns)

    def stamp_rx_done(self, frame_id: int, t_ns: Optional[int] = None) -> None:
        self._set(frame_id, "rx_done", t_ns)

    def stamp_decode_done(self, frame_id: int, t_ns: Optional[int] = None) -> None:
        self._set(frame_id, "decode_done", t_ns)

    def stamp_display(self, frame_id: int, t_ns: Optional[int] = None) -> None:
        """Final stage. Emits the record."""
        self._set(frame_id, "display", t_ns)
        with self._lock:
            stamps = self._stamps.pop(frame_id, None)
        if stamps is None or "capture" not in stamps or "display" not in stamps:
            return
        total_ms = (stamps["display"] - stamps["capture"]) / 1e6
        rec = WorkloadRecord(
            run_id=self.run_id,
            subsystem=self.subsystem,
            operation="glass_to_glass",
            phase=self.phase_provider(),
            g2g_capture_ns=stamps.get("capture"),
            g2g_isp_done_ns=stamps.get("isp_done"),
            g2g_encode_done_ns=stamps.get("encode_done"),
            g2g_tx_done_ns=stamps.get("tx_done"),
            g2g_rx_done_ns=stamps.get("rx_done"),
            g2g_decode_done_ns=stamps.get("decode_done"),
            g2g_display_ns=stamps.get("display"),
            g2g_total_ms=total_ms,
        )
        self.writer.emit(rec)

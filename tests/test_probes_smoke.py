"""Smoke tests for ratchet.probes (Op/G2g/NVENC).

GpuProbe is skipped here — exercising it requires NVML access. The probe
already degrades to a no-op when pynvml isn't available; exercise that path
elsewhere if needed.
"""

from __future__ import annotations

import pyarrow.parquet as pq
import pytest

from ratchet.probes import ProbeWriter, OpProbe, NvencProbe, G2gProbe


# ──────────────────────────────────────────────────────────────────────
# OpProbe
# ──────────────────────────────────────────────────────────────────────

class TestOpProbe:
    def test_measure_emits_record_with_nonzero_latency(self, tmp_path):
        w = ProbeWriter(tmp_path, subsystem="alpha", flush_every=1)
        op = OpProbe(w, run_id="r1", subsystem="alpha", operation="my_op")
        with op.measure(precision="bf16", macs=1_000_000):
            pass
        w.close()
        table = pq.read_table(tmp_path / "alpha.parquet")
        assert table.num_rows == 1
        assert table.column("operation").to_pylist() == ["my_op"]
        assert table.column("precision").to_pylist() == ["bf16"]
        assert table.column("macs").to_pylist() == [1_000_000]
        latency_ns = table.column("latency_ns").to_pylist()[0]
        assert latency_ns is not None and latency_ns > 0

    def test_observation_output_fields_propagate(self, tmp_path):
        w = ProbeWriter(tmp_path, subsystem="beta", flush_every=1)
        op = OpProbe(w, run_id="r1", subsystem="beta", operation="op")
        with op.measure(input_shape="1x3x256x256") as obs:
            obs.output_shape = "1x10"
            obs.output_bytes = 40
        w.close()
        table = pq.read_table(tmp_path / "beta.parquet")
        assert table.column("output_shape").to_pylist() == ["1x10"]
        assert table.column("output_bytes").to_pylist() == [40]

    def test_phase_provider_invoked(self, tmp_path):
        w = ProbeWriter(tmp_path, subsystem="g", flush_every=1)
        op = OpProbe(w, run_id="r", subsystem="g", operation="op",
                     phase_provider=lambda: "custom_phase")
        with op.measure():
            pass
        w.close()
        table = pq.read_table(tmp_path / "g.parquet")
        assert table.column("phase").to_pylist() == ["custom_phase"]

    def test_known_extras_populate_typed_field(self, tmp_path):
        # encode_codec is a typed field on WorkloadRecord — passing it as kwargs
        # should land on the typed field, not in extras.
        w = ProbeWriter(tmp_path, subsystem="e", flush_every=1)
        op = OpProbe(w, run_id="r", subsystem="e", operation="encode")
        with op.measure(encode_codec="h265", encode_keyframe=True):
            pass
        w.close()
        table = pq.read_table(tmp_path / "e.parquet")
        assert table.column("encode_codec").to_pylist() == ["h265"]
        assert table.column("encode_keyframe").to_pylist() == [True]


# ──────────────────────────────────────────────────────────────────────
# G2gProbe
# ──────────────────────────────────────────────────────────────────────

class TestG2gProbe:
    def test_full_seven_stage_emits_record(self, tmp_path):
        w = ProbeWriter(tmp_path, subsystem="video", flush_every=1)
        probe = G2gProbe(w, run_id="r1")
        # Stamp all 7 stages with monotonically increasing timestamps
        probe.stamp_capture(frame_id=42, t_ns=1000)
        probe.stamp_isp_done(frame_id=42, t_ns=2000)
        probe.stamp_encode_done(frame_id=42, t_ns=3000)
        probe.stamp_tx_done(frame_id=42, t_ns=4000)
        probe.stamp_rx_done(frame_id=42, t_ns=5000)
        probe.stamp_decode_done(frame_id=42, t_ns=6000)
        probe.stamp_display(frame_id=42, t_ns=10_000)
        w.close()

        table = pq.read_table(tmp_path / "video.parquet")
        assert table.num_rows == 1
        # 9000 ns total = 0.009 ms
        total_ms = table.column("g2g_total_ms").to_pylist()[0]
        assert total_ms == pytest.approx(0.009, rel=0.01)
        assert table.column("operation").to_pylist() == ["glass_to_glass"]

    def test_missing_capture_does_not_emit(self, tmp_path):
        w = ProbeWriter(tmp_path, subsystem="video", flush_every=1)
        probe = G2gProbe(w, run_id="r1")
        # Skip stamp_capture, only stamp display
        probe.stamp_display(frame_id=99, t_ns=10_000)
        w.close()
        # No record should have been emitted
        out = tmp_path / "video.parquet"
        if out.exists():
            table = pq.read_table(out)
            assert table.num_rows == 0

    def test_inflight_bound_drops_oldest(self, tmp_path):
        w = ProbeWriter(tmp_path, subsystem="v", flush_every=100)
        probe = G2gProbe(w, run_id="r", max_inflight=3)
        # Stamp captures for 5 frames — first 2 should be evicted
        for i in range(5):
            probe.stamp_capture(frame_id=i, t_ns=1000 + i)
        # Internal map should hold at most max_inflight=3
        assert len(probe._stamps) <= 3


# ──────────────────────────────────────────────────────────────────────
# NvencProbe
# ──────────────────────────────────────────────────────────────────────

class TestNvencProbe:
    def test_input_to_encoded_emits_one_record(self, tmp_path):
        w = ProbeWriter(tmp_path, subsystem="enc", flush_every=1)
        probe = NvencProbe(w, run_id="r", codec="h265", bitrate_kbps=4000.0)
        probe.on_input_frame()
        probe.on_encoded_frame(size_bytes=12345, keyframe=True,
                               input_shape="1920x1080")
        w.close()

        table = pq.read_table(tmp_path / "enc.parquet")
        assert table.num_rows == 1
        assert table.column("encode_codec").to_pylist() == ["h265"]
        assert table.column("encode_frame_size_bytes").to_pylist() == [12345]
        assert table.column("encode_keyframe").to_pylist() == [True]
        assert table.column("encode_bitrate_kbps").to_pylist() == [pytest.approx(4000.0)]

    def test_subsystem_label_overridable(self, tmp_path):
        w = ProbeWriter(tmp_path, subsystem="custom", flush_every=1)
        probe = NvencProbe(w, run_id="r", subsystem="custom_label")
        probe.on_input_frame()
        probe.on_encoded_frame(size_bytes=100, keyframe=False)
        w.close()
        table = pq.read_table(tmp_path / "custom.parquet")
        assert table.column("subsystem").to_pylist() == ["custom_label"]

"""Smoke tests for ratchet.schemas.workload_record + ratchet.probes.probe_writer.

Verifies the WorkloadRecord roundtrip through Parquet and the ProbeWriter
buffering/flush behavior.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from ratchet.schemas import WorkloadRecord, WORKLOAD_SCHEMA
from ratchet.probes import ProbeWriter


class TestWorkloadRecordDataclass:
    def test_required_fields_have_defaults(self):
        r = WorkloadRecord()
        assert r.record_id != ""    # uuid auto-generated
        assert r.t_wall_ns > 0      # time auto-stamped
        assert r.subsystem == ""
        assert r.phase == "idle"

    def test_to_dict_contains_all_fields(self):
        r = WorkloadRecord(run_id="r1", subsystem="x", operation="op",
                           latency_ns=12345, macs=1000)
        d = r.to_dict()
        assert d["run_id"] == "r1"
        assert d["subsystem"] == "x"
        assert d["operation"] == "op"
        assert d["latency_ns"] == 12345
        assert d["macs"] == 1000
        assert "extras" in d

    def test_record_id_unique_per_record(self):
        a, b = WorkloadRecord(), WorkloadRecord()
        assert a.record_id != b.record_id


class TestWorkloadSchema:
    def test_schema_is_pyarrow_schema(self):
        # WORKLOAD_SCHEMA should be a usable pa.Schema
        assert WORKLOAD_SCHEMA.names is not None
        assert len(WORKLOAD_SCHEMA.names) > 0

    def test_core_fields_present(self):
        names = set(WORKLOAD_SCHEMA.names)
        for required in ("record_id", "run_id", "t_wall_ns", "subsystem",
                         "operation", "phase", "latency_ns", "macs", "precision"):
            assert required in names, f"missing field: {required}"


class TestProbeWriterRoundtrip:
    def test_emit_and_read_back(self, tmp_path):
        w = ProbeWriter(tmp_path, subsystem="alpha", flush_every=1)
        rec = WorkloadRecord(run_id="r1", subsystem="alpha",
                             operation="op", latency_ns=999)
        w.emit(rec)
        w.close()

        out = tmp_path / "alpha.parquet"
        assert out.exists()
        table = pq.read_table(out)
        assert table.num_rows == 1
        df = table.to_pandas() if hasattr(table, "to_pandas") else None
        # to_pandas requires pandas; we don't depend on it in ratchet, so
        # use pyarrow's column access instead.
        assert table.column("subsystem").to_pylist() == ["alpha"]
        assert table.column("latency_ns").to_pylist() == [999]

    def test_buffered_flush(self, tmp_path):
        # flush_every=10, write 5 records, close — all 5 should land
        w = ProbeWriter(tmp_path, subsystem="b", flush_every=10)
        for i in range(5):
            w.emit(WorkloadRecord(run_id="r", subsystem="b",
                                  operation=f"op{i}", latency_ns=i))
        w.close()
        table = pq.read_table(tmp_path / "b.parquet")
        assert table.num_rows == 5
        ops = table.column("operation").to_pylist()
        assert ops == [f"op{i}" for i in range(5)]

    def test_extras_dropped_from_parquet(self, tmp_path):
        # The schema doesn't include `extras`; emit should still succeed.
        w = ProbeWriter(tmp_path, subsystem="c", flush_every=1)
        rec = WorkloadRecord(run_id="r", subsystem="c", operation="op")
        rec.extras["custom_field"] = "value"
        w.emit(rec)
        w.close()
        table = pq.read_table(tmp_path / "c.parquet")
        assert table.num_rows == 1
        assert "extras" not in table.column_names

    def test_context_manager_flushes(self, tmp_path):
        with ProbeWriter(tmp_path, subsystem="d", flush_every=100) as w:
            w.emit(WorkloadRecord(run_id="r", subsystem="d", operation="op"))
        # Exit closed the writer and flushed
        table = pq.read_table(tmp_path / "d.parquet")
        assert table.num_rows == 1

    def test_close_is_idempotent(self, tmp_path):
        w = ProbeWriter(tmp_path, subsystem="e", flush_every=1)
        w.emit(WorkloadRecord(run_id="r", subsystem="e", operation="op"))
        w.close()
        w.close()    # should not raise

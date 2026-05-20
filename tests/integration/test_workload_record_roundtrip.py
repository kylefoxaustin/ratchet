"""Integration: carried-forward v0.1.0 API reachable via the v0.2.0 top-level
namespace, with a WorkloadRecord → Parquet → read-back roundtrip."""
import pyarrow.parquet as pq

# v0.2.0 contract: carried-forward names are importable from `ratchet` directly.
from ratchet import WORKLOAD_SCHEMA, WorkloadRecord
from ratchet.probes import ProbeWriter


class TestCarriedForwardRoundtrip:
    def test_schema_exposed_at_top_level(self):
        assert "subsystem" in set(WORKLOAD_SCHEMA.names)

    def test_record_roundtrip_through_parquet(self, tmp_path):
        with ProbeWriter(tmp_path, subsystem="alpha", flush_every=1) as w:
            w.emit(WorkloadRecord(run_id="r1", subsystem="alpha",
                                  operation="op", latency_ns=1234, macs=5678))
        table = pq.read_table(tmp_path / "alpha.parquet")
        assert table.num_rows == 1
        assert table.column("latency_ns").to_pylist() == [1234]
        assert table.column("macs").to_pylist() == [5678]

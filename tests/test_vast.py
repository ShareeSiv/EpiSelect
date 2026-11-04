# scripts/vast/vast.py without a vast account: query building, estimates, env forwarding and the launch flow
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts", "vast"))
import vast  # noqa: E402

OFFER = dict(
    id=123,
    gpu_name="H100 SXM",
    dph_total=2.4,
    reliability=0.995,
    cuda_max_good=12.8,
    inet_down=900,
    disk_space=500,
    geolocation="US",
    inet_down_cost=0.002,  # $/GB
    inet_up_cost=0.004,
)


def test_offer_query_parses_with_the_sdks_fields():
    q = vast.offer_query(["H100 SXM", "RTX 4090"], 0.98, 200, max_price=3.0, max_bw_cost=2.048)
    assert "on-demand" not in q and "bid" not in q
    pq = pytest.importorskip("vastai.api.query")
    parsed = pq.parse_query(q, {}, pq.offers_fields, pq.offers_alias, pq.offers_mult)
    assert parsed["gpu_name"] == {"in": ["H100 SXM", "RTX 4090"]}
    assert parsed["num_gpus"] == {"eq": "1"} and parsed["dph_total"] == {"lte": "3.0"}
    assert parsed["cuda_max_good"] == {"gte": "12.6"} and parsed["compute_cap"] == {"gte": "800"}
    assert parsed["inet_down_cost"] == {"lte": "0.002000"} and parsed["inet_up_cost"] == {"lte": "0.002000"}


def test_estimate_and_table():
    hours, cost = vast.estimate(OFFER, build_data=False)
    assert hours == pytest.approx(vast.TOKENS / 400_000 / 3600 + vast.SETUP_HOURS)
    bw = vast.DOWN_GB["pull"] * 0.002 + vast.UP_GB * 0.004
    assert vast.bandwidth_cost(OFFER, False) == pytest.approx(bw)
    assert cost == pytest.approx(hours * 2.4 + bw)
    assert vast.estimate(OFFER, build_data=True)[0] == pytest.approx(hours + vast.BUILD_DATA_HOURS)
    assert vast.estimate({**OFFER, "gpu_name": "GTX 1080"}, False) == (None, None)
    assert "H100 SXM" in vast.format_offers([OFFER], False)


def test_forwarded_env_only_takes_known_credentials():
    env = vast.forwarded_env({"WANDB_API_KEY": "k", "HF_TOKEN": "", "RCLONE_CONFIG_R2_TYPE": "s3", "HOME": "/root"})
    assert env == {"WANDB_API_KEY": "k", "RCLONE_CONFIG_R2_TYPE": "s3"}


def test_onstart_pins_the_commit():
    cmd = vast.onstart_cmd("abc123")
    assert f"--branch {vast.BRANCH}" in cmd and "git checkout -q abc123" in cmd
    assert cmd.endswith("bash scripts/vast/onstart.sh")


class FakeVast:
    def __init__(self):
        self.created = None

    def show_ssh_keys(self):
        return [{"id": 1}]

    def search_offers(self, **kw):
        assert kw["query"] == "id=123" and kw["type"] == "on-demand"
        return [OFFER]

    def create_instance(self, offer_id, **kw):
        self.created = (offer_id, kw)
        return {"success": True, "new_contract": 777}


def test_launch_rents_on_demand_and_records_the_id(tmp_path, monkeypatch):
    fake = FakeVast()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(vast, "client", lambda: fake)
    monkeypatch.setattr(vast, "pushed_head", lambda: "deadbeef")
    monkeypatch.setenv("WANDB_API_KEY", "secret")
    vast.main(["launch", "123", "--yes", "--data-src", "r2:bucket/data", "--overrides", "loader.fit_method=paper"])
    offer_id, kw = fake.created
    assert offer_id == 123 and kw["ssh"] and kw["direct"] and kw["disk"] == 200
    assert "bid_price" not in kw and "price" not in kw  # never interruptible
    assert kw["env"]["WANDB_API_KEY"] == "secret" and kw["env"]["DATA_SRC"] == "r2:bucket/data"
    assert kw["env"]["TRAIN_OVERRIDES"] == "loader.fit_method=paper"
    assert "git checkout -q deadbeef" in kw["onstart_cmd"]
    assert open(tmp_path / ".vast_instance").read() == "777"
    assert vast.resolve_id(None) == 777

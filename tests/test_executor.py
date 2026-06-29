"""Unit tests for cascade/executor.py state machine (no GPU / checkpoints)."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from cascade.executor import (
    CascadePlan,
    ExecutorConfig,
    _execute_sequence,
    _wcet_remaining,
    execute_sample,
)
from cascade.kdet import oracle_context


def _fake_models(*ki_names: str) -> dict[str, object]:
    """Placeholder model dict — _run_ki is mocked, so weights never load."""
    return {name: object() for name in ki_names}


def _feasible_rate(latencies_ms: list[float], deadline_ms: float) -> float:
    """Match run_cascade.py: fraction of samples with latency_ms <= D."""
    if not latencies_ms:
        return 0.0
    return sum(1 for ms in latencies_ms if ms <= deadline_ms) / len(latencies_ms)


def _override_rate(flags: list[bool]) -> float:
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0


# Scripted val samples: different cascade paths → different fixed latencies (table WCET).
MOCK_SCENARIOS: list[dict] = [
    {
        "name": "branch_k0_suv_k4",
        "sequence": ["K0", "K2", "Kdet"],
        "specialized": {"K0:suv": ["K4", "Kdet"]},
        "outcomes": {"K0": ("suv", 1.0), "K4": ("gle350", 1.0)},
        "true_global": "gle350",
        "models": ("K0", "K2", "K4"),
    },
    {
        "name": "global_k2_early",
        "sequence": ["K0", "K2", "Kdet"],
        "specialized": {},
        "outcomes": {"K0": ("IDK", 1.0), "K2": ("cx30", 1.0)},
        "true_global": "cx30",
        "models": ("K0", "K2"),
    },
    {
        "name": "all_idk_kdet",
        "sequence": ["K0", "K1", "Kdet"],
        "specialized": {},
        "outcomes": {"K0": ("IDK", 1.0), "K1": ("IDK", 1.0)},
        "true_global": "mustang",
        "models": ("K0", "K1"),
    },
    {
        "name": "coupe_branch_k6",
        "sequence": ["K0", "Kdet"],
        "specialized": {"K0:coupe": ["K6", "Kdet"]},
        "outcomes": {"K0": ("coupe", 1.0), "K6": ("miata", 1.0)},
        "true_global": "miata",
        "models": ("K0", "K6"),
    },
]

# Deadlines for sweep (ms). Sorted ascending — property tests walk this list.
DEADLINE_SWEEP_MS = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 75.0, 100.0, 125.0, 150.0, 200.0, 500.0]


def _run_mock_scenario(
    scenario: dict,
    wcet: dict[str, float],
    device: torch.device,
    mic_t: torch.Tensor,
    geo_t: torch.Tensor,
    *,
    deadline_ms: float | None,
    deadline_guard: bool,
) -> tuple[float, bool]:
    """Execute one scripted scenario; return (latency_ms, deadline_override)."""
    config = ExecutorConfig(
        timing_mode="table",
        wcet_ms=wcet,
        deadline_ms=deadline_ms,
        deadline_guard=deadline_guard,
        kdet_sleep=False,
        kdet=oracle_context(sleep=False, wcet_ms=wcet.get("Kdet", 100.0)),
    )
    with patch("cascade.executor._run_ki") as mock_run_ki:
        outcomes = scenario["outcomes"]
        mock_run_ki.side_effect = lambda ki, *args, **kwargs: outcomes[ki]
        _, _, _, deadline_override, latency_ms, _ = _execute_sequence(
            list(scenario["sequence"]),
            _fake_models(*scenario["models"]),
            None,
            mic_t,
            geo_t,
            device,
            scenario["true_global"],
            dict(scenario["specialized"]),
            config,
            sample_key=0,
        )
    return latency_ms, deadline_override


class TestWcetRemaining(unittest.TestCase):
    def test_sums_from_start_index(self) -> None:
        wcet = {"K0": 10.0, "K2": 20.0, "Kdet": 100.0}
        self.assertAlmostEqual(_wcet_remaining(["K0", "K2", "Kdet"], 0, wcet), 130.0)
        self.assertAlmostEqual(_wcet_remaining(["K0", "K2", "Kdet"], 1, wcet), 120.0)


class TestExecuteSequence(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.mic_t = torch.zeros(1, 8, 8)
        self.geo_t = torch.zeros(1, 8, 8)
        self.registry = None
        self.wcet = {
            "K0": 10.0,
            "K1": 15.0,
            "K2": 20.0,
            "K4": 5.0,
            "K6": 7.0,
            "Kdet": 100.0,
        }
        self.config = ExecutorConfig(
            timing_mode="table",
            wcet_ms=self.wcet,
            kdet_sleep=False,
            kdet=oracle_context(sleep=False, wcet_ms=self.wcet["Kdet"]),
        )

    @patch("cascade.executor._run_ki")
    def test_k0_suv_k4_gle350_branch(self, mock_run_ki) -> None:
        """K0 defers to suv → specialized K4 resolves gle350 (no Kdet)."""
        mock_run_ki.side_effect = lambda ki, *args, **kwargs: {
            "K0": ("suv", 1.0),
            "K4": ("gle350", 1.0),
        }[ki]

        pred, fired, hit_kdet, deadline_override, latency_ms, branch_key = _execute_sequence(
            ["K0", "K2", "Kdet"],
            _fake_models("K0", "K2", "K4"),
            self.registry,
            self.mic_t,
            self.geo_t,
            self.device,
            "gle350",
            {"K0:suv": ["K4", "Kdet"]},
            self.config,
            sample_key=0,
        )

        self.assertEqual(pred, "gle350")
        self.assertEqual(fired, ["K0", "K4"])
        self.assertFalse(hit_kdet)
        self.assertFalse(deadline_override)
        self.assertEqual(branch_key, "K0:suv")
        self.assertAlmostEqual(latency_ms, 15.0)  # wcet K0 + K4

    @patch("cascade.executor._run_ki")
    def test_global_stop_on_k2(self, mock_run_ki) -> None:
        """K0 IDK → K2 returns global cx30 and stops before Kdet."""
        mock_run_ki.side_effect = lambda ki, *args, **kwargs: {
            "K0": ("IDK", 1.0),
            "K2": ("cx30", 1.0),
        }[ki]

        pred, fired, hit_kdet, _, latency_ms, branch_key = _execute_sequence(
            ["K0", "K2", "Kdet"],
            _fake_models("K0", "K2"),
            self.registry,
            self.mic_t,
            self.geo_t,
            self.device,
            "cx30",
            {},
            self.config,
            sample_key=0,
        )

        self.assertEqual(pred, "cx30")
        self.assertEqual(fired, ["K0", "K2"])
        self.assertFalse(hit_kdet)
        self.assertIsNone(branch_key)
        self.assertAlmostEqual(latency_ms, 30.0)

    @patch("cascade.executor._run_ki")
    def test_all_idk_falls_through_to_kdet(self, mock_run_ki) -> None:
        """Every Ki IDK → plan reaches Kdet stub."""
        mock_run_ki.return_value = ("IDK", 1.0)

        pred, fired, hit_kdet, deadline_override, latency_ms, _ = _execute_sequence(
            ["K0", "K1", "Kdet"],
            _fake_models("K0", "K1"),
            self.registry,
            self.mic_t,
            self.geo_t,
            self.device,
            "mustang",
            {},
            self.config,
            sample_key=0,
        )

        self.assertEqual(pred, "mustang")
        self.assertEqual(fired, ["K0", "K1", "Kdet"])
        self.assertTrue(hit_kdet)
        self.assertFalse(deadline_override)
        self.assertAlmostEqual(latency_ms, 125.0)  # 10 + 15 + 100

    @patch("cascade.executor._run_ki")
    def test_deadline_guard_shortcuts_to_kdet(self, mock_run_ki) -> None:
        """D_remain < WCET(remainder) → skip Ki chain, invoke Kdet immediately."""
        config = ExecutorConfig(
            timing_mode="table",
            wcet_ms=self.wcet,
            deadline_ms=50.0,
            deadline_guard=True,
            kdet_sleep=False,
            kdet=oracle_context(sleep=False, wcet_ms=self.wcet["Kdet"]),
        )

        pred, fired, hit_kdet, deadline_override, latency_ms, _ = _execute_sequence(
            ["K0", "K2", "Kdet"],
            _fake_models("K0", "K2"),
            self.registry,
            self.mic_t,
            self.geo_t,
            self.device,
            "miata",
            {},
            config,
            sample_key=0,
        )

        self.assertEqual(pred, "miata")
        self.assertEqual(fired, ["Kdet"])
        self.assertTrue(hit_kdet)
        self.assertTrue(deadline_override)
        self.assertAlmostEqual(latency_ms, 100.0)
        mock_run_ki.assert_not_called()

    @patch("cascade.executor._run_ki")
    def test_execute_sample_wraps_plan(self, mock_run_ki) -> None:
        """execute_sample() wires CascadePlan.initial_cascade + specialized dict."""
        mock_run_ki.side_effect = lambda ki, *args, **kwargs: {
            "K0": ("coupe", 1.0),
            "K6": ("mustang", 1.0),
        }[ki]

        plan = CascadePlan(
            initial_cascade=["K0", "Kdet"],
            specialized_cascades={"K0:coupe": ["K6", "Kdet"]},
        )
        trace = execute_sample(
            plan,
            _fake_models("K0", "K6"),
            self.registry,
            self.mic_t,
            self.geo_t,
            self.device,
            "mustang",
            self.config,
        )

        self.assertEqual(trace.prediction, "mustang")
        self.assertTrue(trace.correct)
        self.assertEqual(trace.fired_kis, ["K0", "K6"])
        self.assertEqual(trace.branch_key, "K0:coupe")


class TestDeadlineSweep(unittest.TestCase):
    """Parametrized deadline sweeps for Pareto / schedulability sanity checks."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.mic_t = torch.zeros(1, 8, 8)
        self.geo_t = torch.zeros(1, 8, 8)
        self.wcet = {
            "K0": 10.0,
            "K1": 15.0,
            "K2": 20.0,
            "K4": 5.0,
            "K6": 7.0,
            "Kdet": 100.0,
        }

    def test_feasible_rate_monotonic_when_guard_off(self) -> None:
        """
        With guard disabled, each sample has fixed latency; more D never helps fewer samples.

        Latencies for MOCK_SCENARIOS (table WCET): 15, 30, 125, 17 ms.
        """
        latencies = [
            _run_mock_scenario(
                s, self.wcet, self.device, self.mic_t, self.geo_t,
                deadline_ms=None,
                deadline_guard=False,
            )[0]
            for s in MOCK_SCENARIOS
        ]
        self.assertEqual(sorted(latencies), [15.0, 17.0, 30.0, 125.0])

        rates: list[float] = []
        for deadline_ms in DEADLINE_SWEEP_MS:
            rate = _feasible_rate(latencies, deadline_ms)
            rates.append(rate)
            with self.subTest(deadline_ms=deadline_ms, feasible_rate=rate):
                self.assertGreaterEqual(rate, 0.0)
                self.assertLessEqual(rate, 1.0)

        for i in range(len(rates) - 1):
            d_lo, d_hi = DEADLINE_SWEEP_MS[i], DEADLINE_SWEEP_MS[i + 1]
            with self.subTest(d_lo=d_lo, d_hi=d_hi, rate_lo=rates[i], rate_hi=rates[i + 1]):
                self.assertLessEqual(
                    rates[i],
                    rates[i + 1],
                    msg=f"feasible_rate must be non-decreasing: D={d_lo} -> D={d_hi}",
                )

        # Spot-check: below min latency → 0; above max → 1
        self.assertEqual(_feasible_rate(latencies, 0.0), 0.0)
        self.assertEqual(_feasible_rate(latencies, 125.0), 1.0)
        self.assertEqual(_feasible_rate(latencies, 500.0), 1.0)

    def test_feasible_rate_sweep_per_deadline_guard_off(self) -> None:
        """subTest matrix: each deadline in sweep reports expected feasible count."""
        latencies = [15.0, 17.0, 30.0, 125.0]
        expected_feasible = {
            0.0: 0,
            15.0: 1,
            17.0: 2,
            30.0: 3,
            125.0: 4,
            500.0: 4,
        }
        for deadline_ms, want in expected_feasible.items():
            with self.subTest(deadline_ms=deadline_ms):
                got = round(_feasible_rate(latencies, deadline_ms) * len(latencies))
                self.assertEqual(got, want)

    def test_override_rate_non_increasing_when_guard_on(self) -> None:
        """
        Larger D_system → fewer (or equal) deadline overrides across the same scenarios.

        Guard changes paths at tight D, so we test override_rate monotonicity, not latency.
        """
        override_rates: list[float] = []
        for deadline_ms in DEADLINE_SWEEP_MS:
            overrides = [
                _run_mock_scenario(
                    s, self.wcet, self.device, self.mic_t, self.geo_t,
                    deadline_ms=deadline_ms,
                    deadline_guard=True,
                )[1]
                for s in MOCK_SCENARIOS
            ]
            rate = _override_rate(overrides)
            override_rates.append(rate)
            with self.subTest(deadline_ms=deadline_ms, override_rate=rate):
                self.assertGreaterEqual(rate, 0.0)
                self.assertLessEqual(rate, 1.0)

        for i in range(len(override_rates) - 1):
            d_lo, d_hi = DEADLINE_SWEEP_MS[i], DEADLINE_SWEEP_MS[i + 1]
            with self.subTest(d_lo=d_lo, d_hi=d_hi):
                self.assertGreaterEqual(
                    override_rates[i],
                    override_rates[i + 1],
                    msg=f"override_rate must be non-increasing: D={d_lo} -> D={d_hi}",
                )

        # Tight deadline: most samples forced to Kdet shortcut
        self.assertEqual(override_rates[0], 1.0)
        # Loose deadline: natural cascade paths, no overrides
        self.assertEqual(override_rates[-1], 0.0)


if __name__ == "__main__":
    unittest.main()

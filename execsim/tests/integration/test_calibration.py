"""Integration test for calibration grid search."""

from execsim.auction.calibration import calibrate_reserve
from execsim.config import Settings


class TestCalibration:
    """Tests for reserve-price calibration."""

    def test_returns_result(self):
        settings = Settings(sim_num_steps=50)
        result = calibrate_reserve(
            held_out_seeds=[42, 43],
            n_bidders=3,
            grid_max_bps=10,
            grid_step_bps=2,
            allocation_floor=0.0,
            settings=settings,
        )
        assert result.optimal_reserve_bps >= 0
        assert len(result.grid) > 0

    def test_grid_covers_range(self):
        settings = Settings(sim_num_steps=50)
        result = calibrate_reserve(
            held_out_seeds=[42],
            n_bidders=3,
            grid_max_bps=10,
            grid_step_bps=2,
            allocation_floor=0.0,
            settings=settings,
        )
        reserves = [gp.reserve_bps for gp in result.grid]
        assert 0.0 in reserves
        assert 10.0 in reserves

    def test_allocation_floor_filters(self):
        """High floor should reduce the set of feasible points."""
        settings = Settings(sim_num_steps=50)
        result_low = calibrate_reserve(
            held_out_seeds=[42],
            n_bidders=3,
            grid_max_bps=20,
            grid_step_bps=5,
            allocation_floor=0.0,
            settings=settings,
        )
        result_high = calibrate_reserve(
            held_out_seeds=[42],
            n_bidders=3,
            grid_max_bps=20,
            grid_step_bps=5,
            allocation_floor=0.99,
            settings=settings,
        )
        feasible_low = sum(1 for gp in result_low.grid if gp.feasible)
        feasible_high = sum(1 for gp in result_high.grid if gp.feasible)
        assert feasible_high <= feasible_low

    def test_deterministic(self):
        settings = Settings(sim_num_steps=50)
        r1 = calibrate_reserve([42], 3, 10, 2, 0.0, settings)
        r2 = calibrate_reserve([42], 3, 10, 2, 0.0, settings)
        assert r1.optimal_reserve_bps == r2.optimal_reserve_bps
        assert r1.optimal_revenue_bps == r2.optimal_revenue_bps

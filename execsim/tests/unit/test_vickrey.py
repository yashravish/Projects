"""Unit tests for the Vickrey auction.

Tests verify:
  1. Correct second-price payment.
  2. Truthful bidding is weakly dominant (on constructed inputs).
  3. Reserve-price filtering.
  4. Tie-breaking by lowest bidder index.
  5. No allocation when all bids below reserve.
"""

import numpy as np
import pytest

from execsim.auction.vickrey import (
    AuctionOutcome,
    Bid,
    generate_synthetic_bids,
    run_vickrey_auction,
)


class TestVickreyMechanism:
    """Core Vickrey auction property tests."""

    def test_second_price_payment(self):
        """Winner pays the second-highest bid."""
        bids = [[
            Bid(0, 0, 20.0),
            Bid(0, 1, 15.0),
            Bid(0, 2, 10.0),
        ]]
        result = run_vickrey_auction(bids, reserve_price_bps=0.0)
        assert result.outcomes[0].allocated
        assert result.outcomes[0].winner_index == 0
        assert result.outcomes[0].payment_bps == 15.0

    def test_payment_is_reserve_when_single_bidder(self):
        """With one bidder above reserve, payment = reserve."""
        bids = [[Bid(0, 0, 20.0)]]
        result = run_vickrey_auction(bids, reserve_price_bps=5.0)
        assert result.outcomes[0].allocated
        assert result.outcomes[0].payment_bps == 5.0

    def test_reserve_filters_low_bids(self):
        """Bids below reserve are filtered out."""
        bids = [[
            Bid(0, 0, 3.0),
            Bid(0, 1, 2.0),
        ]]
        result = run_vickrey_auction(bids, reserve_price_bps=5.0)
        assert not result.outcomes[0].allocated
        assert result.outcomes[0].payment_bps == 0.0

    def test_tie_breaking_by_index(self):
        """Equal bids are broken by lowest bidder index."""
        bids = [[
            Bid(0, 2, 10.0),
            Bid(0, 0, 10.0),
            Bid(0, 1, 10.0),
        ]]
        result = run_vickrey_auction(bids, reserve_price_bps=0.0)
        assert result.outcomes[0].winner_index == 0

    def test_payment_at_least_reserve(self):
        """Payment is max(second bid, reserve), so it's >= reserve when allocated."""
        bids = [[
            Bid(0, 0, 20.0),
            Bid(0, 1, 3.0),
        ]]
        result = run_vickrey_auction(bids, reserve_price_bps=10.0)
        assert result.outcomes[0].allocated
        # second bid is 3.0 < reserve 10.0, so payment = 10.0
        assert result.outcomes[0].payment_bps == 10.0

    def test_allocation_rate(self):
        bids = [
            [Bid(0, 0, 10.0)],
            [Bid(1, 0, 1.0)],  # below reserve
        ]
        result = run_vickrey_auction(bids, reserve_price_bps=5.0)
        assert result.num_allocated == 1
        assert result.allocation_rate == 0.5

    def test_total_revenue_sum(self):
        bids = [
            [Bid(0, 0, 20.0), Bid(0, 1, 15.0)],
            [Bid(1, 0, 10.0), Bid(1, 1, 8.0)],
        ]
        result = run_vickrey_auction(bids, reserve_price_bps=0.0)
        expected = 15.0 + 8.0  # second prices
        assert abs(result.total_revenue_bps - expected) < 1e-10

    def test_multiple_opportunities(self):
        """Each opportunity is auctioned independently."""
        bids = [
            [Bid(0, 0, 20.0), Bid(0, 1, 10.0)],
            [Bid(1, 0, 5.0), Bid(1, 1, 3.0)],
        ]
        result = run_vickrey_auction(bids, reserve_price_bps=0.0)
        assert len(result.outcomes) == 2
        assert result.outcomes[0].winner_index == 0
        assert result.outcomes[1].winner_index == 0

    def test_empty_input(self):
        result = run_vickrey_auction([], reserve_price_bps=0.0)
        assert result.num_opportunities == 0
        assert result.total_revenue_bps == 0.0


class TestVickreyIncentiveProperty:
    """Tests that truthful bidding is weakly dominant.

    For constructed inputs, verify that the truthful bidder (bidder 0)
    cannot improve their payoff by deviating from their true value.

    Payoff = value - payment if winner, else 0.
    """

    def test_truthful_wins_and_overshading_doesnt_help(self):
        """Bidding above true value doesn't increase payoff."""
        true_value = 15.0

        # Truthful bid
        bids_truthful = [[
            Bid(0, 0, true_value),
            Bid(0, 1, 10.0),
        ]]
        r_truthful = run_vickrey_auction(bids_truthful, reserve_price_bps=0.0)
        payoff_truthful = true_value - r_truthful.outcomes[0].payment_bps

        # Overbid
        bids_over = [[
            Bid(0, 0, 25.0),
            Bid(0, 1, 10.0),
        ]]
        r_over = run_vickrey_auction(bids_over, reserve_price_bps=0.0)
        payoff_over = true_value - r_over.outcomes[0].payment_bps

        # Payoff should be the same (both win, same second price)
        assert abs(payoff_truthful - payoff_over) < 1e-10

    def test_truthful_loses_and_undershading_doesnt_help(self):
        """If truthful bid loses, underbidding also loses (weakly worse)."""
        true_value = 8.0

        # Truthful bid (loses to bidder 1 who bids 20)
        bids_truthful = [[
            Bid(0, 0, true_value),
            Bid(0, 1, 20.0),
        ]]
        r_truthful = run_vickrey_auction(bids_truthful, reserve_price_bps=0.0)
        # Bidder 0 loses, payoff = 0
        assert r_truthful.outcomes[0].winner_index == 1

        # Underbid
        bids_under = [[
            Bid(0, 0, 5.0),
            Bid(0, 1, 20.0),
        ]]
        r_under = run_vickrey_auction(bids_under, reserve_price_bps=0.0)
        # Still loses, payoff = 0
        assert r_under.outcomes[0].winner_index == 1

    def test_shading_can_lose_opportunity(self):
        """Underbidding below the second-highest bid can cause a loss."""
        true_value = 15.0

        # Truthful: wins, pays 12
        bids_truthful = [[
            Bid(0, 0, true_value),
            Bid(0, 1, 12.0),
        ]]
        r_t = run_vickrey_auction(bids_truthful, reserve_price_bps=0.0)
        assert r_t.outcomes[0].winner_index == 0
        payoff_truthful = true_value - r_t.outcomes[0].payment_bps
        assert payoff_truthful > 0  # profit = 3

        # Shade below second-highest: now loses
        bids_shaded = [[
            Bid(0, 0, 11.0),  # below competitor's 12
            Bid(0, 1, 12.0),
        ]]
        r_s = run_vickrey_auction(bids_shaded, reserve_price_bps=0.0)
        assert r_s.outcomes[0].winner_index == 1
        # Payoff = 0 (lost)
        assert payoff_truthful > 0  # truthful was strictly better


class TestSyntheticBids:
    """Tests for synthetic bid generation."""

    def test_correct_count(self):
        bids = generate_synthetic_bids([10.0, 20.0], 5, np.random.default_rng(42))
        assert len(bids) == 2
        assert all(len(b) == 5 for b in bids)

    def test_first_bidder_truthful(self):
        bids = generate_synthetic_bids([10.0], 3, np.random.default_rng(42))
        assert bids[0][0].value_bps == 10.0
        assert bids[0][0].bidder_index == 0

    def test_deterministic(self):
        b1 = generate_synthetic_bids([10.0], 3, np.random.default_rng(42))
        b2 = generate_synthetic_bids([10.0], 3, np.random.default_rng(42))
        for a, b in zip(b1[0], b2[0]):
            assert a.value_bps == b.value_bps

    def test_invalid_n_bidders(self):
        with pytest.raises(ValueError, match="n_bidders must be >= 1"):
            generate_synthetic_bids([10.0], 0, np.random.default_rng(42))

#!/usr/bin/env python3
def analyze_cash_flow(cash_flows):
    """Compute a simple average cash flow score."""
    if not cash_flows:
        return 0.0
    return sum(cash_flows) / len(cash_flows)

def main():
    # Simulated cash flow data for an applicant.
    cash_flows = [1000, 1500, 1200, 1300, 1100]
    avg_cash_flow = analyze_cash_flow(cash_flows)
    print("Average Cash Flow:", avg_cash_flow)

if __name__ == "__main__":
    main()

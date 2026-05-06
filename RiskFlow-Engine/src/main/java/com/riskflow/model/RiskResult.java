package com.riskflow.model;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.Instant;

@Entity
@Table(name = "risk_results", indexes = @Index(name = "idx_risk_portfolio_time", columnList = "portfolio_id, timestamp"))
public class RiskResult {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @ManyToOne(optional = false, fetch = FetchType.LAZY) @JoinColumn(name = "portfolio_id")
    private Portfolio portfolio;
    @Column(nullable = false) private Instant timestamp = Instant.now();
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal totalMarketValue;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal totalPnL;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal equityExposure;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal fixedIncomeExposure;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal delta;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal dv01;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal var95;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal stressEquityDown5;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal stressEquityDown10;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal stressRatesUp25bps;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal stressRatesUp100bps;
    @Column(nullable = false, precision = 12, scale = 6) private BigDecimal concentrationPct = BigDecimal.ZERO;
    public Long getId() { return id; }
    public Portfolio getPortfolio() { return portfolio; }
    public void setPortfolio(Portfolio portfolio) { this.portfolio = portfolio; }
    public Instant getTimestamp() { return timestamp; }
    public void setTimestamp(Instant timestamp) { this.timestamp = timestamp; }
    public BigDecimal getTotalMarketValue() { return totalMarketValue; }
    public void setTotalMarketValue(BigDecimal totalMarketValue) { this.totalMarketValue = totalMarketValue; }
    public BigDecimal getTotalPnL() { return totalPnL; }
    public void setTotalPnL(BigDecimal totalPnL) { this.totalPnL = totalPnL; }
    public BigDecimal getEquityExposure() { return equityExposure; }
    public void setEquityExposure(BigDecimal equityExposure) { this.equityExposure = equityExposure; }
    public BigDecimal getFixedIncomeExposure() { return fixedIncomeExposure; }
    public void setFixedIncomeExposure(BigDecimal fixedIncomeExposure) { this.fixedIncomeExposure = fixedIncomeExposure; }
    public BigDecimal getDelta() { return delta; }
    public void setDelta(BigDecimal delta) { this.delta = delta; }
    public BigDecimal getDv01() { return dv01; }
    public void setDv01(BigDecimal dv01) { this.dv01 = dv01; }
    public BigDecimal getVar95() { return var95; }
    public void setVar95(BigDecimal var95) { this.var95 = var95; }
    public BigDecimal getStressEquityDown5() { return stressEquityDown5; }
    public void setStressEquityDown5(BigDecimal stressEquityDown5) { this.stressEquityDown5 = stressEquityDown5; }
    public BigDecimal getStressEquityDown10() { return stressEquityDown10; }
    public void setStressEquityDown10(BigDecimal stressEquityDown10) { this.stressEquityDown10 = stressEquityDown10; }
    public BigDecimal getStressRatesUp25bps() { return stressRatesUp25bps; }
    public void setStressRatesUp25bps(BigDecimal stressRatesUp25bps) { this.stressRatesUp25bps = stressRatesUp25bps; }
    public BigDecimal getStressRatesUp100bps() { return stressRatesUp100bps; }
    public void setStressRatesUp100bps(BigDecimal stressRatesUp100bps) { this.stressRatesUp100bps = stressRatesUp100bps; }
    public BigDecimal getConcentrationPct() { return concentrationPct; }
    public void setConcentrationPct(BigDecimal concentrationPct) { this.concentrationPct = concentrationPct; }
}

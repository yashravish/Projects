package com.riskflow.model;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDate;

@Entity
@Table(name = "eod_reports", indexes = @Index(name = "idx_eod_portfolio_date", columnList = "portfolio_id, reportDate"))
public class EodReport {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @ManyToOne(optional = false, fetch = FetchType.LAZY) @JoinColumn(name = "portfolio_id")
    private Portfolio portfolio;
    @Column(nullable = false) private LocalDate reportDate;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal totalExposure;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal totalPnL;
    @Column(nullable = false, precision = 19, scale = 4) private BigDecimal var95;
    @Column(nullable = false) private String largestRiskContributor;
    @Column(nullable = false, columnDefinition = "TEXT") private String summaryJson;
    public Long getId() { return id; }
    public Portfolio getPortfolio() { return portfolio; }
    public void setPortfolio(Portfolio portfolio) { this.portfolio = portfolio; }
    public LocalDate getReportDate() { return reportDate; }
    public void setReportDate(LocalDate reportDate) { this.reportDate = reportDate; }
    public BigDecimal getTotalExposure() { return totalExposure; }
    public void setTotalExposure(BigDecimal totalExposure) { this.totalExposure = totalExposure; }
    public BigDecimal getTotalPnL() { return totalPnL; }
    public void setTotalPnL(BigDecimal totalPnL) { this.totalPnL = totalPnL; }
    public BigDecimal getVar95() { return var95; }
    public void setVar95(BigDecimal var95) { this.var95 = var95; }
    public String getLargestRiskContributor() { return largestRiskContributor; }
    public void setLargestRiskContributor(String largestRiskContributor) { this.largestRiskContributor = largestRiskContributor; }
    public String getSummaryJson() { return summaryJson; }
    public void setSummaryJson(String summaryJson) { this.summaryJson = summaryJson; }
}

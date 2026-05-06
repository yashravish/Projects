package com.riskflow.service;

import com.riskflow.exception.NotFoundException;
import com.riskflow.model.*;
import com.riskflow.repository.*;
import org.springframework.stereotype.Service;
import java.math.BigDecimal; import java.time.LocalDate; import java.util.*; import java.util.stream.Collectors;

@Service
public class EodReportService {
    private final PortfolioService portfolios; private final RiskCalculationService riskService; private final EodReportRepository reports; private final TradeRepository trades; private final MarketPriceRepository prices; private final AuditService audit;
    public EodReportService(PortfolioService portfolios, RiskCalculationService riskService, EodReportRepository reports, TradeRepository trades, MarketPriceRepository prices, AuditService audit) { this.portfolios = portfolios; this.riskService = riskService; this.reports = reports; this.trades = trades; this.prices = prices; this.audit = audit; }
    public EodReport generate(Long portfolioId) { Portfolio p = portfolios.get(portfolioId); RiskResult r; try { r = riskService.latest(portfolioId); } catch (NotFoundException ex) { r = riskService.calculate(portfolioId); } String largest = largestContributor(portfolioId); EodReport e = new EodReport(); e.setPortfolio(p); e.setReportDate(LocalDate.now()); e.setTotalExposure(r.getTotalMarketValue()); e.setTotalPnL(r.getTotalPnL()); e.setVar95(r.getVar95()); e.setLargestRiskContributor(largest); e.setSummaryJson(String.format("{\"portfolioId\":%d,\"reportDate\":\"%s\",\"largestRiskContributor\":\"%s\",\"var95\":%s,\"concentrationPct\":%s}", portfolioId, e.getReportDate(), largest, r.getVar95(), r.getConcentrationPct())); EodReport saved = reports.save(e); audit.log("EOD_REPORT_GENERATED", "Portfolio", portfolioId, "Generated EOD risk report"); return saved; }
    public EodReport latest(Long portfolioId) { return reports.findFirstByPortfolioIdOrderByReportDateDesc(portfolioId).orElseThrow(() -> new NotFoundException("No EOD report for portfolio: " + portfolioId)); }
    private String largestContributor(Long portfolioId) { Map<Instrument, List<Trade>> m = trades.findByPortfolioId(portfolioId).stream().collect(Collectors.groupingBy(Trade::getInstrument)); BigDecimal max = BigDecimal.ZERO; String symbol = "NONE"; for (var e : m.entrySet()) { var mp = prices.findFirstByInstrumentIdOrderByTimestampDesc(e.getKey().getId()); if (mp.isEmpty()) continue; BigDecimal qty = e.getValue().stream().map(t -> t.getSide() == TradeSide.BUY ? t.getQuantity() : t.getQuantity().negate()).reduce(BigDecimal.ZERO, BigDecimal::add); BigDecimal exp = qty.multiply(mp.get().getPrice()).abs(); if (exp.compareTo(max) > 0) { max = exp; symbol = e.getKey().getSymbol(); } } return symbol; }
}

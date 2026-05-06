package com.riskflow.dto;

import com.riskflow.model.*;
import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;

public final class Responses {
    private Responses() {}
    public record PortfolioResponse(Long id, String name, Instant createdAt) { public static PortfolioResponse from(Portfolio p) { return new PortfolioResponse(p.getId(), p.getName(), p.getCreatedAt()); } }
    public record InstrumentResponse(Long id, String symbol, InstrumentType type, String name, BigDecimal couponRate, LocalDate maturityDate, BigDecimal faceValue) { public static InstrumentResponse from(Instrument i) { return new InstrumentResponse(i.getId(), i.getSymbol(), i.getType(), i.getName(), i.getCouponRate(), i.getMaturityDate(), i.getFaceValue()); } }
    public record TradeResponse(Long id, Long portfolioId, Long instrumentId, String symbol, TradeSide side, BigDecimal quantity, BigDecimal price, Instant tradeTime) { public static TradeResponse from(Trade t) { return new TradeResponse(t.getId(), t.getPortfolio().getId(), t.getInstrument().getId(), t.getInstrument().getSymbol(), t.getSide(), t.getQuantity(), t.getPrice(), t.getTradeTime()); } }
    public record MarketPriceResponse(Long id, Long instrumentId, String symbol, BigDecimal price, BigDecimal yield, Instant timestamp) { public static MarketPriceResponse from(MarketPrice p) { return new MarketPriceResponse(p.getId(), p.getInstrument().getId(), p.getInstrument().getSymbol(), p.getPrice(), p.getYield(), p.getTimestamp()); } }
    public record RiskResponse(Long id, Long portfolioId, Instant timestamp, BigDecimal totalMarketValue, BigDecimal totalPnL, BigDecimal equityExposure, BigDecimal fixedIncomeExposure, BigDecimal delta, BigDecimal dv01, BigDecimal var95, BigDecimal stressEquityDown5, BigDecimal stressEquityDown10, BigDecimal stressRatesUp25bps, BigDecimal stressRatesUp100bps, BigDecimal concentrationPct) { public static RiskResponse from(RiskResult r) { return new RiskResponse(r.getId(), r.getPortfolio().getId(), r.getTimestamp(), r.getTotalMarketValue(), r.getTotalPnL(), r.getEquityExposure(), r.getFixedIncomeExposure(), r.getDelta(), r.getDv01(), r.getVar95(), r.getStressEquityDown5(), r.getStressEquityDown10(), r.getStressRatesUp25bps(), r.getStressRatesUp100bps(), r.getConcentrationPct()); } }
    public record EodReportResponse(Long id, Long portfolioId, LocalDate reportDate, BigDecimal totalExposure, BigDecimal totalPnL, BigDecimal var95, String largestRiskContributor, String summaryJson) { public static EodReportResponse from(EodReport e) { return new EodReportResponse(e.getId(), e.getPortfolio().getId(), e.getReportDate(), e.getTotalExposure(), e.getTotalPnL(), e.getVar95(), e.getLargestRiskContributor(), e.getSummaryJson()); } }
}

package com.riskflow.service;

import com.riskflow.exception.NotFoundException;
import com.riskflow.model.*;
import com.riskflow.repository.*;
import org.slf4j.Logger; import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import java.math.*; import java.time.LocalDate; import java.time.temporal.ChronoUnit; import java.util.*; import java.util.stream.Collectors;

@Service
public class RiskCalculationService {
    private static final Logger log = LoggerFactory.getLogger(RiskCalculationService.class);
    private static final MathContext MC = new MathContext(18, RoundingMode.HALF_UP);
    private final PortfolioService portfolios; private final TradeRepository trades; private final MarketPriceRepository prices; private final RiskResultRepository risks; private final RedisCacheService cache; private final AuditService audit;
    public RiskCalculationService(PortfolioService portfolios, TradeRepository trades, MarketPriceRepository prices, RiskResultRepository risks, RedisCacheService cache, AuditService audit) { this.portfolios = portfolios; this.trades = trades; this.prices = prices; this.risks = risks; this.cache = cache; this.audit = audit; }
    public RiskResult calculate(Long portfolioId) {
        long start = System.nanoTime(); Portfolio p = portfolios.get(portfolioId); List<Trade> tradeList = trades.findByPortfolioId(portfolioId); if (tradeList.isEmpty()) throw new IllegalArgumentException("Portfolio has no trades: " + portfolioId);
        Map<Instrument, List<Trade>> byInstrument = tradeList.stream().collect(Collectors.groupingBy(Trade::getInstrument));
        BigDecimal total = BigDecimal.ZERO, pnl = BigDecimal.ZERO, equity = BigDecimal.ZERO, fi = BigDecimal.ZERO, delta = BigDecimal.ZERO, dv01 = BigDecimal.ZERO, maxAbs = BigDecimal.ZERO; String maxSymbol = "NONE";
        for (Map.Entry<Instrument, List<Trade>> e : byInstrument.entrySet()) {
            Instrument inst = e.getKey(); BigDecimal qty = position(e.getValue()); if (qty.compareTo(BigDecimal.ZERO) == 0) continue;
            BigDecimal avg = averagePrice(e.getValue()); MarketPrice mp = prices.findFirstByInstrumentIdOrderByTimestampDesc(inst.getId()).orElseThrow(() -> new NotFoundException("Missing price for " + inst.getSymbol()));
            BigDecimal mv; BigDecimal instrumentDv01 = BigDecimal.ZERO;
            if (inst.getType() == InstrumentType.EQUITY) { mv = qty.multiply(mp.getPrice(), MC); equity = equity.add(mv, MC); delta = delta.add(mv, MC); }
            else { BigDecimal y = mp.getYield() == null ? BigDecimal.valueOf(0.04) : mp.getYield(); BigDecimal cleanPrice = bondPrice(inst, y); mv = qty.multiply(cleanPrice, MC); instrumentDv01 = bondDv01(inst, y).multiply(qty, MC); fi = fi.add(mv, MC); dv01 = dv01.add(instrumentDv01, MC); }
            total = total.add(mv, MC); pnl = pnl.add(qty.multiply(inst.getType() == InstrumentType.BOND ? mv.divide(qty, MC).subtract(avg, MC) : mp.getPrice().subtract(avg, MC), MC), MC);
            if (mv.abs().compareTo(maxAbs) > 0) { maxAbs = mv.abs(); maxSymbol = inst.getSymbol(); }
        }
        RiskResult r = new RiskResult(); r.setPortfolio(p); r.setTotalMarketValue(scale(total)); r.setTotalPnL(scale(pnl)); r.setEquityExposure(scale(equity)); r.setFixedIncomeExposure(scale(fi)); r.setDelta(scale(delta)); r.setDv01(scale(dv01)); r.setVar95(scale(total.abs().multiply(BigDecimal.valueOf(0.02), MC))); r.setStressEquityDown5(scale(equity.multiply(BigDecimal.valueOf(-0.05), MC))); r.setStressEquityDown10(scale(equity.multiply(BigDecimal.valueOf(-0.10), MC))); r.setStressRatesUp25bps(scale(dv01.multiply(BigDecimal.valueOf(25), MC))); r.setStressRatesUp100bps(scale(dv01.multiply(BigDecimal.valueOf(100), MC))); r.setConcentrationPct(total.compareTo(BigDecimal.ZERO) == 0 ? BigDecimal.ZERO : maxAbs.divide(total.abs(), 6, RoundingMode.HALF_UP).multiply(BigDecimal.valueOf(100)));
        RiskResult saved = risks.save(r); cache.cacheRisk(saved); audit.log("RISK_CALCULATED", "Portfolio", portfolioId, "Calculated risk; largest contributor " + maxSymbol); log.info("risk calculation portfolioId={} latencyMs={}", portfolioId, (System.nanoTime() - start) / 1_000_000.0); return saved;
    }
    public RiskResult latest(Long id) { return risks.findFirstByPortfolioIdOrderByTimestampDesc(id).orElseThrow(() -> new NotFoundException("No risk result for portfolio: " + id)); }
    public List<RiskResult> history(Long id) { return risks.findByPortfolioIdOrderByTimestampDesc(id); }
    public BigDecimal bondPrice(Instrument b, BigDecimal y) { BigDecimal face = b.getFaceValue() == null ? BigDecimal.valueOf(1000) : b.getFaceValue(); BigDecimal c = b.getCouponRate() == null ? BigDecimal.ZERO : b.getCouponRate(); long years = Math.max(1, ChronoUnit.YEARS.between(LocalDate.now(), b.getMaturityDate() == null ? LocalDate.now().plusYears(5) : b.getMaturityDate())); BigDecimal price = BigDecimal.ZERO; for (int t=1; t<=years; t++) { price = price.add(face.multiply(c, MC).divide(BigDecimal.ONE.add(y).pow(t, MC), MC), MC); } return price.add(face.divide(BigDecimal.ONE.add(y).pow((int)years, MC), MC), MC); }
    public BigDecimal bondDv01(Instrument b, BigDecimal y) { return bondPrice(b, y.add(BigDecimal.valueOf(0.0001))).subtract(bondPrice(b, y), MC); }
    private BigDecimal position(List<Trade> ts) { return ts.stream().map(t -> t.getSide() == TradeSide.BUY ? t.getQuantity() : t.getQuantity().negate()).reduce(BigDecimal.ZERO, BigDecimal::add); }
    private BigDecimal averagePrice(List<Trade> ts) { BigDecimal q = ts.stream().map(Trade::getQuantity).reduce(BigDecimal.ZERO, BigDecimal::add); BigDecimal n = ts.stream().map(t -> t.getQuantity().multiply(t.getPrice(), MC)).reduce(BigDecimal.ZERO, BigDecimal::add); return q.compareTo(BigDecimal.ZERO)==0 ? BigDecimal.ZERO : n.divide(q, MC); }
    private BigDecimal scale(BigDecimal v) { return v.setScale(4, RoundingMode.HALF_UP); }
}

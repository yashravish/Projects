package com.riskflow.service;

import com.riskflow.model.*; import com.riskflow.repository.*;
import org.junit.jupiter.api.Test;
import java.math.BigDecimal; import java.time.LocalDate; import java.util.*;
import static org.junit.jupiter.api.Assertions.*; import static org.mockito.Mockito.*;

class RiskCalculationServiceTest {
    @Test void bondPriceAndDv01AreSensible() {
        RiskCalculationService svc = new RiskCalculationService(mock(PortfolioService.class), mock(TradeRepository.class), mock(MarketPriceRepository.class), mock(RiskResultRepository.class), mock(RedisCacheService.class), mock(AuditService.class));
        Instrument b = new Instrument("BOND", InstrumentType.BOND, "Bond", BigDecimal.valueOf(0.05), LocalDate.now().plusYears(5), BigDecimal.valueOf(1000));
        BigDecimal price = svc.bondPrice(b, BigDecimal.valueOf(0.04)); BigDecimal dv01 = svc.bondDv01(b, BigDecimal.valueOf(0.04));
        assertTrue(price.compareTo(BigDecimal.valueOf(1000)) > 0); assertTrue(dv01.compareTo(BigDecimal.ZERO) < 0);
    }
    @Test void calculatesEquityRiskFallbackVar() {
        PortfolioService ps = mock(PortfolioService.class); TradeRepository tr = mock(TradeRepository.class); MarketPriceRepository pr = mock(MarketPriceRepository.class); RiskResultRepository rr = mock(RiskResultRepository.class); RedisCacheService cache = mock(RedisCacheService.class); AuditService audit = mock(AuditService.class);
        Portfolio p = new Portfolio("Book"); Instrument a = new Instrument("AAPL", InstrumentType.EQUITY, "Apple", null, null, null); Trade t = new Trade(); t.setPortfolio(p); t.setInstrument(a); t.setSide(TradeSide.BUY); t.setQuantity(BigDecimal.TEN); t.setPrice(BigDecimal.valueOf(90)); MarketPrice mp = new MarketPrice(); mp.setInstrument(a); mp.setPrice(BigDecimal.valueOf(100));
        when(ps.get(1L)).thenReturn(p); when(tr.findByPortfolioId(1L)).thenReturn(List.of(t)); when(pr.findFirstByInstrumentIdOrderByTimestampDesc(any())).thenReturn(Optional.of(mp)); when(rr.save(any())).thenAnswer(i -> i.getArgument(0));
        RiskResult r = new RiskCalculationService(ps, tr, pr, rr, cache, audit).calculate(1L);
        assertEquals(new BigDecimal("1000.0000"), r.getTotalMarketValue()); assertEquals(new BigDecimal("20.0000"), r.getVar95()); assertEquals(new BigDecimal("100.0000"), r.getTotalPnL());
    }
}

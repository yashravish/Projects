package com.riskflow.service;

import com.riskflow.model.*; import com.riskflow.repository.*;
import org.junit.jupiter.api.Test;
import java.math.BigDecimal; import java.util.*;
import static org.junit.jupiter.api.Assertions.*; import static org.mockito.Mockito.*;

class EodReportServiceTest {
    @Test void generatesReportFromLatestRisk() {
        PortfolioService ps = mock(PortfolioService.class); RiskCalculationService rs = mock(RiskCalculationService.class); EodReportRepository er = mock(EodReportRepository.class); TradeRepository tr = mock(TradeRepository.class); MarketPriceRepository mp = mock(MarketPriceRepository.class); AuditService audit = mock(AuditService.class);
        Portfolio p = new Portfolio("Book"); RiskResult risk = new RiskResult(); risk.setPortfolio(p); risk.setTotalMarketValue(BigDecimal.valueOf(100)); risk.setTotalPnL(BigDecimal.ONE); risk.setVar95(BigDecimal.TEN); risk.setConcentrationPct(BigDecimal.valueOf(50));
        when(ps.get(1L)).thenReturn(p); when(rs.latest(1L)).thenReturn(risk); when(tr.findByPortfolioId(1L)).thenReturn(List.of()); when(er.save(any())).thenAnswer(i -> i.getArgument(0));
        EodReport out = new EodReportService(ps, rs, er, tr, mp, audit).generate(1L);
        assertEquals(BigDecimal.TEN, out.getVar95()); assertTrue(out.getSummaryJson().contains("portfolioId")); verify(audit).log(eq("EOD_REPORT_GENERATED"), eq("Portfolio"), eq(1L), any());
    }
}

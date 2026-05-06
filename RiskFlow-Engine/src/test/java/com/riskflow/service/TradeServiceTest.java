package com.riskflow.service;

import com.riskflow.dto.Requests.TradeRequest; import com.riskflow.model.*; import com.riskflow.repository.TradeRepository;
import org.junit.jupiter.api.Test;
import java.math.BigDecimal;
import static org.junit.jupiter.api.Assertions.*; import static org.mockito.Mockito.*;

class TradeServiceTest {
    @Test void createsTradeWithResolvedPortfolioAndInstrument() {
        TradeRepository repo = mock(TradeRepository.class); PortfolioService ps = mock(PortfolioService.class); InstrumentService is = mock(InstrumentService.class); AuditService audit = mock(AuditService.class);
        Portfolio p = new Portfolio("Book"); Instrument inst = new Instrument("AAPL", InstrumentType.EQUITY, "Apple", null, null, null);
        when(ps.get(1L)).thenReturn(p); when(is.get(2L)).thenReturn(inst); when(repo.save(any())).thenAnswer(i -> i.getArgument(0));
        Trade t = new TradeService(repo, ps, is, audit).create(new TradeRequest(1L, 2L, TradeSide.BUY, BigDecimal.TEN, BigDecimal.valueOf(100), null));
        assertEquals(TradeSide.BUY, t.getSide()); assertEquals(inst, t.getInstrument()); verify(audit).log(eq("TRADE_CREATED"), eq("Trade"), any(), contains("AAPL"));
    }
}

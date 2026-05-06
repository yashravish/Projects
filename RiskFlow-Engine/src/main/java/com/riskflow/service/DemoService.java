package com.riskflow.service;

import com.riskflow.dto.Requests.*;
import com.riskflow.model.*;
import com.riskflow.repository.*;
import org.springframework.stereotype.Service;
import java.math.BigDecimal; import java.time.LocalDate; import java.util.Map;

@Service
public class DemoService {
    private final PortfolioRepository portfolios; private final InstrumentRepository instruments; private final PortfolioService portfolioService; private final InstrumentService instrumentService; private final TradeService tradeService; private final MarketDataService marketData;
    public DemoService(PortfolioRepository portfolios, InstrumentRepository instruments, PortfolioService portfolioService, InstrumentService instrumentService, TradeService tradeService, MarketDataService marketData) { this.portfolios = portfolios; this.instruments = instruments; this.portfolioService = portfolioService; this.instrumentService = instrumentService; this.tradeService = tradeService; this.marketData = marketData; }
    public Map<String, Object> seed() {
        Portfolio p = portfolios.findByName("Global Macro Demo").orElseGet(() -> portfolioService.create(new PortfolioRequest("Global Macro Demo")));
        Instrument aapl = instruments.findBySymbol("AAPL").orElseGet(() -> instrumentService.create(new InstrumentRequest("AAPL", InstrumentType.EQUITY, "Apple Inc.", null, null, null)));
        Instrument msft = instruments.findBySymbol("MSFT").orElseGet(() -> instrumentService.create(new InstrumentRequest("MSFT", InstrumentType.EQUITY, "Microsoft Corp.", null, null, null)));
        Instrument ust = instruments.findBySymbol("UST10Y").orElseGet(() -> instrumentService.create(new InstrumentRequest("UST10Y", InstrumentType.BOND, "US Treasury 10Y", BigDecimal.valueOf(0.035), LocalDate.now().plusYears(10), BigDecimal.valueOf(1000))));
        if (p.getId() != null) { tradeService.create(new TradeRequest(p.getId(), aapl.getId(), TradeSide.BUY, BigDecimal.valueOf(10000), BigDecimal.valueOf(185), null)); tradeService.create(new TradeRequest(p.getId(), msft.getId(), TradeSide.BUY, BigDecimal.valueOf(4500), BigDecimal.valueOf(410), null)); tradeService.create(new TradeRequest(p.getId(), ust.getId(), TradeSide.BUY, BigDecimal.valueOf(1200), BigDecimal.valueOf(1000), null)); }
        marketData.create(new MarketPriceRequest(aapl.getId(), BigDecimal.valueOf(182.50), null, null)); marketData.create(new MarketPriceRequest(msft.getId(), BigDecimal.valueOf(405.75), null, null)); marketData.create(new MarketPriceRequest(ust.getId(), BigDecimal.valueOf(1000), BigDecimal.valueOf(0.041), null));
        return Map.of("portfolioId", p.getId(), "message", "Seeded demo portfolio, instruments, trades, and prices");
    }
    public Map<String, Object> simulate() {
        instruments.findAll().forEach(i -> { BigDecimal price = i.getType() == InstrumentType.EQUITY ? BigDecimal.valueOf(100 + Math.random() * 350).setScale(2, java.math.RoundingMode.HALF_UP) : BigDecimal.valueOf(1000); BigDecimal y = i.getType() == InstrumentType.BOND ? BigDecimal.valueOf(0.035 + Math.random() * 0.02).setScale(6, java.math.RoundingMode.HALF_UP) : null; marketData.create(new MarketPriceRequest(i.getId(), price, y, null)); });
        return Map.of("message", "Published simulated market-data events to Redis Stream", "stream", RedisCacheService.MARKET_DATA_STREAM);
    }
}

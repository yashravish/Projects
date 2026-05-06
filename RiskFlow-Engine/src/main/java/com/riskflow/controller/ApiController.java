package com.riskflow.controller;

import com.riskflow.dto.Requests.*; import com.riskflow.dto.Responses.*;
import com.riskflow.service.*;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;
import java.util.*;

@RestController
@RequestMapping("/api")
public class ApiController {
    private final PortfolioService portfolios; private final InstrumentService instruments; private final TradeService trades; private final MarketDataService marketData; private final RiskCalculationService risk; private final EodReportService eod; private final DemoService demo;
    public ApiController(PortfolioService portfolios, InstrumentService instruments, TradeService trades, MarketDataService marketData, RiskCalculationService risk, EodReportService eod, DemoService demo) { this.portfolios = portfolios; this.instruments = instruments; this.trades = trades; this.marketData = marketData; this.risk = risk; this.eod = eod; this.demo = demo; }
    @GetMapping("/health") public Map<String, String> health() { return Map.of("status", "UP", "service", "RiskFlow Engine"); }
    @PostMapping("/portfolios") public PortfolioResponse createPortfolio(@Valid @RequestBody PortfolioRequest r) { return PortfolioResponse.from(portfolios.create(r)); }
    @GetMapping("/portfolios") public List<PortfolioResponse> portfolios() { return portfolios.findAll().stream().map(PortfolioResponse::from).toList(); }
    @GetMapping("/portfolios/{id}") public PortfolioResponse portfolio(@PathVariable Long id) { return PortfolioResponse.from(portfolios.get(id)); }
    @PostMapping("/instruments") public InstrumentResponse createInstrument(@Valid @RequestBody InstrumentRequest r) { return InstrumentResponse.from(instruments.create(r)); }
    @GetMapping("/instruments") public List<InstrumentResponse> instruments() { return instruments.findAll().stream().map(InstrumentResponse::from).toList(); }
    @GetMapping("/instruments/{id}") public InstrumentResponse instrument(@PathVariable Long id) { return InstrumentResponse.from(instruments.get(id)); }
    @PostMapping("/trades") public TradeResponse createTrade(@Valid @RequestBody TradeRequest r) { return TradeResponse.from(trades.create(r)); }
    @GetMapping("/trades") public List<TradeResponse> trades() { return trades.findAll().stream().map(TradeResponse::from).toList(); }
    @GetMapping("/portfolios/{portfolioId}/trades") public List<TradeResponse> portfolioTrades(@PathVariable Long portfolioId) { return trades.findByPortfolio(portfolioId).stream().map(TradeResponse::from).toList(); }
    @PostMapping("/market-prices") public MarketPriceResponse createPrice(@Valid @RequestBody MarketPriceRequest r) { return MarketPriceResponse.from(marketData.create(r)); }
    @GetMapping("/instruments/{instrumentId}/prices/latest") public MarketPriceResponse latestPrice(@PathVariable Long instrumentId) { return MarketPriceResponse.from(marketData.latest(instrumentId)); }
    @PostMapping("/portfolios/{portfolioId}/risk/calculate") public RiskResponse calculateRisk(@PathVariable Long portfolioId) { return RiskResponse.from(risk.calculate(portfolioId)); }
    @GetMapping("/portfolios/{portfolioId}/risk/latest") public RiskResponse latestRisk(@PathVariable Long portfolioId) { return RiskResponse.from(risk.latest(portfolioId)); }
    @GetMapping("/portfolios/{portfolioId}/risk/history") public List<RiskResponse> riskHistory(@PathVariable Long portfolioId) { return risk.history(portfolioId).stream().map(RiskResponse::from).toList(); }
    @PostMapping("/portfolios/{portfolioId}/eod") public EodReportResponse generateEod(@PathVariable Long portfolioId) { return EodReportResponse.from(eod.generate(portfolioId)); }
    @GetMapping("/portfolios/{portfolioId}/eod/latest") public EodReportResponse latestEod(@PathVariable Long portfolioId) { return EodReportResponse.from(eod.latest(portfolioId)); }
    @PostMapping("/demo/seed") public Map<String, Object> seed() { return demo.seed(); }
    @PostMapping("/demo/run-market-simulation") public Map<String, Object> simulate() { return demo.simulate(); }
}

package com.riskflow.service;

import com.riskflow.dto.Requests.TradeRequest;
import com.riskflow.model.Trade;
import com.riskflow.repository.TradeRepository;
import org.springframework.stereotype.Service;
import java.time.Instant;
import java.util.List;

@Service
public class TradeService {
    private final TradeRepository repository; private final PortfolioService portfolios; private final InstrumentService instruments; private final AuditService audit;
    public TradeService(TradeRepository repository, PortfolioService portfolios, InstrumentService instruments, AuditService audit) { this.repository = repository; this.portfolios = portfolios; this.instruments = instruments; this.audit = audit; }
    public Trade create(TradeRequest r) { Trade t = new Trade(); t.setPortfolio(portfolios.get(r.portfolioId())); t.setInstrument(instruments.get(r.instrumentId())); t.setSide(r.side()); t.setQuantity(r.quantity()); t.setPrice(r.price()); t.setTradeTime(r.tradeTime() == null ? Instant.now() : r.tradeTime()); Trade saved = repository.save(t); audit.log("TRADE_CREATED", "Trade", saved.getId(), "Created " + saved.getSide() + " trade for " + saved.getInstrument().getSymbol()); return saved; }
    public List<Trade> findAll() { return repository.findAll(); }
    public List<Trade> findByPortfolio(Long portfolioId) { return repository.findByPortfolioIdOrderByTradeTimeDesc(portfolioId); }
}

package com.riskflow.service;

import com.riskflow.dto.Requests.MarketPriceRequest;
import com.riskflow.exception.NotFoundException;
import com.riskflow.model.MarketPrice;
import com.riskflow.repository.MarketPriceRepository;
import org.springframework.stereotype.Service;
import java.time.Instant;

@Service
public class MarketDataService {
    private final MarketPriceRepository repository; private final InstrumentService instruments; private final RedisCacheService cache; private final AuditService audit;
    public MarketDataService(MarketPriceRepository repository, InstrumentService instruments, RedisCacheService cache, AuditService audit) { this.repository = repository; this.instruments = instruments; this.cache = cache; this.audit = audit; }
    public MarketPrice create(MarketPriceRequest r) { MarketPrice p = new MarketPrice(); p.setInstrument(instruments.get(r.instrumentId())); p.setPrice(r.price()); p.setYield(r.yield()); p.setTimestamp(r.timestamp() == null ? Instant.now() : r.timestamp()); MarketPrice saved = repository.save(p); cache.cachePriceAndPublish(saved); audit.log("MARKET_PRICE_UPDATED", "Instrument", saved.getInstrument().getId(), "Updated market price for " + saved.getInstrument().getSymbol()); return saved; }
    public MarketPrice latest(Long instrumentId) { return repository.findFirstByInstrumentIdOrderByTimestampDesc(instrumentId).orElseThrow(() -> new NotFoundException("No market price for instrument: " + instrumentId)); }
}

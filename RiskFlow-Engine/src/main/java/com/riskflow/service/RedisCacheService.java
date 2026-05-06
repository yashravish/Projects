package com.riskflow.service;

import com.riskflow.model.MarketPrice;
import com.riskflow.model.RiskResult;
import org.slf4j.Logger; import org.slf4j.LoggerFactory;
import org.springframework.data.redis.connection.stream.MapRecord;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;
import java.util.Map;

@Service
public class RedisCacheService {
    private static final Logger log = LoggerFactory.getLogger(RedisCacheService.class);
    public static final String MARKET_DATA_STREAM = "market-data";
    private final RedisTemplate<String, String> redis;
    public RedisCacheService(RedisTemplate<String, String> redis) { this.redis = redis; }
    public void cachePriceAndPublish(MarketPrice p) {
        try {
            String symbol = p.getInstrument().getSymbol();
            redis.opsForValue().set("price:" + symbol, p.getPrice().toPlainString());
            redis.opsForStream().add(MapRecord.create(MARKET_DATA_STREAM, Map.of("symbol", symbol, "price", p.getPrice().toPlainString(), "yield", p.getYield() == null ? "" : p.getYield().toPlainString(), "timestamp", p.getTimestamp().toString())));
        } catch (RuntimeException ex) { log.warn("Redis unavailable while publishing price: {}", ex.getMessage()); }
    }
    public void cacheRisk(RiskResult r) {
        try {
            Long id = r.getPortfolio().getId();
            String json = String.format("{\"portfolioId\":%d,\"totalMarketValue\":%s,\"totalPnL\":%s,\"var95\":%s,\"concentrationPct\":%s}", id, r.getTotalMarketValue(), r.getTotalPnL(), r.getVar95(), r.getConcentrationPct());
            redis.opsForValue().set("portfolio:" + id + ":risk:latest", json);
            redis.opsForValue().set("portfolio:" + id + ":exposure", r.getTotalMarketValue().toPlainString());
            redis.opsForValue().set("portfolio:" + id + ":alerts", r.getVar95().abs().compareTo(r.getTotalMarketValue().abs().multiply(java.math.BigDecimal.valueOf(0.05))) > 0 ? "HIGH_VAR" : "OK");
        } catch (RuntimeException ex) { log.warn("Redis unavailable while caching risk: {}", ex.getMessage()); }
    }
}

package com.riskflow.model;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.Instant;

@Entity
@Table(name = "market_prices", indexes = @Index(name = "idx_market_price_instrument_time", columnList = "instrument_id, timestamp"))
public class MarketPrice {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @ManyToOne(optional = false, fetch = FetchType.LAZY) @JoinColumn(name = "instrument_id")
    private Instrument instrument;
    @Column(nullable = false, precision = 19, scale = 6)
    private BigDecimal price;
    @Column(name = "market_yield", precision = 12, scale = 8)
    private BigDecimal yield;
    @Column(nullable = false)
    private Instant timestamp = Instant.now();
    public Long getId() { return id; }
    public Instrument getInstrument() { return instrument; }
    public void setInstrument(Instrument instrument) { this.instrument = instrument; }
    public BigDecimal getPrice() { return price; }
    public void setPrice(BigDecimal price) { this.price = price; }
    public BigDecimal getYield() { return yield; }
    public void setYield(BigDecimal yield) { this.yield = yield; }
    public Instant getTimestamp() { return timestamp; }
    public void setTimestamp(Instant timestamp) { this.timestamp = timestamp; }
}

package com.riskflow.model;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.Instant;

@Entity
@Table(name = "trades", indexes = {@Index(name = "idx_trade_portfolio_time", columnList = "portfolio_id, tradeTime"), @Index(name = "idx_trade_instrument", columnList = "instrument_id")})
public class Trade {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @ManyToOne(optional = false, fetch = FetchType.LAZY) @JoinColumn(name = "portfolio_id")
    private Portfolio portfolio;
    @ManyToOne(optional = false, fetch = FetchType.LAZY) @JoinColumn(name = "instrument_id")
    private Instrument instrument;
    @Enumerated(EnumType.STRING) @Column(nullable = false)
    private TradeSide side;
    @Column(nullable = false, precision = 19, scale = 4)
    private BigDecimal quantity;
    @Column(nullable = false, precision = 19, scale = 6)
    private BigDecimal price;
    @Column(nullable = false)
    private Instant tradeTime = Instant.now();
    public Long getId() { return id; }
    public Portfolio getPortfolio() { return portfolio; }
    public void setPortfolio(Portfolio portfolio) { this.portfolio = portfolio; }
    public Instrument getInstrument() { return instrument; }
    public void setInstrument(Instrument instrument) { this.instrument = instrument; }
    public TradeSide getSide() { return side; }
    public void setSide(TradeSide side) { this.side = side; }
    public BigDecimal getQuantity() { return quantity; }
    public void setQuantity(BigDecimal quantity) { this.quantity = quantity; }
    public BigDecimal getPrice() { return price; }
    public void setPrice(BigDecimal price) { this.price = price; }
    public Instant getTradeTime() { return tradeTime; }
    public void setTradeTime(Instant tradeTime) { this.tradeTime = tradeTime; }
}

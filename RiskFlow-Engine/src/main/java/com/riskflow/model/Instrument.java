package com.riskflow.model;

import jakarta.persistence.*;
import java.math.BigDecimal;
import java.time.LocalDate;

@Entity
@Table(name = "instruments", indexes = @Index(name = "idx_instrument_symbol", columnList = "symbol", unique = true))
public class Instrument {
    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    @Column(nullable = false, unique = true)
    private String symbol;
    @Enumerated(EnumType.STRING) @Column(nullable = false)
    private InstrumentType type;
    @Column(nullable = false)
    private String name;
    private BigDecimal couponRate;
    private LocalDate maturityDate;
    private BigDecimal faceValue;
    public Instrument() {}
    public Instrument(String symbol, InstrumentType type, String name, BigDecimal couponRate, LocalDate maturityDate, BigDecimal faceValue) {
        this.symbol = symbol; this.type = type; this.name = name; this.couponRate = couponRate; this.maturityDate = maturityDate; this.faceValue = faceValue;
    }
    public Long getId() { return id; }
    public String getSymbol() { return symbol; }
    public void setSymbol(String symbol) { this.symbol = symbol; }
    public InstrumentType getType() { return type; }
    public void setType(InstrumentType type) { this.type = type; }
    public String getName() { return name; }
    public void setName(String name) { this.name = name; }
    public BigDecimal getCouponRate() { return couponRate; }
    public void setCouponRate(BigDecimal couponRate) { this.couponRate = couponRate; }
    public LocalDate getMaturityDate() { return maturityDate; }
    public void setMaturityDate(LocalDate maturityDate) { this.maturityDate = maturityDate; }
    public BigDecimal getFaceValue() { return faceValue; }
    public void setFaceValue(BigDecimal faceValue) { this.faceValue = faceValue; }
}

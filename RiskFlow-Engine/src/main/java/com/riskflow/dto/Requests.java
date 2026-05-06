package com.riskflow.dto;

import com.riskflow.model.InstrumentType;
import com.riskflow.model.TradeSide;
import jakarta.validation.constraints.*;
import java.math.BigDecimal;
import java.time.Instant;
import java.time.LocalDate;

public final class Requests {
    private Requests() {}
    public record PortfolioRequest(@NotBlank String name) {}
    public record InstrumentRequest(@NotBlank String symbol, @NotNull InstrumentType type, @NotBlank String name, BigDecimal couponRate, LocalDate maturityDate, BigDecimal faceValue) {}
    public record TradeRequest(@NotNull Long portfolioId, @NotNull Long instrumentId, @NotNull TradeSide side, @NotNull @Positive BigDecimal quantity, @NotNull @Positive BigDecimal price, Instant tradeTime) {}
    public record MarketPriceRequest(@NotNull Long instrumentId, @NotNull @Positive BigDecimal price, BigDecimal yield, Instant timestamp) {}
}

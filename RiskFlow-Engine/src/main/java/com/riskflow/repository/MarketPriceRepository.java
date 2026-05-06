package com.riskflow.repository;
import com.riskflow.model.MarketPrice;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.Optional;
public interface MarketPriceRepository extends JpaRepository<MarketPrice, Long> { Optional<MarketPrice> findFirstByInstrumentIdOrderByTimestampDesc(Long instrumentId); List<MarketPrice> findTop30ByInstrumentIdOrderByTimestampDesc(Long instrumentId); }

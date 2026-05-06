package com.riskflow.repository;
import com.riskflow.model.RiskResult;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
import java.util.Optional;
public interface RiskResultRepository extends JpaRepository<RiskResult, Long> { Optional<RiskResult> findFirstByPortfolioIdOrderByTimestampDesc(Long portfolioId); List<RiskResult> findByPortfolioIdOrderByTimestampDesc(Long portfolioId); }

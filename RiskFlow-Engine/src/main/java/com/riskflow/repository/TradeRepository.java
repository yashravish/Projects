package com.riskflow.repository;
import com.riskflow.model.Trade;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.List;
public interface TradeRepository extends JpaRepository<Trade, Long> { List<Trade> findByPortfolioIdOrderByTradeTimeDesc(Long portfolioId); List<Trade> findByPortfolioId(Long portfolioId); }

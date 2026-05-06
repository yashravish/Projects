package com.riskflow.repository;
import com.riskflow.model.Portfolio;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;
public interface PortfolioRepository extends JpaRepository<Portfolio, Long> { Optional<Portfolio> findByName(String name); }

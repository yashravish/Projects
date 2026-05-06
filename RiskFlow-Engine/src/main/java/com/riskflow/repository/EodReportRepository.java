package com.riskflow.repository;
import com.riskflow.model.EodReport;
import org.springframework.data.jpa.repository.JpaRepository;
import java.util.Optional;
public interface EodReportRepository extends JpaRepository<EodReport, Long> { Optional<EodReport> findFirstByPortfolioIdOrderByReportDateDesc(Long portfolioId); }

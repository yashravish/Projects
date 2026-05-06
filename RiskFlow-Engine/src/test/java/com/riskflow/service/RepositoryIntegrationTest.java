package com.riskflow.service;

import com.riskflow.model.Portfolio; import com.riskflow.repository.PortfolioRepository;
import org.junit.jupiter.api.Test; import org.springframework.beans.factory.annotation.Autowired; import org.springframework.boot.test.autoconfigure.orm.jpa.DataJpaTest;
import static org.junit.jupiter.api.Assertions.*;

@DataJpaTest
class RepositoryIntegrationTest {
    @Autowired PortfolioRepository portfolios;
    @Test void persistsPortfolioWithH2AlternativeToTestcontainers() { Portfolio saved = portfolios.save(new Portfolio("Integration Book")); assertTrue(portfolios.findById(saved.getId()).isPresent()); }
}

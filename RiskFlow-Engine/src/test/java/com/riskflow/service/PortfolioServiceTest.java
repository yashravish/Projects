package com.riskflow.service;

import com.riskflow.dto.Requests.PortfolioRequest;
import com.riskflow.model.Portfolio;
import com.riskflow.repository.PortfolioRepository;
import org.junit.jupiter.api.Test;
import java.util.Optional;
import static org.junit.jupiter.api.Assertions.*; import static org.mockito.Mockito.*;

class PortfolioServiceTest {
    @Test void createsPortfolio() { PortfolioRepository repo = mock(PortfolioRepository.class); when(repo.save(any())).thenAnswer(i -> i.getArgument(0)); PortfolioService svc = new PortfolioService(repo); Portfolio p = svc.create(new PortfolioRequest("PM Book")); assertEquals("PM Book", p.getName()); verify(repo).save(any(Portfolio.class)); }
    @Test void missingPortfolioThrows() { PortfolioRepository repo = mock(PortfolioRepository.class); when(repo.findById(9L)).thenReturn(Optional.empty()); assertThrows(RuntimeException.class, () -> new PortfolioService(repo).get(9L)); }
}

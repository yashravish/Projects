package com.riskflow.service;

import com.riskflow.dto.Requests.PortfolioRequest;
import com.riskflow.exception.NotFoundException;
import com.riskflow.model.Portfolio;
import com.riskflow.repository.PortfolioRepository;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class PortfolioService {
    private final PortfolioRepository repository;
    public PortfolioService(PortfolioRepository repository) { this.repository = repository; }
    public Portfolio create(PortfolioRequest request) { return repository.save(new Portfolio(request.name())); }
    public List<Portfolio> findAll() { return repository.findAll(); }
    public Portfolio get(Long id) { return repository.findById(id).orElseThrow(() -> new NotFoundException("Portfolio not found: " + id)); }
}

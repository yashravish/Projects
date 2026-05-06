package com.riskflow.service;

import com.riskflow.dto.Requests.InstrumentRequest;
import com.riskflow.exception.NotFoundException;
import com.riskflow.model.Instrument;
import com.riskflow.repository.InstrumentRepository;
import org.springframework.stereotype.Service;
import java.util.List;

@Service
public class InstrumentService {
    private final InstrumentRepository repository;
    public InstrumentService(InstrumentRepository repository) { this.repository = repository; }
    public Instrument create(InstrumentRequest r) { return repository.save(new Instrument(r.symbol(), r.type(), r.name(), r.couponRate(), r.maturityDate(), r.faceValue())); }
    public List<Instrument> findAll() { return repository.findAll(); }
    public Instrument get(Long id) { return repository.findById(id).orElseThrow(() -> new NotFoundException("Instrument not found: " + id)); }
    public Instrument getBySymbol(String symbol) { return repository.findBySymbol(symbol).orElseThrow(() -> new NotFoundException("Instrument not found: " + symbol)); }
}

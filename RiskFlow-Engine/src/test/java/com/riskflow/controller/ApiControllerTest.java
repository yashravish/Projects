package com.riskflow.controller;

import com.riskflow.service.*; import org.junit.jupiter.api.Test; import org.springframework.beans.factory.annotation.Autowired; import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest; import org.springframework.boot.test.mock.mockito.MockBean; import org.springframework.test.web.servlet.MockMvc;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get; import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.*;

@WebMvcTest(ApiController.class)
class ApiControllerTest {
    @Autowired MockMvc mvc;
    @MockBean PortfolioService portfolios; @MockBean InstrumentService instruments; @MockBean TradeService trades; @MockBean MarketDataService marketData; @MockBean RiskCalculationService risk; @MockBean EodReportService eod; @MockBean DemoService demo;
    @Test void healthIsUp() throws Exception { mvc.perform(get("/api/health")).andExpect(status().isOk()).andExpect(jsonPath("$.status").value("UP")); }
}

package com.fraudshield.api;

import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/risk")
public class RiskScoringController {

    @PostMapping("/score")
    public RiskScore scoreTransaction(@RequestBody Transaction txn) {
        // Example risk scoring logic: assign high risk for large transactions
        double riskScore = txn.getAmount() > 10000 ? 0.9 : 0.1;
        return new RiskScore(txn.getId(), riskScore);
    }
}

package com.neurodrive.api;

import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;

import jakarta.validation.constraints.*;
import java.util.*;

@RestController
@RequestMapping("/api/v1")
@Validated
public class ApiController {

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "ok");
    }

    public record InferRequest(
            @DecimalMin(value = "0.0") @DecimalMax(value = "1.0") Double eyeClosureRatio,
            Boolean phoneUsage,
            @Min(0) Integer speed
    ) {}

    @PostMapping("/infer")
    public ResponseEntity<Map<String, Object>> infer(@RequestBody InferRequest body) {
        double ratio = body.eyeClosureRatio() == null ? 0.2 : body.eyeClosureRatio();
        boolean phone = body.phoneUsage() != null && body.phoneUsage();
        int speed = body.speed() == null ? 65 : body.speed();

        // Minimal mirror of Python logic for demo parity
        double eyeClosureTimeSeconds = ratio * 3.0;
        boolean drowsinessOk = eyeClosureTimeSeconds < 2.0;
        boolean distractionOk = !phone;
        boolean speedOk = speed <= 80;
        double drowsinessSub = drowsinessOk ? 100 : Math.max(0, Math.min(100, 100 - ((eyeClosureTimeSeconds - 2.0) * 40)));
        double distractionSub = distractionOk ? 100 : 35;
        double speedSub = speedOk ? 100 : Math.max(0, Math.min(100, 100 - ((speed - 80) * 1.2)));
        double baseWeighted = (drowsinessSub * 0.25 + distractionSub * 0.30 + 100 * 0.25 + 100 * 0.20) / (0.25 + 0.30 + 0.25 + 0.20);
        int riskScore = (int) Math.round(Math.max(0, Math.min(100, baseWeighted * (speedSub / 100.0))));

        Map<String, Object> resp = new HashMap<>();
        resp.put("riskScore", riskScore);
        resp.put("statusText", statusText(riskScore));
        return ResponseEntity.ok(resp);
    }

    private String statusText(int score) {
        if (score < 30) return "Critical Risk";
        if (score < 50) return "High Risk";
        if (score < 70) return "Elevated Risk";
        if (score < 85) return "Low Risk";
        return "Alert";
    }
}



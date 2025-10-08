package com.neurodrive.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class DriverStatusResponse {
    private Long driverId;
    private String driverName;
    private Boolean isOnline;
    private String currentLocation;
    private Double latitude;
    private Double longitude;
    private Integer currentSpeed;
    private Integer riskScore;
    private Boolean isSosActive;
    private LocalDateTime lastUpdate;
}

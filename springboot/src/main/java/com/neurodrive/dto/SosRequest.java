package com.neurodrive.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class SosRequest {
    private String location;
    private Double latitude;
    private Double longitude;
    private String reason;
}

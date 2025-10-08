package com.neurodrive.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class FamilyMemberResponse {
    private Long id;
    private String name;
    private String relationship;
    private Boolean canReceiveAlerts;
    private Boolean canViewLocation;
}

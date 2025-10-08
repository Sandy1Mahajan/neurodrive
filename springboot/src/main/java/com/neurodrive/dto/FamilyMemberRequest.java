package com.neurodrive.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor
public class FamilyMemberRequest {
    private String name;
    private String phoneNumber;
    private String email;
    private String relationship;
    private Boolean canReceiveAlerts;
    private Boolean canViewLocation;
}

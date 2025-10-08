package com.neurodrive.controller;

import com.neurodrive.dto.FamilyMemberRequest;
import com.neurodrive.dto.FamilyMemberResponse;
import com.neurodrive.dto.DriverStatusResponse;
import com.neurodrive.entity.FamilyMember;
import com.neurodrive.entity.User;
import com.neurodrive.entity.DriverSession;
import com.neurodrive.repository.DriverSessionRepository;
import com.neurodrive.repository.FamilyMemberRepository;
import com.neurodrive.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.util.List;
import java.util.stream.Collectors;

@RestController
@RequestMapping("/api/v1/family")
@RequiredArgsConstructor
public class FamilyController {
    
    private final FamilyMemberRepository familyMemberRepository;
    private final UserRepository userRepository;
    private final DriverSessionRepository sessionRepository;
    
    @PostMapping("/members")
    public ResponseEntity<FamilyMemberResponse> addFamilyMember(
            @RequestBody FamilyMemberRequest request, 
            Authentication authentication) {
        String username = authentication.getName();
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        
        FamilyMember member = FamilyMember.builder()
                .user(user)
                .name(request.getName())
                .phoneNumber(request.getPhoneNumber())
                .email(request.getEmail())
                .relationship(request.getRelationship())
                .canReceiveAlerts(request.getCanReceiveAlerts())
                .canViewLocation(request.getCanViewLocation())
                .isActive(true)
                .build();
        
        familyMemberRepository.save(member);
        
        return ResponseEntity.ok(FamilyMemberResponse.builder()
                .id(member.getId())
                .name(member.getName())
                .relationship(member.getRelationship())
                .canReceiveAlerts(member.getCanReceiveAlerts())
                .canViewLocation(member.getCanViewLocation())
                .build());
    }
    
    @GetMapping("/members")
    public ResponseEntity<List<FamilyMemberResponse>> getFamilyMembers(Authentication authentication) {
        String username = authentication.getName();
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        
        List<FamilyMember> members = familyMemberRepository.findByUserIdAndIsActive(user.getId(), true);
        
        List<FamilyMemberResponse> responses = members.stream()
                .map(member -> FamilyMemberResponse.builder()
                        .id(member.getId())
                        .name(member.getName())
                        .relationship(member.getRelationship())
                        .canReceiveAlerts(member.getCanReceiveAlerts())
                        .canViewLocation(member.getCanViewLocation())
                        .build())
                .collect(Collectors.toList());
        
        return ResponseEntity.ok(responses);
    }
    
    @GetMapping("/driver-status/{driverId}")
    public ResponseEntity<DriverStatusResponse> getDriverStatus(
            @PathVariable Long driverId, 
            Authentication authentication) {
        String username = authentication.getName();
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        
        // Check if user has permission to view this driver's status
        boolean hasPermission = familyMemberRepository.findByUserIdAndIsActive(user.getId(), true)
                .stream()
                .anyMatch(member -> member.getCanViewLocation());
        
        if (!hasPermission) {
            return ResponseEntity.status(403).build();
        }
        
        User driver = userRepository.findById(driverId)
                .orElseThrow(() -> new RuntimeException("Driver not found"));
        
        DriverSession activeSession = sessionRepository.findByUserIdAndStatus(driverId, DriverSession.Status.ACTIVE)
                .stream()
                .findFirst()
                .orElse(null);
        
        DriverStatusResponse response = DriverStatusResponse.builder()
                .driverId(driverId)
                .driverName(driver.getFirstName() + " " + driver.getLastName())
                .isOnline(activeSession != null)
                .currentLocation(activeSession != null ? activeSession.getLocation() : null)
                .latitude(activeSession != null ? activeSession.getLatitude() : null)
                .longitude(activeSession != null ? activeSession.getLongitude() : null)
                .currentSpeed(activeSession != null ? activeSession.getCurrentSpeed() : null)
                .riskScore(activeSession != null ? activeSession.getRiskScore() : null)
                .isSosActive(activeSession != null ? activeSession.getIsSosActive() : false)
                .lastUpdate(activeSession != null ? activeSession.getSessionStart() : null)
                .build();
        
        return ResponseEntity.ok(response);
    }
}

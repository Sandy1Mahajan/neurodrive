package com.neurodrive.controller;

import com.neurodrive.dto.SosRequest;
import com.neurodrive.dto.SosResponse;
import com.neurodrive.entity.Alert;
import com.neurodrive.entity.DriverSession;
import com.neurodrive.entity.User;
import com.neurodrive.repository.AlertRepository;
import com.neurodrive.repository.DriverSessionRepository;
import com.neurodrive.repository.FamilyMemberRepository;
import com.neurodrive.repository.UserRepository;
import com.neurodrive.service.NotificationService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;

@RestController
@RequestMapping("/api/v1/sos")
@RequiredArgsConstructor
public class SosController {
    
    private final DriverSessionRepository sessionRepository;
    private final AlertRepository alertRepository;
    private final FamilyMemberRepository familyMemberRepository;
    private final UserRepository userRepository;
    private final NotificationService notificationService;
    
    @PostMapping("/activate")
    public ResponseEntity<SosResponse> activateSos(@RequestBody SosRequest request, Authentication authentication) {
        String username = authentication.getName();
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        
        // Find active session
        DriverSession session = sessionRepository.findByUserIdAndStatus(user.getId(), DriverSession.Status.ACTIVE)
                .stream()
                .findFirst()
                .orElseThrow(() -> new RuntimeException("No active session found"));
        
        // Activate SOS
        session.setIsSosActive(true);
        session.setStatus(DriverSession.Status.EMERGENCY);
        sessionRepository.save(session);
        
        // Create critical alert
        Alert alert = Alert.builder()
                .session(session)
                .type(Alert.AlertType.SOS_ACTIVATED)
                .severity(Alert.Severity.CRITICAL)
                .message("SOS ACTIVATED - Emergency assistance needed")
                .details("Driver activated SOS at location: " + request.getLocation())
                .latitude(request.getLatitude())
                .longitude(request.getLongitude())
                .build();
        alertRepository.save(alert);
        
        // Notify family members
        familyMemberRepository.findByUserIdAndIsActive(user.getId(), true)
                .forEach(familyMember -> {
                    if (familyMember.getCanReceiveAlerts()) {
                        notificationService.sendSosNotification(familyMember, session, alert);
                    }
                });
        
        return ResponseEntity.ok(SosResponse.builder()
                .success(true)
                .message("SOS activated successfully")
                .alertId(alert.getId())
                .build());
    }
    
    @PostMapping("/deactivate")
    public ResponseEntity<SosResponse> deactivateSos(Authentication authentication) {
        String username = authentication.getName();
        User user = userRepository.findByUsername(username)
                .orElseThrow(() -> new RuntimeException("User not found"));
        
        DriverSession session = sessionRepository.findByUserIdAndStatusAndIsSosActive(user.getId(), DriverSession.Status.EMERGENCY, true)
                .orElseThrow(() -> new RuntimeException("No active SOS session found"));
        
        session.setIsSosActive(false);
        session.setStatus(DriverSession.Status.ACTIVE);
        sessionRepository.save(session);
        
        // Create resolution alert
        Alert alert = Alert.builder()
                .session(session)
                .type(Alert.AlertType.SOS_ACTIVATED)
                .severity(Alert.Severity.LOW)
                .message("SOS deactivated - Emergency resolved")
                .details("Driver deactivated SOS")
                .isResolved(true)
                .resolvedAt(LocalDateTime.now())
                .build();
        alertRepository.save(alert);
        
        return ResponseEntity.ok(SosResponse.builder()
                .success(true)
                .message("SOS deactivated successfully")
                .build());
    }
}

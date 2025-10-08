package com.neurodrive.service;

import com.neurodrive.entity.Alert;
import com.neurodrive.entity.DriverSession;
import com.neurodrive.entity.FamilyMember;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.stereotype.Service;

@Service
@RequiredArgsConstructor
@Slf4j
public class NotificationService {
    
    private final SimpMessagingTemplate messagingTemplate;
    
    public void sendSosNotification(FamilyMember familyMember, DriverSession session, Alert alert) {
        try {
            // Send WebSocket notification
            String destination = "/topic/sos/" + familyMember.getId();
            messagingTemplate.convertAndSend(destination, alert);
            
            // TODO: Send SMS/Email notifications
            log.info("SOS notification sent to family member: {} for driver: {}", 
                    familyMember.getName(), session.getUser().getUsername());
            
        } catch (Exception e) {
            log.error("Failed to send SOS notification to family member: {}", familyMember.getId(), e);
        }
    }
    
    public void sendRiskAlert(FamilyMember familyMember, DriverSession session, Alert alert) {
        try {
            String destination = "/topic/alerts/" + familyMember.getId();
            messagingTemplate.convertAndSend(destination, alert);
            
            log.info("Risk alert sent to family member: {} for driver: {}", 
                    familyMember.getName(), session.getUser().getUsername());
            
        } catch (Exception e) {
            log.error("Failed to send risk alert to family member: {}", familyMember.getId(), e);
        }
    }
}

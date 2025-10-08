package com.neurodrive.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "alerts")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class Alert {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "session_id", nullable = false)
    private DriverSession session;
    
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private AlertType type;
    
    @Enumerated(EnumType.STRING)
    @Column(nullable = false)
    private Severity severity;
    
    @Column(columnDefinition = "TEXT")
    private String message;
    
    @Column(columnDefinition = "TEXT")
    private String details;
    
    @Column
    private Double latitude;
    
    @Column
    private Double longitude;
    
    @Column
    private Boolean isAcknowledged = false;
    
    @Column
    private Boolean isResolved = false;
    
    @CreationTimestamp
    private LocalDateTime createdAt;
    
    @Column
    private LocalDateTime acknowledgedAt;
    
    @Column
    private LocalDateTime resolvedAt;
    
    public enum AlertType {
        FATIGUE_DETECTED,
        DISTRACTION_DETECTED,
        EMOTION_STRESS,
        SPEED_VIOLATION,
        SOS_ACTIVATED,
        AUTHENTICATION_FAILED,
        SYSTEM_ERROR
    }
    
    public enum Severity {
        LOW, MEDIUM, HIGH, CRITICAL
    }
}

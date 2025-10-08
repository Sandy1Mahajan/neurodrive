package com.neurodrive.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "driver_sessions")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class DriverSession {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;
    
    @Column(nullable = false)
    private String vehicleId;
    
    @Column
    private String location;
    
    @Column
    private Double latitude;
    
    @Column
    private Double longitude;
    
    @Column
    private Integer currentSpeed;
    
    @Column
    private Integer riskScore;
    
    @Enumerated(EnumType.STRING)
    private Status status;
    
    @CreationTimestamp
    private LocalDateTime sessionStart;
    
    @Column
    private LocalDateTime sessionEnd;
    
    @Column
    private Boolean isSosActive = false;
    
    public enum Status {
        ACTIVE, INACTIVE, EMERGENCY
    }
}

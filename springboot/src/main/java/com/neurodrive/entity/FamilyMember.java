package com.neurodrive.entity;

import jakarta.persistence.*;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.AllArgsConstructor;
import lombok.Builder;
import org.hibernate.annotations.CreationTimestamp;

import java.time.LocalDateTime;

@Entity
@Table(name = "family_members")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class FamilyMember {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;
    
    @Column(nullable = false)
    private String name;
    
    @Column
    private String phoneNumber;
    
    @Column
    private String email;
    
    @Column
    private String relationship; // e.g., "spouse", "parent", "child"
    
    @Column
    private Boolean canReceiveAlerts = true;
    
    @Column
    private Boolean canViewLocation = true;
    
    @Column
    private Boolean isActive = true;
    
    @CreationTimestamp
    private LocalDateTime createdAt;
}

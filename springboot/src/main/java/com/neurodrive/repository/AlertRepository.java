package com.neurodrive.repository;

import com.neurodrive.entity.Alert;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface AlertRepository extends JpaRepository<Alert, Long> {
    List<Alert> findBySessionUserIdAndIsAcknowledged(Long userId, Boolean isAcknowledged);
    List<Alert> findBySessionUserIdAndIsResolved(Long userId, Boolean isResolved);
    List<Alert> findByTypeAndCreatedAtAfter(Alert.AlertType type, LocalDateTime after);
    
    @Query("SELECT a FROM Alert a WHERE a.session.user.id = :userId ORDER BY a.createdAt DESC")
    List<Alert> findRecentAlertsByUserId(Long userId);
    
    List<Alert> findBySeverityAndIsResolved(Alert.Severity severity, Boolean isResolved);
}

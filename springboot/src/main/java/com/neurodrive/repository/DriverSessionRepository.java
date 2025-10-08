package com.neurodrive.repository;

import com.neurodrive.entity.DriverSession;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.stereotype.Repository;

import java.util.List;
import java.util.Optional;

@Repository
public interface DriverSessionRepository extends JpaRepository<DriverSession, Long> {
    List<DriverSession> findByUserIdAndStatus(Long userId, DriverSession.Status status);
    Optional<DriverSession> findByUserIdAndStatusAndIsSosActive(Long userId, DriverSession.Status status, Boolean isSosActive);
    
    @Query("SELECT ds FROM DriverSession ds WHERE ds.user.id = :userId ORDER BY ds.sessionStart DESC")
    List<DriverSession> findRecentSessionsByUserId(Long userId);
    
    List<DriverSession> findByStatus(DriverSession.Status status);
}

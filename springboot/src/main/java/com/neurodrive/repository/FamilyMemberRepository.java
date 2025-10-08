package com.neurodrive.repository;

import com.neurodrive.entity.FamilyMember;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface FamilyMemberRepository extends JpaRepository<FamilyMember, Long> {
    List<FamilyMember> findByUserIdAndIsActive(Long userId, Boolean isActive);
    List<FamilyMember> findByUserIdAndCanReceiveAlerts(Long userId, Boolean canReceiveAlerts);
}

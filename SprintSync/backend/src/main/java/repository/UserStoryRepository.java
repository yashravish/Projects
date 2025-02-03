// backend/src/main/java/com/example/sprintsync/repository/UserStoryRepository.java
package com.example.sprintsync.repository;

import com.example.sprintsync.model.UserStory;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

@Repository
public interface UserStoryRepository extends JpaRepository<UserStory, Long> {
}

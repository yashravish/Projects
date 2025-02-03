// backend/src/main/java/com/example/sprintsync/controller/UserStoryController.java
package com.example.sprintsync.controller;

import com.example.sprintsync.model.UserStory;
import com.example.sprintsync.repository.UserStoryRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.messaging.simp.SimpMessagingTemplate;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/userstories")
public class UserStoryController {

    @Autowired
    private UserStoryRepository userStoryRepository;

    @Autowired
    private SimpMessagingTemplate messagingTemplate;

    @GetMapping
    public List<UserStory> getAllUserStories() {
        return userStoryRepository.findAll();
    }

    @PostMapping
    public UserStory createUserStory(@RequestBody UserStory userStory) {
        UserStory created = userStoryRepository.save(userStory);
        // Notify all subscribers about the new user story
        messagingTemplate.convertAndSend("/topic/userstories", created);
        return created;
    }

    @PutMapping("/{id}")
    public UserStory updateUserStory(@PathVariable Long id, @RequestBody UserStory userStory) {
        userStory.setId(id);
        UserStory updated = userStoryRepository.save(userStory);
        // Broadcast the updated user story
        messagingTemplate.convertAndSend("/topic/userstories", updated);
        return updated;
    }
}


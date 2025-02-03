// backend/src/main/java/com/example/sprintsync/model/UserStory.java
package com.example.sprintsync.model;

import javax.persistence.*;

@Entity
public class UserStory {
    
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;
    
    private String title;
    
    @Column(length = 1000)
    private String description;
    
    // For simplicity, status is a string such as "backlog", "in progress", "done"
    private String status;

    // Constructors
    public UserStory() {}

    public UserStory(String title, String description, String status) {
        this.title = title;
        this.description = description;
        this.status = status;
    }

    // Getters and Setters
    public Long getId() { return id; }
    public void setId(Long id) { this.id = id; }
    
    public String getTitle() { return title; }
    public void setTitle(String title) { this.title = title; }
    
    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }
    
    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }
}

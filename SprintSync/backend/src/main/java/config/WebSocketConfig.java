// backend/src/main/java/com/example/sprintsync/config/WebSocketConfig.java
package com.example.sprintsync.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.messaging.simp.config.MessageBrokerRegistry;
import org.springframework.web.socket.config.annotation.*;

@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {
    
    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        // Endpoint for WebSocket connections with SockJS fallback support
        registry.addEndpoint("/ws").setAllowedOrigins("*").withSockJS();
    }
    
    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        // Enable a simple in-memory broker for broadcasting messages to "/topic" destinations
        config.enableSimpleBroker("/topic");
        // Set prefix for messages bound for @MessageMapping methods (if any)
        config.setApplicationDestinationPrefixes("/app");
    }
}

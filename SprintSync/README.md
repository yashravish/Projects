SprintSync
SprintSync is an Agile Task Manager with Real‑Time Collaboration. The project uses a Java Spring Boot backend (with JPA and WebSocket support) and a React frontend (with Axios, SockJS, and STOMP) to provide a live sprint board for creating and updating user stories.

Features
User Stories Management: Create, update, and list tasks (user stories) using RESTful endpoints.
Real‑Time Collaboration: Receive instant updates via WebSockets (STOMP over SockJS) whenever a user story is created or updated.
Sprint Board: Display user stories organized by status (Backlog, In Progress, Done) in an interactive board.
Quick Setup: Uses an in-memory H2 database for rapid development.
Deployment Ready: Instructions for deploying on AWS EC2 (or any server) are provided.
Technology Stack
Backend: Java, Spring Boot, Spring Data JPA, H2 Database, Spring WebSocket
Frontend: React, Axios, SockJS, StompJS
Prerequisites
Backend:
Java 11 or higher
Maven
Frontend:
Node.js and npm
Project Structure
bash
Copy
SprintSync/
├── backend/          # Spring Boot application
└── frontend/         # React application
Backend Setup
Clone the repository and navigate to the backend folder.

Build the Project:

bash
Copy
mvn clean package
Run the Application:

bash
Copy
java -jar target/sprintsync-0.0.1-SNAPSHOT.jar
The backend server will start on http://localhost:8080.
You can also access the H2 console at http://localhost:8080/h2-console.

Frontend Setup
Navigate to the frontend folder.

Install Dependencies:

bash
Copy
npm install
Start the Development Server:

bash
Copy
npm start
The React application will open in your browser at http://localhost:3000.

How It Works
User Stories API:
The backend exposes a REST API under /api/userstories to create, update, and list user stories. When a new user story is created or updated, the backend broadcasts the change on the /topic/userstories WebSocket channel.

Real-Time Updates:
The React frontend connects to the WebSocket endpoint (/ws using SockJS) and subscribes to /topic/userstories. When updates arrive, the sprint board automatically reflects the new or updated user stories.

Sprint Board:
The board displays user stories categorized into "Backlog," "In Progress," and "Done" columns. Users can create new stories via the form, and all connected clients will see updates in real‑time.

Deployment on AWS EC2
Backend Deployment:
Package the Application:

bash
Copy
mvn clean package
Launch an AWS EC2 Instance:
Ensure Java is installed on the instance.

Deploy the JAR:
Upload the generated JAR (sprintsync-0.0.1-SNAPSHOT.jar) to your EC2 instance and run it:

bash
Copy
java -jar sprintsync-0.0.1-SNAPSHOT.jar
Configure Security:
Open port 8080 in the instance’s security group.

Frontend Deployment:
Build the React App:

bash
Copy
npm run build
Host the Build Folder:

Option 1: Upload the build folder to an AWS S3 bucket and serve it using CloudFront.
Option 2: Serve static files from the backend by copying the build files to the src/main/resources/static directory of your Spring Boot project and rebuilding the backend.
Domain & SSL:
Use AWS Route 53 for DNS management.
Configure an SSL certificate via AWS Certificate Manager and, if needed, set up a load balancer.
Future Enhancements
User Authentication: Integrate a user management system for secure access.
Advanced Task Features: Expand the UserStory model with fields such as priority, assignees, comments, and attachments.
Enhanced UI/UX: Improve the frontend styling and interactions.
Conclusion
SprintSync provides a solid foundation for an agile task management tool that supports real‑time collaboration. Enjoy building and enhancing your agile workflow with SprintSync!


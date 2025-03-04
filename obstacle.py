import numpy as np
import cv2
import random
import math

class DroneNavigation:
    def __init__(self, frame_size=(600, 800), score_tracker=None):
        # Frame setup
        self.frame_size = frame_size
        self.frame = np.zeros((frame_size[0], frame_size[1], 3), dtype=np.uint8)
        
        # Scoring and game state
        self.score_tracker = score_tracker if score_tracker else {'successes': 0, 'failures': 0}
        
        # Randomly spawn drone and destination
        self.spawn_points()
        
        # Movement parameters
        self.speed = 3
        self.safe_distance = 50
        self.prediction_distance = 50  # Increased prediction distance
        
        # Obstacles list
        self.obstacles = []
        
        # Mission status
        self.mission_complete = False
        self.collision_detected = False
        
        # Movement direction
        self.current_direction = [0, 0]

    def spawn_points(self):
        """Randomly spawn drone and destination on opposite sides"""
        drone_side = random.choice(['left', 'right'])
        
        if drone_side == 'left':
            self.drone_pos = [
                random.randint(0, self.frame_size[0] // 4),  # y 
                random.randint(0, self.frame_size[1] // 4)   # x
            ]
            self.destination_pos = [
                random.randint(3 * self.frame_size[0] // 4, self.frame_size[0] - 1),  # y
                random.randint(3 * self.frame_size[1] // 4, self.frame_size[1] - 1)   # x
            ]
        else:
            self.drone_pos = [
                random.randint(0, self.frame_size[0] // 4),  # y
                random.randint(3 * self.frame_size[1] // 4, self.frame_size[1] - 1)  # x
            ]
            self.destination_pos = [
                random.randint(3 * self.frame_size[0] // 4, self.frame_size[0] - 1),  # y
                random.randint(0, self.frame_size[1] // 4)   # x
            ]

    def is_point_in_obstacle(self, point):
        """Check if a point is inside any obstacle"""
        for obstacle in self.obstacles:
            if (obstacle['x'] <= point[1] <= obstacle['x'] + obstacle['size'] and
                obstacle['y'] <= point[0] <= obstacle['y'] + obstacle['size']):
                return True
        return False

    def create_dynamic_obstacles(self):
        """Create and move dynamic obstacles with more controlled movement"""
        # Clear previous frame
        self.frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
        
        # Draw destination (green dot)
        cv2.circle(
            self.frame, 
            (self.destination_pos[1], self.destination_pos[0]), 
            15, 
            (0, 255, 0), 
            -1  # Green destination marker
        )
        
        # Initial obstacle generation if no obstacles exist
        if not self.obstacles:
            for _ in range(30):
                x = random.randint(100, self.frame_size[1] - 100)
                y = random.randint(50, self.frame_size[0] - 50)
                size = random.randint(30, 50)
                
                self.obstacles.append({
                    'x': x, 
                    'y': y, 
                    'size': size,
                    'dx': random.uniform(-1.5, 1.5),
                    'dy': random.uniform(-1.5, 1.5),
                    'min_x': max(0, x - 150),
                    'max_x': min(self.frame_size[1], x + 150),
                    'min_y': max(0, y - 150),
                    'max_y': min(self.frame_size[0], y + 150)
                })
        
        # Update and draw obstacles
        for obstacle in self.obstacles:
            # Move obstacles within their boundaries
            obstacle['x'] += obstacle['dx']
            obstacle['y'] += obstacle['dy']
            
            # Boundary checking with more dynamic bouncing
            if obstacle['x'] <= obstacle['min_x'] or obstacle['x'] >= obstacle['max_x']:
                obstacle['dx'] *= -1.1  # Slight speed variation
            if obstacle['y'] <= obstacle['min_y'] or obstacle['y'] >= obstacle['max_y']:
                obstacle['dy'] *= -1.1
            
            # Draw obstacle
            cv2.rectangle(
                self.frame, 
                (int(obstacle['x']), int(obstacle['y'])), 
                (int(obstacle['x'] + obstacle['size']), int(obstacle['y'] + obstacle['size'])), 
                (255, 255, 255), 
                -1
            )

    def calculate_direction(self):
        """Advanced direction calculation with improved obstacle avoidance"""
        # Calculate direct path to destination
        dx = self.destination_pos[1] - self.drone_pos[1]
        dy = self.destination_pos[0] - self.drone_pos[0]
        
        # Calculate distance to destination
        distance = math.sqrt(dx**2 + dy**2)
        
        # Prevent division by zero
        if distance == 0:
            return [0, 0], 0
        
        # Normalize direction vector
        direction = [dx / distance, dy / distance]
        
        # Generate alternative directions
        alternative_directions = [
            direction,  # Original direction
            [direction[0], -direction[1]],  # Vertical deviation
            [-direction[0], direction[1]],  # Horizontal deviation
            [-direction[0], -direction[1]],  # Opposite direction
            [direction[0] * 0.5, direction[1] * 0.5],  # Slower original direction
        ]
        
        # Find the first safe direction
        for alt_dir in alternative_directions:
            # Predict multiple points along the path
            is_safe = True
            for step in range(1, 6):  # Check multiple points along the path
                predicted_pos = [
                    self.drone_pos[0] + alt_dir[1] * self.speed * step,
                    self.drone_pos[1] + alt_dir[0] * self.speed * step
                ]
                
                # Check if predicted position is in an obstacle or out of bounds
                if (self.is_point_in_obstacle(predicted_pos) or 
                    predicted_pos[0] < 0 or predicted_pos[0] >= self.frame_size[0] or
                    predicted_pos[1] < 0 or predicted_pos[1] >= self.frame_size[1]):
                    is_safe = False
                    break
            
            # If path is safe, return this direction
            if is_safe:
                return alt_dir, distance
        
        # If no safe direction, return original direction
        return direction, distance

    def navigate(self):
        """Navigate the drone to destination with advanced avoidance"""
        # Calculate direction and distance
        direction, distance = self.calculate_direction()
        
        # Update current direction for visualization
        self.current_direction = direction
        
        # Move drone
        self.drone_pos[0] += direction[1] * self.speed
        self.drone_pos[1] += direction[0] * self.speed
        
        # Keep drone within frame
        self.drone_pos[0] = max(0, min(self.drone_pos[0], self.frame_size[0]))
        self.drone_pos[1] = max(0, min(self.drone_pos[1], self.frame_size[1]))
        
        # Check for mission completion or collision
        self.mission_complete = self.check_mission_complete()
        self.collision_detected = self.is_point_in_obstacle(self.drone_pos)

    def check_mission_complete(self):
        """Check if drone has reached destination"""
        distance = math.sqrt(
            (self.destination_pos[1] - self.drone_pos[1])**2 + 
            (self.destination_pos[0] - self.drone_pos[0])**2
        )
        
        return distance < 20

    def visualize(self):
        """Visualize the drone, its navigation, and direction"""
        # Draw current drone position
        cv2.circle(
            self.frame, 
            (int(self.drone_pos[1]), int(self.drone_pos[0])), 
            10, 
            (0, 0, 255), 
            -1  # Red drone marker
        )
        
        # Draw direction arrow
        arrow_length = 30
        arrow_end = (
            int(self.drone_pos[1] + self.current_direction[0] * arrow_length),
            int(self.drone_pos[0] + self.current_direction[1] * arrow_length)
        )
        cv2.arrowedLine(
            self.frame, 
            (int(self.drone_pos[1]), int(self.drone_pos[0])), 
            arrow_end, 
            (255, 0, 0),  # Blue arrow
            2
        )
        
        # Display score
        score_text = f"Successes: {self.score_tracker['successes']} | Failures: {self.score_tracker['failures']}"
        cv2.putText(
            self.frame, 
            score_text, 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Show the frame
        cv2.imshow('Drone Navigation Simulation', self.frame)

def main():
    # Create score tracker
    score_tracker = {'successes': 0, 'failures': 0}
    
    # Create drone navigation instance
    drone = DroneNavigation(score_tracker=score_tracker)
    
    # Simulation loop
    max_iterations = 100000
    for _ in range(max_iterations):
        # Create and move obstacles
        drone.create_dynamic_obstacles()
        
        # Navigate
        drone.navigate()
        
        # Visualize
        drone.visualize()
        
        # Check mission status
        if drone.mission_complete:
            print("Mission Accomplished! Drone reached destination.")
            score_tracker['successes'] += 1
            # Restart simulation
            drone = DroneNavigation(score_tracker=score_tracker)
        
        # Check for collision
        if drone.collision_detected:
            print("Collision! Restarting simulation.")
            score_tracker['failures'] += 1
            # Restart simulation
            drone = DroneNavigation(score_tracker=score_tracker)
        
        # Wait between frames
        key = cv2.waitKey(50)  # 50ms between frames
        
        # Exit if 'q' is pressed
        if key & 0xFF == ord('q'):
            break
    
    # Close all windows
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
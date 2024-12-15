import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, dt=0.1, sp=0.001, sm=0.05):
        # State dimension = 6 (x, y, vx, vy, ax, ay)
        # Measurement dimension = 2 (x, y)
        self.dt = dt
        
        # State transition matrix
        self.psi = np.array([
            [1, 0, dt, 0, dt**2/2, 0],
            [0, 1, 0, dt, 0, dt**2/2],
            [0, 0, 1, 0, dt, 0],
            [0, 0, 0, 1, 0, dt],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ])
        
        # Measurement matrix
        self.phi = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(6) * sp
        
        # Measurement noise covariance 
        self.R = np.eye(2) * sm
        
        # Initialize state and covariance
        self.x = np.array([-10, -150, 1, -2, 0, 0])
        self.P = np.eye(6)
        
    def predict(self):
        # Predict state
        self.x = self.psi @ self.x
        # Predict covariance
        self.P = self.psi @ self.P @ self.psi.T + self.Q
        return self.x[:2]  # Return predicted position
        
    def update(self, measurement):
        # Kalman gain
        S = self.phi @ self.P @ self.phi.T + self.R
        K = self.P @ self.phi.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - self.phi @ self.x
        self.x = self.x + K @ y
        
        # Update covariance
        self.P = (np.eye(6) - K @ self.phi) @ self.P
        return self.x[:2]  # Return updated position

class FixedLagSmoothing:
    pass

# Load observations
observations = np.load('data/observations.npy')

# Run Kalman filter
kf = KalmanFilter()
filtered_states = []

for obs in observations:
    kf.predict()
    filtered_state = kf.update(obs)
    filtered_states.append(filtered_state)

filtered_states = np.array(filtered_states)

# Visualize results
plt.figure(figsize=(10, 8))
plt.plot(observations[:, 0], observations[:, 1], 'r.', label='Observations', alpha=0.5)
plt.plot(filtered_states[:, 0], filtered_states[:, 1], 'b-', label='Filtered')
plt.legend()
plt.grid(True)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Kalman Filter: Observations vs. Filtered Estimates')
plt.show()
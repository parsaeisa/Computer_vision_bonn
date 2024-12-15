import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, dt=0.1, sp=0.001, sm=0.05):
        # State dimension: 6
        # Measurement dimension: 2
        self.dt = dt

        self.psi = np.array([
            [1, 0, dt, 0,  dt**2/2,  0],
            [0, 1, 0,  dt, 0,       dt**2/2],
            [0, 0, 1,  0,  dt,       0],
            [0, 0, 0,  1,  0,       dt],
            [0, 0, 0,  0,  1,        0],
            [0, 0, 0,  0,  0,        1]
        ])
        self.phi = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        self.Q = np.eye(6) * sp
        self.R = np.eye(2) * sm
        self.x = np.array([-10, -150, 1, -2, 0, 0])
        self.P = np.eye(6)
        
    def predict(self):
        self.x = self.psi @ self.x
        self.P = self.psi @ self.P @ self.psi.T + self.Q
        return self.x[:2]  # Return predicted position
        
    def update(self, measurement):
        # Kalman gain
        S = self.phi @ self.P @ self.phi.T + self.R + np.eye(2) * 1e-6
        K = self.P @ self.phi.T @ np.linalg.inv(S)
        
        # Update state
        y = measurement - self.phi @ self.x
        if np.isnan(y).any():
            return self.x[:2]

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.phi) @ self.P
        return self.x[:2]  # Return updated position

class FixedLagSmoothing:
    pass

observations = np.load('data/observations.npy')

kf = KalmanFilter()
filtered_states = []

for obs in observations:
    kf.predict()
    filtered_state = kf.update(obs)
    filtered_states.append(filtered_state)

filtered_states = np.array(filtered_states)

plt.figure(figsize=(10, 8))
plt.plot(observations[:, 0], observations[:, 1], 'r.', label='Observations', alpha=0.5)
plt.plot(filtered_states[:, 0], filtered_states[:, 1], 'b-', label='Filtered')
plt.legend()
plt.grid(True)
plt.title('Observations vs. Filtered Estimates')
plt.show()
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
        return self.x[:2]
        
    def update(self, measurement):
        # Kalman gain
        S = self.phi @ self.P @ self.phi.T + self.R + np.eye(2) * 1e-6
        K = self.P @ self.phi.T @ np.linalg.inv(S)

        y = measurement - self.phi @ self.x
        if np.isnan(y).any():
            return self.x[:2]

        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ self.phi) @ self.P
        return self.x[:2]

class FixedLagSmoothing:
    def __init__(self, lag=5, dt=0.1, sp=0.001, sm=0.05):
        self.lag = lag
        self.kf = KalmanFilter(dt, sp, sm)
        self.state_history = []
    
    def smooth(self, observations):
        smoothed_states = []
        
        for obs in observations:
            self.kf.predict()
            updated_state = self.kf.update(obs)
            self.state_history.append([self.kf.x.copy(), self.kf.P.copy()])

            if len(self.state_history) > self.lag:
                self._perform_smoothing()
                smoothed_states.append(self.state_history[0][0][:2])
                self.state_history.pop(0)
            
            else:
                smoothed_states.append(updated_state[:2])

        while len(self.state_history) > 1:
            self._perform_smoothing()
            smoothed_states.append(self.state_history[0][0][:2])
            self.state_history.pop(0)
        
        return smoothed_states

    def _perform_smoothing(self):
        """Perform fixed lag smoothing over the last two states in the history."""
        current_state, current_cov = self.state_history[-1]
        prev_state, prev_cov = self.state_history[-2]

        # Compute smoother gain
        psi_T = self.kf.psi.T
        smoother_gain = prev_cov @ psi_T @ np.linalg.inv(current_cov)
        
        # Update the previous state using smoother gain
        corrected_state = prev_state + smoother_gain @ (current_state - self.kf.psi @ prev_state)
        corrected_cov = prev_cov + smoother_gain @ (current_cov - prev_cov) @ smoother_gain.T
        
        # Replace the smoothed previous state
        self.state_history[-2][0] = corrected_state
        self.state_history[-2][1] = corrected_cov


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

#################################################################################################
##############                          Fix lag smoothing                          ##############
#################################################################################################
smoother = FixedLagSmoothing(lag=30)
smoothed_states = smoother.smooth(observations)

for i, state in enumerate(smoothed_states):
    print(f"Time Step {i}: Smoothed State: {state}")

smoothed_states = np.array(smoothed_states)

plt.plot(smoothed_states[:, 0], smoothed_states[:, 1], 'g-', label='Smoothed states')

#################################################################################################
##############                           Final plotting                            ##############
#################################################################################################
plt.legend()
plt.grid(True)
plt.title('Observations vs. Filtered Estimates')
plt.show()

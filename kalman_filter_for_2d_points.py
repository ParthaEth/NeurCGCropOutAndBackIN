import numpy as np


class ExtendedKalmanFilter:
    def __init__(self, num_points, dt, std_acc, std_meas):
        # std_acc: Represents the uncertainty or the expected variability in the acceleration of
        # the keypoints or bounding box across video frames. This parameter influences how much the
        # acceleration can vary from one frame to another, which is not directly observable but
        # estimated by the filter. i.e. Process noise
        # std_meas: How accurate are the measurements.
        self.num_points = num_points
        self.dt = dt
        self.state_dim = 2 * num_points + 4  # 2 coordinates per point + 2 velocity + 2 acceleration

        # Define the state transition matrix
        self.F = np.eye(self.state_dim)
        for i in range(num_points):
            self.F[i*2, -4] = dt  # vx influence on x
            self.F[i*2, -2] = 0.5 * dt**2  # ax influence on x
            self.F[i*2+1, -3] = dt  # vy influence on y
            self.F[i*2+1, -1] = 0.5 * dt**2  # ay influence on y

        # Process noise covariance
        q = std_acc**2
        self.Q = np.zeros((self.state_dim, self.state_dim))
        self.Q[-4:, -4:] = q * np.eye(4)

        # Measurement noise covariance
        self.R = std_meas**2 * np.eye(2 * num_points)

        # Observation matrix
        self.H = np.zeros((2 * num_points, self.state_dim))
        for i in range(num_points):
            self.H[2*i, 2*i] = 1
            self.H[2*i+1, 2*i+1] = 1

        # Initial state and covariance
        self.x = np.zeros((self.state_dim, 1))
        self.P = np.eye(self.state_dim)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self):
        return self.x

    def set_state(self, state):
        assert state.shape == (self.state_dim, 1), (f'Dimension missmatch expected statedim {(self.state_dim, 1)},'
                                                    f'got {state.shape}')
        self.x = state


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # Example usage
    num_points = 70  # 68 keypoints + 2 for the bounding box
    init_state = np.random.randn(2 * num_points + 4, 1)
    measurements = init_state[:2*num_points, :]
    dt = 0.1
    kf = ExtendedKalmanFilter(num_points=num_points, dt=dt, std_acc=5000, std_meas=0.0001)
    kf.set_state(init_state)  # numpoints plus 2 velocity plus 2 acceleration


    # Simulate some measurements and update
    states = []
    meas = []
    for i in range(20):
        states.append(kf.get_state()[:2])
        meas.append(measurements[:2])

        measurements = measurements + dt * np.tile(np.random.randn(2), num_points).reshape(2*num_points, 1)  # Simulated noisy measurements
        kf.predict()
        kf.update(measurements)

    states = np.array(states)
    meas = np.array(meas)

    plt.plot(states[:, 0, 0], states[:, 1, 0])
    plt.plot(meas[:, 0, 0], meas[:, 1, 0], 'r')
    for idx in range(len(meas)):
        plt.text(meas[idx, 0, 0], meas[idx, 1, 0], str(idx), color="blue", fontsize=12, ha='right')
    plt.show()

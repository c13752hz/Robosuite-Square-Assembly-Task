import numpy as np

class CanonicalSystem:
    """
    Skeleton of the discrete canonical dynamical system.
    """
    def __init__(self, dt: float, ax: float = 1.0):
        """
        Args:
            dt (float): Timestep duration.
            ax (float): Gain on the canonical decay.
        """
        # Initialize time parameters
        self.dt: float = dt
        self.ax: float = ax
        self.run_time: float = None  # TODO: set total runtime
        self.timesteps: int = None  # TODO: compute from run_time and dt
        self.x: float = None  # phase variable

    def reset(self) -> None:
        """
        Reset the phase variable to its initial value.
        """
        # TODO: implement
        self.x = 1.0
        pass

    def step(self, tau: float = 1.0, error_coupling: float = 1.0) -> float:
        """
        Advance the phase by one timestep.

        Returns:
            float: Updated phase value.
        """
        # TODO: implement update rule
        self.x -= self.ax * self.x * self.dt * error_coupling / tau
        return self.x

    def rollout(self, tau: float = 1.0, ec: float = 1.0) -> np.ndarray:
        """
        Generate the entire phase sequence.

        Returns:
            np.ndarray: Array of phase values over time.
        """
        # TODO: call reset() then repeatedly call step()
        self.reset()
        x_track = np.zeros(self.timesteps)
        for t in range(self.timesteps):
            x_track[t] = self.x
            self.step(tau, ec)
        return x_track

class DMP:
    """
    Skeleton of the discrete Dynamic Motor Primitive.
    """
    def __init__(
        self,
        n_dmps: int,
        n_bfs: int,
        dt: float = 0.01,
        y0: float = 0.0,
        goal: float = 1.0,
        ay: float = 25.0,
        by: float = None
    ):
        """
        Args:
            n_dmps (int): Number of dimensions.
            n_bfs (int): Number of basis functions per dimension.
            dt (float): Timestep duration.
            y0 (float|array): Initial state.
            goal (float|array): Goal state.
            ay (float|array): Attractor gain.
            by (float|array): Damping gain.
        """
        # TODO: initialize parameters

        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        self.dt = dt
        self.y0 = np.ones(n_dmps) * y0
        self.goal = np.ones(n_dmps) * goal
        self.ay = np.ones(n_dmps) * ay
        self.by = np.ones(n_dmps) * (by if by is not None else ay / 4.0)
        self.w = np.zeros((n_dmps, n_bfs))

        self.cs = CanonicalSystem(dt)
        self.cs.run_time = 2
        self.cs.timesteps = int(self.cs.run_time / self.dt)

        self.c = np.exp(-self.cs.ax * np.linspace(0, 1, self.n_bfs))
        self.h = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.c / self.cs.ax
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset trajectories and canonical system state.
        """
        # TODO: reset y, dy, ddy and call self.cs.reset()
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset()

    def imitate(self, y_des: np.ndarray) -> np.ndarray:
        """
        Learn DMP weights from a demonstration.

        Args:
            y_des (np.ndarray): Desired trajectory, shape (D, T).

        Returns:
            np.ndarray: Interpolated demonstration (D x T').
        """
        # TODO: interpolate, compute forcing term, solve for w
            # Ensure y_des has shape (D, T)
        if y_des.ndim == 1:
            y_des = y_des[None, :]  # Convert to (1, T)
        
        D, T = y_des.shape
        t_original = np.linspace(0, (T - 1) * self.dt, T)
        t_target = np.linspace(0, (self.cs.timesteps - 1) * self.dt, self.cs.timesteps)

        y_des_interp = np.zeros((D, self.cs.timesteps))
        for d in range(D):
            y_des_interp[d] = np.interp(t_target, t_original, y_des[d])

        self.reset_state()
        x_track = self.cs.rollout()

        dy_des = np.gradient(y_des_interp, self.dt, axis=1)
        ddy_des = np.gradient(dy_des, self.dt, axis=1)

        f_target = ((ddy_des * self.dt**2) -
                    self.ay[:, None] * (self.by[:, None] * (self.goal[:, None] - y_des_interp) - dy_des)) / (x_track + 1e-10)

        psi = np.exp(-self.h[:, None] * (x_track[None, :] - self.c[:, None])**2)

        for d in range(self.n_dmps):
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi[b] * f_target[d])
                denom = np.sum((x_track ** 2) * psi[b])
                self.w[d, b] = numer / (denom + 1e-10)

        return y_des_interp

    def rollout(
        self,
        tau: float = 1.0,
        error: float = 0.0,
        new_goal: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate a new trajectory from the DMP.

        Args:
            tau (float): Temporal scaling.
            error (float): Feedback coupling.
            new_goal (np.ndarray, optional): Override goal.

        Returns:
            np.ndarray: Generated trajectory (T x D).
        """
        # TODO: implement dynamical update loop
        if new_goal is not None:
            self.goal = new_goal
        self.reset_state()
        x_track = self.cs.rollout(tau)
        y_track = np.zeros((self.cs.timesteps, self.n_dmps))

        for t in range(self.cs.timesteps):
            x = x_track[t]
            psi = np.exp(-self.h * (x - self.c) ** 2)
            f = np.zeros(self.n_dmps)
            for d in range(self.n_dmps):
                f[d] = (psi @ self.w[d]) / (np.sum(psi) + 1e-10) * x
                self.ddy[d] = (
                    self.ay[d] * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d] * tau)
                    + f[d]
                ) / tau ** 2
                self.dy[d] += self.ddy[d] * self.dt
                self.y[d] += self.dy[d] * self.dt
                y_track[t, d] = self.y[d]

        return y_track

# ==============================
# DMP Unit test
# ==============================
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Test canonical system
    cs = CanonicalSystem(dt=0.05)
    x_track = cs.rollout()
    plt.figure()
    plt.plot(x_track, label='Canonical x')
    plt.title('Canonical System Rollout')
    plt.xlabel('Timestep')
    plt.ylabel('x')
    plt.legend()

    # Test DMP behavior with a sine-wave trajectory
    dt = 0.01
    T = 1.0
    t = np.arange(0, T, dt)
    y_des = np.sin(2 * np.pi * 2 * t)

    dmp = DMP(n_dmps=1, n_bfs=50, dt=dt)
    y_interp = dmp.imitate(y_des)
    y_run = dmp.rollout()

    plt.figure()
    plt.plot(t, y_des, 'k--', label='Original')
    plt.plot(np.linspace(0, T, y_interp.shape[1]), y_interp.flatten(), 'b-.', label='Interpolated')
    plt.plot(np.linspace(0, T, y_run.shape[0]), y_run.flatten(), 'r-', label='DMP Rollout')
    plt.title('DMP Imitation and Rollout')
    plt.xlabel('Time (s)')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()

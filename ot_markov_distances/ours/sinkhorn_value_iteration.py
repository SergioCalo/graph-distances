import numpy as np
from dataclasses import dataclass, field


@dataclass(frozen=False)
class Settings:
    eta: float  # entropy factor
    gamma: float  # discount factor
    N: int  # num of projections
    K: int  # num of ours iterations
    dimX: int
    dimY: int
    round: bool
    eta_decay: float


@dataclass(frozen=False, slots=False)
class Matrix2D:
    m: np.array = field(default_factory=lambda: np.array(''))
    rows: int = 1
    cols: int = 0

    def set_rand_2D_matrix(self, rows: int, cols: int):
        self.m = get_rand_2D_matrix(rows=rows, cols=cols)
        self.rows = rows
        self.cols = cols

    def set_2D_matrix(self, M: np.array):
        self.m = M.copy()
        self.rows, self.cols = self.m.shape

    def normalize(self):
        self.m = normalize_matrix(m=self.m)

    def repeat(self, rep_rows: int, rep_cols: int):
        m = np.repeat(self.m, rep_cols, axis=0)
        m = np.repeat(m, rep_rows, axis=1)
        return Matrix2D(m, m.shape[0], m.shape[1])

    def tile(self, t_rows: int, t_cols: int):
        m = np.tile(self.m, (t_rows, t_cols))
        return Matrix2D(m, m.shape[0], m.shape[1])

    def flatten(self):
        m = self.m.flatten()[:, None]
        rows, cols = self.m.shape
        return Matrix2D(m, rows, cols)

    def sum_along_rows(self):
        """ Produces one row after adding all elements column-wise """
        m = np.sum(self.m, axis=0)[None, :]
        rows, cols = m.shape
        return Matrix2D(m, rows, cols)

    def sum_along_cols(self):
        """ Produces one column after adding all elements row-wise """
        m = np.sum(self.m, axis=1)[:, None]
        rows, cols = m.shape
        return Matrix2D(m, rows, cols)

    def transpose(self):
        m = self.m.T
        rows, cols = m.shape
        return Matrix2D(m, rows, cols)

    def __mul__(self, other):
        return np.multiply(self.m, other.m)  # element-wise multiplication

    def __add__(self, other):
        return np.add(self.m, other.m)  # element-wise addition


def get_independent_coupling(Px: Matrix2D, Py: Matrix2D) -> Matrix2D:
    """ Compute the independent coupling of two transition kernels """
    """ Assumption: Px & Py are square matrices """
    rPx = Px.repeat(rep_rows=Py.rows, rep_cols=Py.cols)
    tPy = Py.tile(t_rows=Px.rows, t_cols=Px.cols)
    return Matrix2D(rPx * tPy, Px.rows * Py.rows, Px.cols * Py.cols)


def stationary_dist(m: np.array) -> np.ndarray:
    eigen_vals, eigen_vecs = np.linalg.eig(m.T)
    eigen_vec_1 = eigen_vecs[:, np.isclose(eigen_vals, 1)][:, 0]
    return (eigen_vec_1 / eigen_vec_1.sum()).real


def get_rand_2D_matrix(rows: int, cols: int) -> np.array:
    return np.random.rand(rows * cols).reshape((rows, cols))


def normalize_matrix(m: np.array) -> np.array:
    row_sums = m.sum(axis=1)
    m /= row_sums[:, np.newaxis]
    return m



def round_transpoly(X, r, c):
    A = X.copy()
    n1, n2 = A.shape
    r_A = np.sum(A, axis=1)

    for i in range(n1):
        scaling = min(1, r[i] / r_A[i])
        A[i, :] = scaling * A[i, :]

    c_A = np.sum(A, axis=0)

    for j in range(n2):
        scaling = min(1, c[j] / c_A[j])
        A[:, j] = scaling * A[:, j]

    r_A = np.sum(A, axis=1)[:, np.newaxis]
    c_A = np.sum(A, axis=0)
    err_r = r_A - r
    err_c = c_A - c

    if not np.all(err_r == 0) and not np.all(err_c == 0):
        A = A + np.outer(err_r, err_c) / np.sum(np.abs(err_r))

    return A
def compute_mu(nu, pi):
    mu = np.zeros(pi.shape)
    for xy in range(pi.shape[0]):
        for xy_prime in range(pi.shape[1]):
            mu[xy, xy_prime] = nu[xy] * pi[xy, xy_prime]
    return mu
def check_constraint_satisfaction(Pi, Px, Py):
    #print('Constraint satisfaction:')
    nu = stationary_dist(Pi.m)
    mu = compute_mu(nu, Pi.m)

    #print('sum over y x_prime y_prime of mu: ', mu.reshape((Px.rows, Py.rows, Px.cols, Py.cols)).sum(3).sum(2).sum(1))
    #print('nu_x: ', stationary_dist(Px.m))
    #print('sum over x x_prime y_prime of mu: ', mu.reshape((Px.rows, Py.rows, Px.cols, Py.cols)).sum(3).sum(2).sum(0))
    #print('nu_y: ', stationary_dist(Py.m))
    nu_prime_x = mu.reshape((Px.rows, Py.rows, Px.cols, Py.cols)).sum(3).sum(2).sum(1)
    nu_prime_y = mu.reshape((Px.rows, Py.rows, Px.cols, Py.cols)).sum(3).sum(2).sum(0)
    nu_x = stationary_dist(Px.m)
    nu_y = stationary_dist(Py.m)
    if not np.allclose(nu_prime_x, nu_x, rtol=1e-9) or not np.allclose(nu_prime_y, nu_y, rtol=1e-9):
        print("FAILED CONSTRAINT SATISFACTION TEST")


def update_Q(cost: Matrix2D, pi: Matrix2D, Q: Matrix2D, settings: Settings):
    """ Apply the update rule for the Q-function """
    """
    The arguments are 2D matrices of sizes:
    - |cost| = XxY
    - |pi| = XYxXY
    - |Q| = XYxXY
    """
    # Q_{k+1}(xy,x'y') = c(xy) + gamma * sum_{x",y"} pi_k(x"y"|x'y')*Q_k(x'y', x"y")
    # compute element-wise res1 = pi_k * Q_k
    # aggregate over x"y": res2 = res1 * 1_{xy} (1 * xy)
    # get flatten representation of cost: cost.flatten (xy * 1)
    # extend res2 & flatten cost to be xy * xy
    # Q = extended_res2 + extended_flatten_cost
    # return Q

    # OTHER
    # X, Y = cost.m.shape
    # CF = cost.m.flatten()[:, None].repeat(X * Y, 1)
    # QM = CF + settings.gamma * np.sum(pi.m * Q.m, 1)[None, :].repeat(X * Y, 0)

    # Create COST matrix # ToDo:
    #c = cost.flatten()
    #c = c.repeat(rep_rows=c.rows * c.cols, rep_cols=1)

    # Create pi_k*Q_k matrix
    res1 = pi * Q
    res2 = Matrix2D(res1, res1.shape[0], res1.shape[1])
    res2 = res2.sum_along_cols()
    res2 = res2.repeat(rep_rows=res2.rows * res2.cols, rep_cols=1)
    res2 = res2.transpose()

    # Update Q_{k+1} matrix
    qm = cost.m + settings.gamma * res2.m
    Q = Matrix2D(qm, qm.shape[0], qm.shape[1])

    return Q



def odd_update_policy(pi: Matrix2D, Px: Matrix2D, Q: Matrix2D, settings: Settings):
    """ Apply the update rule for the policy """
    """
    The arguments are 2D matrices of sizes:  
    - |pi| = XYxXY
    - |P| XYxXY (extended versions Px in XxX and Py in YxY)
    - |Q| = XYxXY 
    """
    # Javi pseudocode comments
    # norm(x') = sum_{y"} pi_k(x'y"|xy) * exp(-eta*Q_k(xy,x'y"))
    # pi_{k+1}(x'y'|xy) = pi_k(x'y'|xy) * exp(-eta*Q_k(xy,x'y')) * Px(x'|x) / norm(x')

    # Sergio code version
    X, Y = settings.dimX, settings.dimY
    Pi = pi.m.copy()
    Num = Pi * np.exp(-settings.eta * Q.m)
    Den = np.sum(Num.reshape((X * Y, X, Y)), 2).repeat(Y, 1)
    Pi[Den!=0] = (Num[Den!=0] / Den[Den!=0])
    Pi *= Px.m

    # Update pi from Pi
    pi = Matrix2D(Pi, Pi.shape[0], Pi.shape[1])

    return pi

def even_update_policy(pi: Matrix2D, Py: Matrix2D, Q: Matrix2D, settings: Settings):
    # Javi pseudocode comments
    # norm(y') = sum_{x"} pi_k(x"y'|xy) * exp(-eta*Q_k(xy,x"y'))
    # pi_{k+1}(x'y'|xy) = pi_k(x'y'|xy) * exp(-eta*Q_k(xy,x'y')) * Py(y'|y) / norm(y')

    # Sergio code version
    X, Y = settings.dimX, settings.dimY
    Pi = pi.m.copy()
    Num = Pi * np.exp(-settings.eta * Q.m)
    Den = np.tile(np.sum(Num.reshape((X * Y, X, Y)), 1), (1, X))
    Pi[Den!=0] = (Num[Den!=0] / Den[Den!=0])
    Pi *= Py.m

    # Update pi from Pi
    pi = Matrix2D(Pi, Pi.shape[0], Pi.shape[1])

    return pi

def evaluate_pi(pi: Matrix2D, pi_0: Matrix2D, cost: Matrix2D, settings: Settings):
    nu_0 = stationary_dist(pi_0.m)
    a = (np.eye(pi.rows, dtype=int) - settings.gamma*pi.transpose().m)
    b = (1-settings.gamma)*nu_0
    nu = np.linalg.solve(a,b)
    c = np.reshape(cost.m, (pi.rows, -1))
    distance = nu @ c
    return distance



def round(pi, Px: Matrix2D, Py: Matrix2D):

    for x in range(Px.rows):
        for y in range(Py.rows):
            idx = Py.rows * x + y
            pi.m[idx] = round_transpoly(np.reshape(pi.m[idx], (Px.rows, Py.rows)), Px.m[x, :][:, np.newaxis], Py.m[y, :]).flatten()

    return pi


def sinkhorn_value_iteration(Mx, My, cost_):
    settings = Settings(eta=3., gamma=0.95, N=1, K=50, dimX=Mx.shape[0], dimY=My.shape[1], round=True, eta_decay=False)
    Px = Matrix2D()
    Px.set_2D_matrix(M=np.array(Mx, dtype=float))
    Py = Matrix2D()
    Py.set_2D_matrix(M=np.array(My, dtype=float))
    cost = Matrix2D()
    cost.set_2D_matrix(M=np.array(cost_, dtype=float))
    pi_0 = get_independent_coupling(Px=Px, Py=Py)
    pi = Matrix2D(pi_0.m)
    Q = Matrix2D()
    #Q.set_2D_matrix(np.zeros((Px.rows * Py.rows, Px.cols * Py.cols)))
    Q.set_rand_2D_matrix(Px.rows * Py.rows, Px.cols * Py.cols) # ToDo: remove this and keep the initial Q_0

    # ToDo:
    X, Y = settings.dimX, settings.dimY
    #  1. create cost matrix (flatten version, ...)
    c = cost.flatten()
    cm = c.repeat(rep_rows=c.rows * c.cols, rep_cols=1)
    #  2. create Px matrix of X*X version
    # PXF = PX.repeat(Y, 0).repeat(Y, 1)
    pxm = Px.m.repeat(Y, 0).repeat(Y, 1)
    Pxm = Matrix2D(pxm, pxm.shape[0], pxm.shape[1])
    #  3. create Py matrix of Y*Y version
    # PYF = np.tile(PY, (X, X))
    pym = np.tile(Py.m, (X, X))
    Pym = Matrix2D(pym, pym.shape[0], pym.shape[1])

    for k in range(1, settings.K+1):
        if settings.eta_decay:
            settings.eta =  settings.eta_decay * settings.eta
        """ Step 1. Update Q matrix """
        for n in range(settings.N):
            Q = update_Q(cost=cm, pi=pi, Q=Q, settings=settings)

        """ Step 2. Update the policy """
        if (k & 1) == 1:  # k is odd
            pi = odd_update_policy(pi=pi, Px=Pxm, Q=Q, settings=settings)
        else:
            pi = even_update_policy(pi=pi, Py=Pym, Q=Q, settings=settings)

    if settings.round:
        pi = round(pi, Px, Py)
    distance = evaluate_pi(pi, pi_0, c, settings)
    #distance = 0
    return distance

def print_results(mu, mu_0, pi, pi_0, cost, verbose):
    if verbose == 2:
        print(f'Pi:\n{np.around(pi, decimals=3)}')
        print(f'Pi sum by columns={pi.sum(1)}')
        print(f'Pi total sum={pi.sum()}')
        print(f"pi_0:\n{np.around(pi_0, decimals=3)}")
        print(f'pi_0 sum by columns={pi_0.sum(1)}')

    if verbose == 1 or verbose == 2:
        c = np.reshape(cost, (pi.shape[0], -1))
        print('Final cost: ', mu @ c)
        print('Independent coupling cost: ', mu_0 @ c)
        # print('cost: ', np.sum(pi @ c) / pi.shape[0])
        # print('Pi_0 cost: ', np.sum(pi_0 @ c) / pi.shape[0])

    if verbose == 0:
        pass



import numpy as np
from dataclasses import dataclass, field


'''def old_get_independent_coupling(P: np.array, Q: np.array) -> np.array:
    """ Function to compute the independent coupling between P and Q """
    dx, dx_col = P.shape
    dy, dy_col = Q.shape
    
    pi_0 = np.zeros((dx*dy, dx_col*dy_col))
    for x_row in range(dx):
        for x_col in range(dx_col):
            for y_row in range(dy):
                for y_col in range(dy_col):
                    idx1 = dy*(x_row)+y_row
                    idx2 = dy*(x_col)+y_col
                    pi_0[idx1, idx2] = P[x_row, x_col]*Q[y_row, y_col]
                    
    return pi_0
'''

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
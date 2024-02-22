import numpy as np
import radau


def Robertson(DAE=False):
    """Robertson problem of semi-stable chemical reaction, see mathworks and Shampine2005.

    References:
    -----------
    mathworks: https://de.mathworks.com/help/matlab/math/solve-differential-algebraic-equations-daes.html#bu75a7z-5 \\
    Shampine2005: https://doi.org/10.1016/j.amc.2004.12.011
    """

    n = 3

    if DAE:
        var_index = np.array([0, 0, 1])
    else:
        var_index = np.zeros(n)
        # var_index = None

    def mas(am, rpar, ipar, n, lmas):
        am[0, 0] = 1
        am[1, 1] = 1
        if not DAE:
            am[2, 2] = 1

    def fcn(x, y, f, rpar, ipar, n):
        y1, y2, y3 = y
        f = np.zeros(n, dtype=float)
        f[0] = -0.04 * y1 + 1e4 * y2 * y3
        f[1] = 0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2
        if DAE:
            f[2] = y1 + y2 + y3 - 1
        else:
            f[2] = 3e7 * y2**2

    def jac(t, y):
        raise NotImplementedError

    return mas, fcn, jac, var_index, n


if __name__ == "__main__":
    M, f, jac, var_index, n = Robertson()

    t0 = 0
    t1 = 1e-1
    h0 = 1e-3
    y0 = np.array([1, 0, 0], dtype=float)
    # y_dot0 = f(t0, y0)

    rtol = 1e-3
    atol = 1e-3
    itol = 0

    ijac = 0
    mjac = n
    mujac = n  # TODO:???

    imas = 1
    mlmas = n
    mumas = n  # TODO:???

    sol_t = []
    sol_y = []

    def solout(t, y):
        sol_t.append(y)
        sol_y.append(y)

    iout = 1

    # lwork = N*(LJAC+LMAS+NSMAX*LE+3*NSMAX+3)+20
    lwork = 1000
    work = np.zeros(lwork, dtype=float)
    # liwork = (2+(NSMAX-1)/2)*N+20
    liwork = 1000
    iwork = np.zeros(liwork, dtype=int)

    rpar = np.zeros(10, dtype=float)
    ipar = np.zeros(10, dtype=int)
    # rpar = 0
    # ipar = 0

    idid = -99

    print(radau.radau.__doc__)

    sol = radau.radau(
        f,
        t0,
        y0,
        t1,
        h0,
        rtol,
        atol,
        itol,
        jac,
        ijac,
        mjac,
        mujac,
        M,
        imas,
        mlmas,
        mumas,
        solout,
        iout,
        work,
        iwork,
        rpar,
        ipar,
        idid,
        # # optional arguments
        # n,
        # lwork,
        # liwork,
    )


import sympy as sp
from sympy import expand_complex
import numpy as np
from numpy.linalg import eig  # or from scipy.linalg import eig

class AnalyticalODE:
    # Eigenvalue problem for ODEs
    def eigenvalue_method(self, A, y0):
        if A.shape[0] != A.shape[1]:
            raise ValueError("Matrix A must be square.")

        # Get eigenvalues and eigenvectors (NumPy)
        eigvals, eigvecs_np = eig(np.array(A).astype(np.float64))

        # Convert eigenvectors to SymPy column matrices
        eigvecs = [sp.Matrix(eigvecs_np[:, i]) for i in range(len(eigvals))]
        eigvals = [sp.nsimplify(ev) for ev in eigvals]  # Optional: clean them up

        y0 = sp.Matrix(y0)

        # Form the matrix of eigenvectors
        V = sp.Matrix.hstack(*eigvecs)

        # Solve for c: V * c = y0 → c = V.inv() * y0
        c = V.inv() * y0

        # Construct the symbolic solution
        t = sp.symbols('t')
        n = len(y0)  # assuming y0 is a list or vector
        y_t = sum(
        (c[i] * eigvecs[i] * sp.exp(eigvals[i] * t) for i in range(len(eigvals))),
        start=sp.zeros(n, 1)
        )
        # Expand the complex expression to get a real-valued solution (for complex eigenvalues)
        y_t = expand_complex(y_t)  

        return y_t
A = sp.Matrix([[0, 1], [-2, -3]])
y0 = sp.Matrix([1, 0])
analytical_ode = AnalyticalODE()
solution = analytical_ode.eigenvalue_method(A, y0)
print("Symbolic solution:", solution)
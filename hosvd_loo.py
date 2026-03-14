import torch
import tensorly as tl
from tensorly.decomposition import tucker
from tensorly.tucker_tensor import tucker_to_tensor

tl.set_backend('pytorch')

def remove_slice_0(X, i):
    """
    Given X of shape (n, a, b), remove slice i from dim=0.
    Returns subX of shape (n-1, a, b) and a mask indicating indices used.
    """
    n = X.shape[0]
    mask = torch.arange(n, device=X.device) != i
    subX = X[mask]
    return subX, mask

def partial_tucker_core(G, U_1, U_2):
    """
    Build a (r0, a, b) tensor by applying mode-1 product with U_1
    and mode-2 product with U_2, but using an identity matrix on mode-0.

    In Tucker terms, it's tucker_to_tensor((G, [I_{r0}, U_1, U_2])),
    which results in shape (r0, a, b).

    We'll use it when solving for the missing row in the mode-0 factor.
    """
    r0, r1, r2 = G.shape

    # We'll create an identity factor for mode-0
    I0 = torch.eye(r0, dtype=G.dtype, device=G.device)

    # Reconstruct a partial 3D tensor of shape (r0, a, b)
    # by ignoring the factor on mode-0 (except an identity).
    partial_reconstruction = tucker_to_tensor((G, [I0, U_1, U_2]))
    # shape => (r0, a, b)
    return partial_reconstruction

def fit_missing_slice(M_i, G, U_1, U_2):
    """
    Solve for the row (alpha in R^(r0)) in the mode-0 factor that best
    reconstructs the missing slice M_i (shape (a,b)).

    i.e. we want to minimize || M_i - sum_r alpha[r] * partial[r,:,:] ||^2
    where 'partial' = (r0,a,b).

    We'll do a least-squares solution: flatten (a,b) => (a*b),
    flatten partial => shape (r0, a*b),
    and solve alpha for alpha * partial_flat ~ M_i_flat.
    """
    # partial => shape (r0, a, b)
    partial = partial_tucker_core(G, U_1, U_2)  # (r0, a, b)

    # Flatten partial => (r0, a*b)
    partial_flat = partial.reshape(partial.shape[0], -1)
    # Flatten M_i => (a*b)
    M_i_flat = M_i.reshape(-1)

    # We want alpha in R^(r0) s.t. partial_flat^T @ alpha ~ M_i_flat
    # Let A = partial_flat^T => shape((a*b, r0)), y = M_i_flat => shape(a*b)
    A = partial_flat.t()  # (a*b, r0)
    y = M_i_flat  # (a*b)

    # Solve least squares: alpha = argmin ||A@alpha - y||^2
    # PyTorch >= 1.9 has torch.linalg.lstsq
    # returns object with .solution
    alpha_solution = torch.linalg.lstsq(A, y.unsqueeze(-1)).solution.squeeze(-1)
    # shape => (r0,)

    return alpha_solution

def reconstruct_with_missing_slice(X, i, core, factors, mask):
    """
    Given a Tucker decomposition (core,factors) of shape (n-1,a,b),
    build a new factor for mode-0 of shape (n,r0) that includes
    the 'fitted' row for slice i.

    Return the reconstructed (n,a,b) approximation.
    """
    # factors => [U0, U1, U2], where U0 is (n-1, r0)
    U0_sub, U1, U2 = factors
    r0 = U0_sub.shape[1]
    n = X.shape[0]

    # 1) Build a new U0_full => shape (n, r0)
    U0_full = torch.empty((n, r0), dtype=U0_sub.dtype, device=U0_sub.device)
    # Fill the entries from U0_sub, skipping index i
    j_sub = 0
    for idx in range(n):
        if idx == i:
            continue
        U0_full[idx] = U0_sub[j_sub]
        j_sub += 1

    # 2) Solve for row i in U0_full
    M_i = X[i]  # shape (a,b)
    alpha = fit_missing_slice(M_i, core, U1, U2)
    U0_full[i] = alpha

    # 3) Reconstruct full tensor => shape(n,a,b)
    # re-assemble new Tucker factors
    new_factors = [U0_full, U1, U2]

    X_approx = tucker_to_tensor((core, new_factors))  # shape(n,a,b)
    return X_approx

def leave_one_out_hosvd_error(X, ranks):
    """
    For each i in [0..n-1], remove slice i from X (shape(n,a,b)),
    do Tucker decomposition on (n-1,a,b),
    embed the missing slice i back,
    reconstruct the full (n,a,b) => measure Frobenius error.
    
    Return a list/array of errors. The bigger the error, 
    the more unique info that slice i contributed.
    """
    n = X.shape[0]
    device = X.device

    errors = []
    for i in range(n):
        # Remove slice i
        subX, mask = remove_slice_0(X, i)  # shape(n-1,a,b)

        # Decompose subX
        core, factors = tucker(subX, rank=ranks, init='svd')
        # factors => [U0_sub(n-1,r0), U1(a,r1), U2(b,r2)], core(r0,r1,r2)

        # Embed + reconstruct the full shape(n,a,b)
        X_approx = reconstruct_with_missing_slice(X, i, core, factors, mask)

        # Measure Frobenius norm error
        err = (X - X_approx).norm().item()
        errors.append(err)

    return torch.tensor(errors, device=device)

# -------------------------------------------------------------------
# Usage Example
if __name__ == "__main__":
    # Suppose we have X of shape (n,a,b)
    n, a, b = 5, 4, 3
    X = torch.randn(n, a, b)

    # Choose Tucker ranks for each mode
    # e.g. rank for mode-0 could be n or n-1,
    # and for (a,b), pick something <= (a,b).
    ranks = [min(n, 4), min(a, 4), min(b, 3)]

    # Compute leave-one-out errors
    errors = leave_one_out_hosvd_error(X, ranks)
    print("Errors for each removed slice:", errors)
    print("Index with highest error => most 'information'")
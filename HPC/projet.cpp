#include <iostream>
#include <mpi.h>
#include <cmath>
#include <vector>

std::vector<double> richardson_poisson_1d(
    const std::vector<double>& f,
    double u0,
    double u1,
    double omega,
    int maxIters,
    double tol,
    const std::vector<double>& u_init = {}) {
    const int N = static_cast<int>(f.size());
    if (N <= 0) throw std::invalid_argument("N must be positive (f must be non-empty).");
    if (maxIters <= 0) throw std::invalid_argument("maxIters must be positive.");
    if (tol <= 0.0) throw std::invalid_argument("tol must be positive.");
    if (!(omega > 0.0)) throw std::invalid_argument("omega must be > 0.");

    const double h = 1.0 / (N + 1.0);
    const double inv_h2 = 1.0 / (h * h);

    // Build RHS b = f + boundary contributions
    std::vector<double> b(N);
    for (int i = 0; i < N; ++i) b[i] = f[i];
    b[0] += u0 * inv_h2;
    b[N - 1] += u1 * inv_h2;

    // Initial guess
    std::vector<double> u(N, 0.0);
    if (!u_init.empty()) {
        if (static_cast<int>(u_init.size()) != N)
            throw std::invalid_argument("u_init must have the same size as f.");
        u = u_init;
    }

    // Helper: compute A*u (tridiagonal apply with Dirichlet BCs)
    auto apply_A = [&](const std::vector<double>& x, std::vector<double>& Ax) {
        Ax.assign(N, 0.0);
        for (int i = 0; i < N; ++i) {
            const double xim1 = (i == 0) ? u0 : x[i - 1];
            const double xip1 = (i == N - 1) ? u1 : x[i + 1];
            Ax[i] = (2.0 * x[i] - xim1 - xip1) * inv_h2;
        }
    };

    // Norm helpers
    auto norm2 = [](const std::vector<double>& v) {
        long double s = 0.0L;
        for (double vi : v) s += static_cast<long double>(vi) * vi;
        return std::sqrt(static_cast<double>(s));
    };

    const double bnorm = std::max(norm2(b), 1e-30); // avoid division by 0

    std::vector<double> Au(N), r(N);

    for (int k = 0; k < maxIters; ++k) {
        apply_A(u, Au);

        // r = b - A*u
        for (int i = 0; i < N; ++i) r[i] = b[i] - Au[i];

        // Check stopping criterion
        if (norm2(r) / bnorm <= tol) break;

        // u <- u + omega*r
        for (int i = 0; i < N; ++i) u[i] += omega * r[i];
    }

    return u;
}  
int main(int argc,char *argv[]) {
  

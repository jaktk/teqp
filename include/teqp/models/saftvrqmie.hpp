#pragma once

/**
 SAFT-VRQ Mie: the Feynman-Hibbs corrected SAFT-VR Mie equation of state for spherical molecules of
 
 A. Aasen, M. Hammer, Å. Ervik, E. A. Müller, Ø. Wilhelmsen, "Equation of state and force fields for
 Feynman-Hibbs-corrected Mie fluids. I. Application to pure helium, neon, hydrogen, and deuterium",
 J. Chem. Phys. 151, 064508 (2019), https://doi.org/10.1063/1.5111364
 
 A. Aasen, M. Hammer, E. A. Müller, Ø. Wilhelmsen, "... II. Application to mixtures of helium, neon,
 hydrogen, and deuterium", J. Chem. Phys. 152, 074507 (2020), https://doi.org/10.1063/1.5136079
 
 Where the printed equations and the reference implementation in thermopack (SINTEF, MIT/Apache-2.0,
 https://github.com/thermotools/thermopack, commit 1884204, version 2.2.3) differ, this follows
 thermopack: the prefactor x0 = sigma_ij/d_ij with I and J at sigma_eff,ij/d_ij, the factor 2 on the
 cross terms of u^2, the Carnahan-Starling residual for the pure hard-sphere fluid, and the signs of
 a2 and a3. Barker-Henderson diameters are integrated for every pair from the cross potential with
 the 21-point Gauss-Kronrod rule, and the hard-sphere term is the non-additive mixture of Santos et al.
 */

#include "teqp/types.hpp"
#include "teqp/constants.hpp"
#include "teqp/exceptions.hpp"
#include "nlohmann/json.hpp"
#include <Eigen/Dense>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace teqp::SAFTVRQMie {

/// Parameters of one spherical (m = 1) fluid
struct PureParameters{
    std::string name;
    double sigma_m, epsilon_over_k, lambda_a, lambda_r, mass_kg;
};

/// Parameters of one pair (i, j), i == j included; combining rules of Part II Eqs. 2-8
struct PairParameters{
    double sigma;     ///< sigma_ij in m, with l_ij
    double eps;       ///< epsilon_ij/k_B in K, with k_ij
    double la, lr;    ///< lambda_a,ij and lambda_r,ij
    double C;         ///< Mie prefactor
    double Q1a, Q1r, Q2a, Q2r;  ///< Feynman-Hibbs coefficients Q1(lambda) = lambda(lambda-1), Q2 = 1/2 lambda(lambda-1)(lambda+1)(lambda+2)
    double Dparam;    ///< h^2/(96 mu_ij k_B pi^2), m^2 K; D_ij(T) = Dparam/T
    int fh;           ///< Feynman-Hibbs order, 1 or 2
};

inline PairParameters make_pair(const PureParameters& pi, const PureParameters& pj, double kij, double lij, int fh, double h_Js){
    if (fh != 1 && fh != 2){
        throw teqp::InvalidArgument("SAFT-VRQ Mie: the Feynman-Hibbs order must be 1 or 2");
    }
    PairParameters p;
    p.fh = fh;
    p.la = 3.0 + std::sqrt((pi.lambda_a - 3.0)*(pj.lambda_a - 3.0));
    p.lr = 3.0 + std::sqrt((pi.lambda_r - 3.0)*(pj.lambda_r - 3.0));
    p.sigma = 0.5*(pi.sigma_m + pj.sigma_m)*(1.0 - lij);
    p.eps = (1.0 - kij)*std::sqrt(std::pow(pi.sigma_m, 3)*std::pow(pj.sigma_m, 3))*std::sqrt(pi.epsilon_over_k*pj.epsilon_over_k)/std::pow(p.sigma, 3);
    p.C = p.lr/(p.lr - p.la)*std::pow(p.lr/p.la, p.la/(p.lr - p.la));
    p.Q1a = p.la*(p.la - 1.0); p.Q1r = p.lr*(p.lr - 1.0);
    p.Q2a = 0.5*p.la*(p.la - 1.0)*(p.la + 1.0)*(p.la + 2.0);
    p.Q2r = 0.5*p.lr*(p.lr - 1.0)*(p.lr + 1.0)*(p.lr + 2.0);
    double mu = pi.mass_kg*pj.mass_kg/(pi.mass_kg + pj.mass_kg);
    p.Dparam = h_Js*h_Js/(96.0*mu*constants::k_B*constants::PI*constants::PI);
    return p;
}

/// Lafitte et al. 2013 Table 3 / Part I Table I: phi[n][k-1], n = 0..6, k = 1..6
inline constexpr std::array<std::array<double, 6>, 7> PHI{{
    {7.5365557, -359.44, 1550.9, -1.19932, -1911.28, 9236.9},
    {-37.60463, 1825.6, -5070.1, 9.063632, 21390.175, -129430.0},
    {71.745953, -3168.0, 6534.6, -17.9482, -51320.7, 357230.0},
    {-46.83552, 1884.2, -3288.7, 11.34027, 37064.54, -315530.0},
    {-2.467982, -0.82376, -2.7171, 20.52142, 1103.742, 1390.2},
    {-0.50272, -3.1935, 2.0883, -56.6377, -3264.61, -4518.2},
    {8.0956883, 3.709, 0.0, 40.53683, 2556.181, 4241.6}
}};

/// 21-point Gauss-Kronrod nodes on [-1, 1] in ascending order and their weights (thermopack quadratures.f90:17-39, 316-322, 401-407)
inline constexpr std::array<double, 21> GK21_X{
    -0.995657163025808080735527280689003, -0.973906528517171720077964012084452, -0.930157491355708226001207180059508,
    -0.865063366688984510732096688423493, -0.780817726586416897063717578345042, -0.679409568299024406234327365114874,
    -0.562757134668604683339000099272694, -0.433395394129247190799265943165784, -0.294392862701460198131126603103866,
    -0.148874338981631210884826001129720, 0.0,
    0.148874338981631210884826001129720, 0.294392862701460198131126603103866, 0.433395394129247190799265943165784,
    0.562757134668604683339000099272694, 0.679409568299024406234327365114874, 0.780817726586416897063717578345042,
    0.865063366688984510732096688423493, 0.930157491355708226001207180059508, 0.973906528517171720077964012084452,
    0.995657163025808080735527280689003};
inline constexpr std::array<double, 21> GK21_W{
    0.011694638867371874278064396062192, 0.032558162307964727478818972459390, 0.054755896574351996031381300244580,
    0.075039674810919952767043140916190, 0.093125454583697605535065465083366, 0.109387158802297641899210590325805,
    0.123491976262065851077958109831074, 0.134709217311473325928054001771707, 0.142775938577060080797094273138717,
    0.147739104901338491374841515972068, 0.149445554002916905664936468389821,
    0.147739104901338491374841515972068, 0.142775938577060080797094273138717, 0.134709217311473325928054001771707,
    0.123491976262065851077958109831074, 0.109387158802297641899210590325805, 0.093125454583697605535065465083366,
    0.075039674810919952767043140916190, 0.054755896574351996031381300244580, 0.032558162307964727478818972459390,
    0.011694638867371874278064396062192};

/// u_ij(r)/k_B in K and its first two derivatives in r, Part II Eq. 1 (thermopack saftvrmie_hardsphere.f90:1189-1387)
template<typename RType, typename DType>
auto pair_potential(const PairParameters& p, const RType& r, const DType& D){
    using num = std::common_type_t<RType, DType>;
    const double la = p.la, lr = p.lr;
    num a = pow(p.sigma/r, la), b = pow(p.sigma/r, lr);
    num U = b - a;
    num U1 = (-lr*b + la*a)/r;
    num U2 = (lr*(lr + 1.0)*b - la*(la + 1.0)*a)/(r*r);
    if (p.fh >= 1){
        U += D/(r*r)*(p.Q1r*b - p.Q1a*a);
        U1 += -D/(r*r*r)*((lr + 2.0)*p.Q1r*b - (la + 2.0)*p.Q1a*a);
        U2 += D/(r*r*r*r)*((lr + 3.0)*(lr + 2.0)*p.Q1r*b - (la + 3.0)*(la + 2.0)*p.Q1a*a);
    }
    if (p.fh >= 2){
        U += D*D/(r*r*r*r)*(p.Q2r*b - p.Q2a*a);
        U1 += -D*D/(r*r*r*r*r)*((lr + 4.0)*p.Q2r*b - (la + 4.0)*p.Q2a*a);
        U2 += D*D/(r*r*r*r*r*r)*((lr + 5.0)*(lr + 4.0)*p.Q2r*b - (la + 5.0)*(la + 4.0)*p.Q2a*a);
    }
    const double f = p.C*p.eps;
    return std::make_tuple(num(f*U), num(f*U1), num(f*U2));
}

/// Newton on g(r) = 0 from r_start: converged in double (T at its base value), then N_AD steps in the
/// type of T, which make the root's derivatives in T exact to order 2^N_AD - 1
template<typename TType, typename G>
TType newton_root(const G& g, double r_start, const TType& T){
    constexpr int N_AD = 3;
    const double Tb = getbaseval(T);
    double r = r_start;
    bool converged = false;
    for (int it = 0; it < 100; ++it){
        auto [val, der] = g(r, Tb);
        double dr = -val/der;
        r += dr;
        if (std::abs(dr) <= 4*std::numeric_limits<double>::epsilon()*std::abs(r)){ converged = true; break; }
    }
    if (!converged){
        throw teqp::IterationError("SAFT-VRQ Mie: a Newton root did not converge in 100 steps");
    }
    TType rT = r;
    for (int k = 0; k < N_AD; ++k){
        auto [val, der] = g(rT, T);
        rT = rT - val/der;
    }
    return rT;
}

template<typename TType>
struct TemperatureQuantities{
    TType D, Dt, sigma_eff, eps_eff, r0, d, alpha;
    std::array<TType, 6> f;
};

/// sigma_eff, eps_eff, the Barker-Henderson diameter d and alpha, f_k(alpha) of one pair at T (EQUATIONS.md sections 4, 6, 7)
template<typename TType>
auto temperature_quantities(const PairParameters& p, const TType& T){
    TemperatureQuantities<TType> q;
    q.D = p.Dparam/T;
    q.Dt = q.D/(p.sigma*p.sigma);
    // sigma_eff: u(sigma_eff) = 0, from r = sigma
    q.sigma_eff = newton_root([&](const auto& r, const auto& TT){
        auto [U, U1, U2] = pair_potential(p, r, forceeval(p.Dparam/TT)); return std::make_tuple(U, U1); }, p.sigma, T);
    // eps_eff = -u(r_min), du/dr(r_min) = 0, from r = sigma
    auto rmin = newton_root([&](const auto& r, const auto& TT){
        auto [U, U1, U2] = pair_potential(p, r, forceeval(p.Dparam/TT)); return std::make_tuple(U1, U2); }, p.sigma, T);
    q.eps_eff = -std::get<0>(pair_potential(p, rmin, q.D));
    // r0: u(r0)/T = -ln(eps_mach); the integrand of d is 1 to machine precision on [0, r0]
    const double target = -std::log(std::numeric_limits<double>::epsilon());
    q.r0 = newton_root([&](const auto& r, const auto& TT){
        auto [U, U1, U2] = pair_potential(p, r, forceeval(p.Dparam/TT)); return std::make_tuple(forceeval(U/TT - target), forceeval(U1/TT)); }, 0.7*getbaseval(q.sigma_eff), T);
    // d = r0 + integral_{r0}^{sigma_eff} [1 - exp(-u/T)] dr, GK21, summed in node order
    TType half = (q.sigma_eff - q.r0)/2.0;
    q.d = q.r0;
    for (std::size_t k = 0; k < 21; ++k){
        TType rk = half*GK21_X[k] + half + q.r0;
        auto U = std::get<0>(pair_potential(p, rk, q.D));
        q.d += half*GK21_W[k]*(1.0 - exp(-U/T));
    }
    // dimensionless van der Waals energy, Part I Eqs. 18-21 (thermopack calcAlpha)
    const double la = p.la, lr = p.lr;
    TType sr = p.sigma/q.sigma_eff;
    TType al = pow(sr, la)/(la - 3.0) - pow(sr, lr)/(lr - 3.0);
    if (p.fh >= 1){ al += q.Dt*(p.Q1a*pow(sr, la + 2.0)/(la - 1.0) - p.Q1r*pow(sr, lr + 2.0)/(lr - 1.0)); }
    if (p.fh >= 2){ al += q.Dt*q.Dt*(p.Q2a*pow(sr, la + 4.0)/(la + 1.0) - p.Q2r*pow(sr, lr + 4.0)/(lr + 1.0)); }
    q.alpha = p.C*p.eps/q.eps_eff*al;
    for (int k = 0; k < 6; ++k){
        TType num = 0.0, den = 1.0;
        for (int n = 0; n < 4; ++n){ num += PHI[n][k]*pow(q.alpha, n); }
        for (int n = 4; n < 7; ++n){ den += PHI[n][k]*pow(q.alpha, n - 3); }
        q.f[k] = num/den;
    }
    return q;
}

} // namespace teqp::SAFTVRQMie

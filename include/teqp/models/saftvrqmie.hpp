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
#include "teqp/json_tools.hpp"
#include "nlohmann/json.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
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
    /// Feynman-Hibbs coefficients Q1(l) = l(l-1) and Q2(l) = 1/2 l(l-1)(l+1)(l+2), l = lambda_a or lambda_r
    double Q1a, Q1r, Q2a, Q2r;
    double Dparam;    ///< h^2/(96 mu_ij k_B pi^2), m^2 K; D_ij(T) = Dparam/T
    int fh;           ///< Feynman-Hibbs order, 1 or 2
    std::string label;  ///< "name_i-name_j", for error messages
};

inline PairParameters make_pair(const PureParameters& pi, const PureParameters& pj, double kij, double lij,
                                int fh, double h_Js){
    if (fh != 1 && fh != 2){
        throw teqp::InvalidArgument("SAFT-VRQ Mie: the Feynman-Hibbs order must be 1 or 2");
    }
    PairParameters p;
    p.fh = fh;
    p.label = pi.name + "-" + pj.name;
    p.la = 3.0 + std::sqrt((pi.lambda_a - 3.0)*(pj.lambda_a - 3.0));
    p.lr = 3.0 + std::sqrt((pi.lambda_r - 3.0)*(pj.lambda_r - 3.0));
    p.sigma = 0.5*(pi.sigma_m + pj.sigma_m)*(1.0 - lij);
    p.eps = (1.0 - kij)*std::sqrt(std::pow(pi.sigma_m, 3)*std::pow(pj.sigma_m, 3))
            *std::sqrt(pi.epsilon_over_k*pj.epsilon_over_k)/std::pow(p.sigma, 3);
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

/// 21-point Gauss-Kronrod nodes on [-1, 1] in ascending order and their weights
/// (thermopack quadratures.f90:17-39, 316-322, 401-407)
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

/// Lengths inside the pair potential and its roots are in angstrom: in metres r^6 is ~1e-57, and the
/// recursive-squaring inverse of multicomplex numbers (MultiComplex.hpp:545-554) underflows at order >= 3.
/// The classical model does the same (saftvrmie.hpp: sigma_A = sigma_m*1e10).
inline constexpr double M_TO_A = 1e10;

/// u_ij(r)/k_B in K and its first two derivatives in r, Part II Eq. 1
/// (thermopack saftvrmie_hardsphere.f90:1189-1387).
/// r in angstrom, D in angstrom^2 (D = 1e20*Dparam/T); U1 in K/angstrom, U2 in K/angstrom^2.
template<typename RType, typename DType>
auto pair_potential(const PairParameters& p, const RType& r, const DType& D){
    using num = std::common_type_t<RType, DType>;
    const double la = p.la, lr = p.lr, s = p.sigma*M_TO_A;
    num a = pow(s/r, la), b = pow(s/r, lr);
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

/// Newton on g(r) = 0 from r_start > 0: converged in double (T at its base value), then N_AD steps in the
/// type of T, which make the root's derivatives in T exact to order 2^N_AD - 1.
///
/// The double stage never steps below half the current iterate. For r0 the function u/T + ln(eps_mach) is
/// convex and decreasing, and from 0.7 sigma_eff the first plain Newton step overshoots: below r/2 for He FH1
/// from 111 K, and to r <= 0 with a diverging iteration from 199.8 K. Once an iterate is left of the root,
/// Newton on a convex decreasing function (r0, sigma_eff) or a concave increasing one (du/dr for r_min)
/// climbs monotonically to it, so the guard acts at most on the first steps and the iterate stays positive.
/// A non-finite iterate throws.
template<typename TType, typename G>
TType newton_root(const G& g, double r_start, const TType& T, const char* root, const PairParameters& p){
    constexpr int N_AD = 3;
    const double Tb = getbaseval(T);
    double r = r_start;
    bool converged = false;
    for (int it = 0; it < 100; ++it){
        auto [val, der] = g(r, Tb);
        double r_new = std::max(r - val/der, 0.5*r);
        double dr = r_new - r;
        r = r_new;
        if (!std::isfinite(r)){ break; }
        if (std::abs(dr) <= 4*std::numeric_limits<double>::epsilon()*std::abs(r)){ converged = true; break; }
    }
    if (!converged){
        throw teqp::IterationError("SAFT-VRQ Mie: the Newton iteration for " + std::string(root) + " of pair " + p.label
                                   + " did not converge at T = " + std::to_string(Tb) + " K");
    }
    if constexpr (std::is_same_v<std::decay_t<TType>, double>){
        return r;
    }
    else{
        TType rT = r;
        for (int k = 0; k < N_AD; ++k){
            auto [val, der] = g(rT, T);
            rT = rT - val/der;
        }
        return rT;
    }
}

template<typename TType>
struct TemperatureQuantities{
    TType D, Dt, sigma_eff, eps_eff, r0, d, alpha;  ///< D in m^2, sigma_eff, r0 and d in m, eps_eff in K
    std::array<TType, 6> f;
};

/// sigma_eff, eps_eff, the Barker-Henderson diameter d and alpha, f_k(alpha) of one pair at T
/// (EQUATIONS.md sections 4, 6, 7)
template<typename TType>
auto temperature_quantities(const PairParameters& p, const TType& T){
    if (!(getbaseval(T) > 0.0)){
        throw teqp::InvalidArgument("SAFT-VRQ Mie: T must be positive, got T = "
                                    + std::to_string(getbaseval(T)) + " K");
    }
    TemperatureQuantities<TType> q;
    q.D = p.Dparam/T;
    q.Dt = q.D/(p.sigma*p.sigma);
    // in angstrom from here to the conversion back at the end
    const double sA = p.sigma*M_TO_A;
    const double DparamA = p.Dparam*M_TO_A*M_TO_A;
    TType DA = q.D*(M_TO_A*M_TO_A);
    // sigma_eff: u(sigma_eff) = 0, from r = sigma
    TType sigma_effA = newton_root([&](const auto& r, const auto& TT){
        auto [U, U1, U2] = pair_potential(p, r, forceeval(DparamA/TT)); return std::make_tuple(U, U1); }, sA, T, "sigma_eff", p);
    // eps_eff = -u(r_min), du/dr(r_min) = 0, from r = sigma
    TType rminA = newton_root([&](const auto& r, const auto& TT){
        auto [U, U1, U2] = pair_potential(p, r, forceeval(DparamA/TT)); return std::make_tuple(U1, U2); }, sA, T, "r_min", p);
    q.eps_eff = -std::get<0>(pair_potential(p, rminA, DA));
    // r0: u(r0)/T = -ln(eps_mach); the integrand of d is 1 to machine precision on [0, r0]
    const double target = -std::log(std::numeric_limits<double>::epsilon());
    TType r0A = newton_root([&](const auto& r, const auto& TT){
        auto [U, U1, U2] = pair_potential(p, r, forceeval(DparamA/TT));
        return std::make_tuple(forceeval(U/TT - target), forceeval(U1/TT)); }, 0.7*getbaseval(sigma_effA), T, "r0", p);
    // d = r0 + integral_{r0}^{sigma_eff} [1 - exp(-u/T)] dr, GK21, summed in node order
    TType half = (sigma_effA - r0A)/2.0;
    TType dA = r0A;
    for (std::size_t k = 0; k < 21; ++k){
        TType rk = half*GK21_X[k] + half + r0A;
        auto U = std::get<0>(pair_potential(p, rk, DA));
        dA += half*GK21_W[k]*(1.0 - exp(-U/T));
    }
    q.sigma_eff = sigma_effA/M_TO_A;
    q.r0 = r0A/M_TO_A;
    q.d = dA/M_TO_A;
    // dimensionless van der Waals energy, Part I Eqs. 18-21 (thermopack calcAlpha)
    const double la = p.la, lr = p.lr;
    TType sr = sA/sigma_effA;
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


/// Part I Eq. 43: coefficients of eta_eff(lambda) = sum_k c_k(lambda) zeta_x^k
inline constexpr std::array<std::array<double, 4>, 4> CETA{{
    {0.81096, 1.7888, -37.578, 92.284},
    {1.0205, -19.341, 151.26, -463.50},
    {-1.9057, 22.845, -228.14, 973.92},
    {1.0885, -6.1962, 106.98, -677.64}
}};

class SAFTVRQMieMixture{
private:
    const std::vector<std::string> m_names;
    const std::vector<std::vector<PairParameters>> m_pairs;
public:
    SAFTVRQMieMixture(std::vector<std::string> names, std::vector<std::vector<PairParameters>> pairs)
        : m_names(std::move(names)), m_pairs(std::move(pairs)) {}
    
    auto get_names() const { return m_names; }
    const auto& get_pairs() const { return m_pairs; }
    
    template<class VecType>
    auto R([[maybe_unused]] const VecType& molefrac) const { return get_R_gas<decltype(molefrac[0])>(); }
    
    /// The terms a_HS, a1 [K], a2 [K^2], a3 [K^3] and alphar = a_HS + a1/T + a2/T^2 + a3/T^3
    /// (EQUATIONS.md sections 8-13).
    ///
    /// Lengths are in angstrom and the number density in 1/angstrom^3, so every packing fraction and every
    /// length ratio is the same number as in SI. In metres d^3 is ~1e-29 and 1/(6 d3na^2) or 1/d_ij,
    /// evaluated by the recursive-squaring inverse of multicomplex numbers, underflow to NaN for derivatives
    /// in T of order >= 3.
    template<typename TType, typename RhoType, typename VecType>
    auto alphar_terms(const TType& T, const RhoType& rho, const VecType& x) const {
        using result_t = std::common_type_t<TType, RhoType, std::decay_t<decltype(x[0])>>;
        const auto N = m_pairs.size();
        if (static_cast<std::size_t>(x.size()) != N){
            throw teqp::InvalidArgument("SAFT-VRQ Mie: wrong number of mole fractions");
        }
        std::vector<std::vector<TemperatureQuantities<TType>>> tq(N, std::vector<TemperatureQuantities<TType>>(N));
        std::vector<std::vector<TType>> dA(N, std::vector<TType>(N)), sigma_effA(N, std::vector<TType>(N));
        for (auto i = 0U; i < N; ++i){
            for (auto j = i; j < N; ++j){
                tq[i][j] = temperature_quantities(m_pairs[i][j], T);
                tq[j][i] = tq[i][j];
                dA[i][j] = tq[i][j].d*M_TO_A;  dA[j][i] = dA[i][j];
                sigma_effA[i][j] = tq[i][j].sigma_eff*M_TO_A;  sigma_effA[j][i] = sigma_effA[i][j];
            }
        }
        auto d = [&](std::size_t i, std::size_t j) -> const TType& { return dA[i][j]; };
        // m = 1: rho_s = N_A rho, here in 1/angstrom^3
        RhoType rhoN = rho*(constants::N_A/(M_TO_A*M_TO_A*M_TO_A));
        
        // Hard spheres: Santos et al. non-additive mixture with the Carnahan-Starling pure fluid
        // (Part II Eqs. 14-24)
        result_t d3na = 0.0, B2 = 0.0, B3 = 0.0;
        for (auto i = 0U; i < N; ++i){ d3na += x[i]*pow(d(i,i), 3); }
        for (auto i = 0U; i < N; ++i){ for (auto j = 0U; j < N; ++j){ B2 += 4.0*x[i]*x[j]*pow(d(i,j), 3); } }
        auto delta = [&](std::size_t k, std::size_t i, std::size_t j) -> TType {
            TType v = d(i,k) + d(j,k) - d(i,j);
            return (getbaseval(v) > 0) ? v : TType(0.0);
        };
        auto c = [&](std::size_t k, std::size_t i, std::size_t j) -> TType {
            return pow(delta(k,i,j), 3) + 1.5*pow(delta(k,i,j), 2)*delta(i,j,k)*delta(j,i,k)/d(i,j);
        };
        for (auto i = 0U; i < N; ++i){
            for (auto j = 0U; j < N; ++j){
                for (auto k = 0U; k < N; ++k){
                    TType Bijk = 4.0/3.0*(c(k,i,j)*pow(d(i,j), 3) + c(j,i,k)*pow(d(i,k), 3) + c(i,j,k)*pow(d(j,k), 3));
                    B3 += x[i]*x[j]*x[k]*Bijk;
                }
            }
        }
        result_t A1 = (10.0*d3na*B2 - 4.0*B3)/(6.0*d3na*d3na);
        result_t A2 = (B3 - d3na*B2)/(6.0*d3na*d3na);
        result_t eta = constants::PI/6.0*rhoN*d3na;
        result_t a_cs = (4.0*eta - 3.0*eta*eta)/((1.0 - eta)*(1.0 - eta));
        // thermopack uses the same -log(1 - eta) form. Its relative error is ~eps_mach/eta at very low
        // density (2.4e-8 on alphar/rho at rho = 1e-3 mol/m^3 for Ne-H2); the derivatives are not affected.
        result_t ahs = -log(1.0 - eta)*A1 + a_cs*A2;
        
        // Packing fractions of the perturbation terms
        result_t zx = 0.0, zxb = 0.0;
        for (auto i = 0U; i < N; ++i){
            for (auto j = 0U; j < N; ++j){
                zx += constants::PI/6.0*rhoN*x[i]*x[j]*pow(d(i,j), 3);
                zxb += constants::PI/6.0*rhoN*x[i]*x[j]*pow(sigma_effA[i][j], 3);
            }
        }
        result_t KHS = pow(1.0 - zx, 4)/(1.0 + 4.0*zx + 4.0*zx*zx - 4.0*pow(zx, 3) + pow(zx, 4));
        result_t k1 = (2.0 - zx)/pow(1.0 - zx, 3);
        result_t k2 = -9.0*zx*(1.0 + zx)/pow(1.0 - zx, 3);
        
        result_t a1 = 0.0, a2 = 0.0, a3 = 0.0;
        for (auto i = 0U; i < N; ++i){
            for (auto j = 0U; j < N; ++j){
                const auto& p = m_pairs[i][j];
                const auto& q = tq[i][j];
                TType x0 = (p.sigma*M_TO_A)/d(i,j);
                TType y = sigma_effA[i][j]/d(i,j);
                result_t eta_ij = constants::PI/6.0*rhoN*pow(d(i,j), 3);
                // x0^lambda [aS~(lambda) + B~(lambda)], EQUATIONS.md section 10
                auto term = [&](double lam) -> result_t {
                    double c1 = CETA[0][0] + CETA[0][1]/lam + CETA[0][2]/(lam*lam) + CETA[0][3]/(lam*lam*lam);
                    double c2 = CETA[1][0] + CETA[1][1]/lam + CETA[1][2]/(lam*lam) + CETA[1][3]/(lam*lam*lam);
                    double c3 = CETA[2][0] + CETA[2][1]/lam + CETA[2][2]/(lam*lam) + CETA[2][3]/(lam*lam*lam);
                    double c4 = CETA[3][0] + CETA[3][1]/lam + CETA[3][2]/(lam*lam) + CETA[3][3]/(lam*lam*lam);
                    result_t ef = c1*zx + c2*zx*zx + c3*pow(zx, 3) + c4*pow(zx, 4);
                    result_t aS = -12.0*p.eps/(lam - 3.0)*(1.0 - 0.5*ef)/pow(1.0 - ef, 3);
                    result_t B = 0.0;
                    if (getbaseval(y) > 1.0){
                        TType I = -(pow(y, 3.0 - lam) - 1.0)/(lam - 3.0);
                        TType J = -((lam - 3.0)*pow(y, 4.0 - lam) - (lam - 4.0)*pow(y, 3.0 - lam) - 1.0)
                                  /((lam - 3.0)*(lam - 4.0));
                        B = 6.0*p.eps*(k1*I + k2*J);
                    }
                    return pow(x0, lam)*(aS + B);
                };
                const double la = p.la, lr = p.lr;
                result_t a1t = p.C*(term(la) - term(lr));
                if (p.fh >= 1){ a1t += q.Dt*p.C*(p.Q1a*term(la + 2.0) - p.Q1r*term(lr + 2.0)); }
                if (p.fh >= 2){ a1t += q.Dt*q.Dt*p.C*(p.Q2a*term(la + 4.0) - p.Q2r*term(lr + 4.0)); }
                result_t a2t = term(2.0*la) - 2.0*term(la + lr) + term(2.0*lr);
                for (int pw = 1; pw <= 2*p.fh; ++pw){
                    double rf, af, cf;
                    if (pw == 1){ rf = 2*p.Q1r; af = 2*p.Q1a; cf = -(rf + af); }
                    else if (pw == 2){
                        rf = p.Q1r*p.Q1r; af = p.Q1a*p.Q1a; cf = -2*p.Q1r*p.Q1a;
                        if (p.fh == 2){ rf += 2*p.Q2r; af += 2*p.Q2a; cf -= 2*(p.Q2r + p.Q2a); }
                    }
                    else if (pw == 3){ rf = 2*p.Q2r*p.Q1r; af = 2*p.Q2a*p.Q1a; cf = -2*(p.Q2r*p.Q1a + p.Q2a*p.Q1r); }
                    else { rf = p.Q2r*p.Q2r; af = p.Q2a*p.Q2a; cf = -2*p.Q2r*p.Q2a; }
                    a2t += pow(q.Dt, pw)*(af*term(2.0*la + 2.0*pw) + rf*term(2.0*lr + 2.0*pw) + cf*term(la + lr + 2.0*pw));
                }
                result_t chi = q.f[0]*zxb + q.f[1]*pow(zxb, 5) + q.f[2]*pow(zxb, 8);
                a1 += x[i]*x[j]*eta_ij*a1t;
                a2 += x[i]*x[j]*(1.0 + chi)*KHS*0.5*p.eps*p.C*p.C*eta_ij*a2t;
                a3 += x[i]*x[j]*(-pow(q.eps_eff, 3)*q.f[3]*zxb*exp(q.f[4]*zxb + q.f[5]*zxb*zxb));
            }
        }
        result_t alphar = ahs + a1/T + a2/(T*T) + a3/(T*T*T);
        return std::make_tuple(ahs, a1, a2, a3, alphar);
    }
    
    template<typename TType, typename RhoType, typename VecType>
    auto alphar(const TType& T, const RhoType& rho, const VecType& x) const {
        return forceeval(std::get<4>(alphar_terms(T, rho, x)));
    }
};

/// JSON: {"feynman_hibbs_order": 1|2, "coeffs": [{"name", "sigma_m", "epsilon_over_k", "lambda_a",
/// "lambda_r", "mass_kg"}, ...], "kmat" (optional), "lmat" (optional), "Planck_constant / J s" (optional,
/// default CODATA 2018 6.62607015e-34; thermopack 2.2.3 uses 6.626069311e-34, which moves alphar by up to
/// 4e-6)}
inline auto SAFTVRQMiefactory(const nlohmann::json& spec){
    const auto& jfh = spec.at("feynman_hibbs_order");
    if (!jfh.is_number_integer() || (jfh.get<int>() != 1 && jfh.get<int>() != 2)){
        throw teqp::InvalidArgument("SAFT-VRQ Mie: feynman_hibbs_order must be the integer 1 or 2, got " + jfh.dump());
    }
    const int fh = jfh.get<int>();
    const double h = spec.value("Planck_constant / J s", 6.62607015e-34);
    if (!spec.at("coeffs").is_array() || spec.at("coeffs").empty()){
        throw teqp::InvalidArgument("SAFT-VRQ Mie: coeffs must be a non-empty array");
    }
    std::vector<PureParameters> pures;
    std::vector<std::string> names;
    for (const auto& c : spec.at("coeffs")){
        const std::string name = c.at("name");
        if (c.contains("m") && c.at("m").get<double>() != 1.0){
            throw teqp::InvalidArgument("SAFT-VRQ Mie is defined for spherical molecules only (m = 1); " + name
                                        + " has m = " + c.at("m").dump());
        }
        PureParameters pp{name, c.at("sigma_m"), c.at("epsilon_over_k"), c.at("lambda_a"), c.at("lambda_r"),
                          c.at("mass_kg")};
        auto positive = [&](double v, const char* key){
            if (!(v > 0.0)){
                throw teqp::InvalidArgument("SAFT-VRQ Mie: " + std::string(key) + " of " + name
                                            + " must be positive, got " + std::to_string(v));
            }
        };
        positive(pp.sigma_m, "sigma_m");
        positive(pp.epsilon_over_k, "epsilon_over_k");
        positive(pp.mass_kg, "mass_kg");
        if (!(pp.lambda_a > 3.0 && pp.lambda_r > pp.lambda_a)){
            throw teqp::InvalidArgument("SAFT-VRQ Mie: " + name + " needs lambda_r > lambda_a > 3, got lambda_a = "
                                        + std::to_string(pp.lambda_a) + ", lambda_r = " + std::to_string(pp.lambda_r));
        }
        pures.push_back(pp);
        names.push_back(name);
    }
    const auto N = pures.size();
    // pairs[i][j] and pairs[j][i] are built separately, while the temperature quantities are computed for
    // j >= i and copied, so an asymmetric matrix would give an inconsistent model. The diagonal is not used.
    auto matrix = [&](const char* key){
        Eigen::ArrayXXd M = Eigen::ArrayXXd::Zero(N, N);
        if (spec.contains(key) && spec.at(key).size() > 0){
            M = build_square_matrix(spec.at(key));
            if (static_cast<std::size_t>(M.rows()) != N){
                throw teqp::InvalidArgument(std::string("SAFT-VRQ Mie: ") + key + " has the wrong size");
            }
            for (auto i = 0U; i < N; ++i){
                if (M(i,i) != 0.0){
                    throw teqp::InvalidArgument(std::string("SAFT-VRQ Mie: ") + key
                                                + " has a nonzero diagonal entry for " + names[i]);
                }
                for (auto j = i + 1; j < N; ++j){
                    if (M(i,j) != M(j,i)){
                        throw teqp::InvalidArgument(std::string("SAFT-VRQ Mie: ") + key + " is not symmetric for "
                                                    + names[i] + "-" + names[j]);
                    }
                }
            }
        }
        return M;
    };
    Eigen::ArrayXXd kmat = matrix("kmat"), lmat = matrix("lmat");
    std::vector<std::vector<PairParameters>> pairs(N, std::vector<PairParameters>(N));
    for (auto i = 0U; i < N; ++i){
        for (auto j = 0U; j < N; ++j){
            pairs[i][j] = make_pair(pures[i], pures[j], (i == j) ? 0.0 : kmat(i,j), (i == j) ? 0.0 : lmat(i,j), fh, h);
        }
    }
    return SAFTVRQMieMixture(std::move(names), std::move(pairs));
}

} // namespace teqp::SAFTVRQMie

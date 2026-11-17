#pragma once

#include "teqp/types.hpp"
#include "teqp/exceptions.hpp"
#include "teqp/models/multifluid.hpp"
#include "teqp/models/saftvrmie.hpp"
#include "teqp/models/saftvrqmie.hpp"

#include <variant>

namespace teqp::multifluid::multifluid_excess {

/**
 A multifluid model plus the residual excess of a SAFT-VR Mie or SAFT-VRQ Mie model at the same
 temperature and molar density:
 
 \f[
 \alpha^{\rm r}(T,\rho,\vec{x}) = \alpha^{\rm r}_{\rm MF}(T,\rho,\vec{x}) + \Delta(T,\rho,\vec{x}),\qquad
 \Delta = \alpha^{\rm r}_{\rm M}(T,\rho,\vec{x}) - \sum_i x_i\,\alpha^{\rm r}_{\rm M}(T,\rho,\vec{e}_i)
 \f]
 
 with \f$\vec{e}_i\f$ the pure fluid i. \f$\Delta\f$ is zero for every pure fluid, so the pure-fluid
 limits are those of the multifluid model. The multifluid part may carry its own departure function.
 
 Each pure term is evaluated at the mixture's density, so every component's excess-model
 \f$\alpha^{\rm r}\f$ must be finite at that density; where one is not (packing fraction at or above
 1 for the larger molecule of a size-asymmetric pair at very high density), \f$\alpha^{\rm r}\f$ is NaN,
 also at the pure limits. One evaluation costs N+1 excess-model evaluations.
 
 The gas constant is the multifluid's R(x), so the pure-fluid pressures are those of the multifluid
 model. Neither excess model's \f$\alpha^{\rm r}\f$ depends on R, so the sum is consistent.
 
 JSON: {"multifluid": <a "multifluid" model block>, "excess": {"kind": <kind>, "model": <a model block
 of that kind>}}, with kind "SAFT-VR-Mie" or "SAFT-VRQ-Mie" (Feynman-Hibbs corrected SAFT-VR Mie, for
 quantum fluids). The constructor checks that both blocks have the same number of
 components; the caller must give the components in the same order in both blocks (multifluid
 component entries may be file paths or JSON, so names cannot be compared in general).
 
 The SAFT-VR Mie excess model is nonpolar (monomer and chain terms). A SAFT-VR Mie block with
 dipole or quadrupole moments, or with polar options, raises: carrying the polar terms through every
 derivative type the interface instantiates doubles the compile time and memory of this model.
 */
class MultifluidPlusExcess{
public:
    using multifluid_t = decltype(multifluidfactory(nlohmann::json{}));
    using excess_t = std::variant<SAFTVRMie::SAFTVRMieNonpolarMixture, SAFTVRQMie::SAFTVRQMieMixture>;
private:
    const multifluid_t m_multifluid;
    const excess_t m_excess;
    
    /// Raise if a SAFT-VR Mie block asks for polar terms; SAFTVRMieNonpolarfactory would ignore them
    static void require_nonpolar(const nlohmann::json& model){
        for (const auto* key : {"polar_model", "polar_flags"}){
            if (model.contains(key)){
                throw teqp::InvalidArgument(std::string("multifluid-excess: the SAFT-VR Mie excess model is nonpolar; remove \"") + key + "\"");
            }
        }
        if (!model.contains("coeffs")){
            return; // "names" are looked up in SAFTVRMieLibrary, which holds nonpolar fluids only
        }
        for (const auto& c : model.at("coeffs")){
            for (const auto* key : {"(mu^*)^2", "nmu", "mu_Cm", "mu_D", "(Q^*)^2", "nQ", "Q_Cm2", "Q_DA"}){
                if (c.contains(key)){
                    throw teqp::InvalidArgument(std::string("multifluid-excess: the SAFT-VR Mie excess model is nonpolar; component ")
                        + c.value("name", std::string("?")) + " has \"" + key + "\"");
                }
            }
        }
    }
    
    static excess_t build_excess(const nlohmann::json& spec){
        const auto& kind = spec.at("kind");
        if (kind == "SAFT-VR-Mie"){
            require_nonpolar(spec.at("model"));
            return SAFTVRMie::SAFTVRMieNonpolarfactory(spec.at("model"));
        }
        if (kind == "SAFT-VRQ-Mie"){
            return SAFTVRQMie::SAFTVRQMiefactory(spec.at("model"));
        }
        throw teqp::InvalidArgument("multifluid-excess: the excess model must be of kind SAFT-VR-Mie or SAFT-VRQ-Mie, got " + kind.dump());
    }
public:
    MultifluidPlusExcess(const nlohmann::json& spec) :
        m_multifluid(multifluidfactory(spec.at("multifluid"))),
        m_excess(build_excess(spec.at("excess")))
    {
        auto N_MF = static_cast<std::size_t>(m_multifluid.redfunc.Tc.size());
        auto N_M = std::visit([](const auto& m){ return m.get_names().size(); }, m_excess);
        if (N_MF != N_M){
            throw teqp::InvalidArgument("multifluid-excess: the multifluid model has " + std::to_string(N_MF)
                + " components and the excess model " + std::to_string(N_M));
        }
    }
    
    const auto& get_multifluid() const { return m_multifluid; }
    const auto& get_excess() const { return m_excess; }
    
    template<class VecType>
    auto R(const VecType& molefrac) const {
        return m_multifluid.R(molefrac);
    }
    
    /// \f$\Delta(T,\rho,\vec{x})\f$, the residual excess of the excess model at the same (T, rho)
    template <typename TType, typename RhoType, typename MoleFractions>
    auto alphar_excess(const TType& T, const RhoType& rho, const MoleFractions& molefrac) const {
        using result_t = std::common_type_t<TType, RhoType, std::decay_t<decltype(molefrac[0])>>;
        const auto N = static_cast<Eigen::Index>(molefrac.size());
        return std::visit([&](const auto& model) -> result_t {
            result_t val = model.alphar(T, rho, molefrac);
            Eigen::ArrayXd e = Eigen::ArrayXd::Zero(N);
            for (Eigen::Index i = 0; i < N; ++i){
                e.setZero(); e[i] = 1.0;
                val -= molefrac[i]*model.alphar(T, rho, e);
            }
            return forceeval(val);
        }, m_excess);
    }
    
    template <typename TType, typename RhoType, typename MoleFractions>
    auto alphar(const TType& T, const RhoType& rho, const MoleFractions& molefrac) const {
        return forceeval(m_multifluid.alphar(T, rho, molefrac) + alphar_excess(T, rho, molefrac));
    }
};

} // namespace teqp::multifluid::multifluid_excess

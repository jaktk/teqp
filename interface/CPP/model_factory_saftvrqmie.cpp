#include "teqp/cpp/teqpcpp.hpp"
#include "teqp/cpp/deriv_adapter.hpp"

#include "model_flags.hpp" // Contains (optionally) macros to disable various models

#ifndef DISABLE_SAFTVRQMIE
#include "teqp/models/saftvrqmie.hpp"
#endif

namespace teqp{
    namespace cppinterface{
#if defined(DISABLE_SAFTVRQMIE)
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SAFTVRQMie(const nlohmann::json &){
            throw teqp::NotImplementedError("The SAFT-VRQ Mie model has been disabled");
        }
#else
        std::unique_ptr<teqp::cppinterface::AbstractModel> make_SAFTVRQMie(const nlohmann::json &j){
            return teqp::cppinterface::adapter::make_owned(SAFTVRQMie::SAFTVRQMiefactory(j));
        }
#endif
    }
}

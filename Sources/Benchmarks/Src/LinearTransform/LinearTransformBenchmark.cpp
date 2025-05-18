#include "BenchmarkBase.hpp"

#include <array>
#include <vector>
#include <algorithm>

namespace Calibration {

class LinearTransformBenchmark : public BenchmarkBase {

public:
    using BenchmarkBase::BenchmarkBase;

    LinearTransformBenchmark() = default;

    void Run(CppBenchmark::Context &aContext) override
    {
        for (const auto &[pointX, pointY]: points) {
            [[maybe_unused]] auto vectorX = coefs[0] * pointX + coefs[1] * pointY + coefs[4];
            [[maybe_unused]] auto vectorY = coefs[2] * pointX + coefs[3] * pointY + coefs[5];
        }
        aContext.metrics().AddItems(points.size());
    }

protected:
	std::array<float, 6> coefs{-4.00455, -0.02869, 0.01143, -4.01812, -0.00938, -0.12272};
};
} // namespace Calibration

BENCHMARK_CLASS(Calibration::LinearTransformBenchmark, "Linear transformation", Settings().Param(10).Duration(1).Attempts(5))

BENCHMARK_MAIN()
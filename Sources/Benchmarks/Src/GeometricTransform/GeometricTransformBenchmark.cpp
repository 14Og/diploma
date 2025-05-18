#include "BenchmarkBase.hpp"

#include <array>
#include <vector>
#include <algorithm>
#include <cmath>

namespace Calibration {
class GeometricTransformNoOffsetsBenchmark : public BenchmarkBase {

public:
	using BenchmarkBase::BenchmarkBase;

	GeometricTransformNoOffsetsBenchmark() = default;

	void Run(CppBenchmark::Context &aContext) override
	{
		for (const auto &[pointX, pointY] : points) {
			[[maybe_unused]] auto vectorX = std::asin(std::atan(-pointX / d));
			[[maybe_unused]] auto vectorY = std::asin(std::atan(-pointY / d));
		}
		aContext.metrics().AddItems(points.size());
	}

private:
	float d{0.24033};
};

class GeometricTransformLinearOffsetsBenchmark : public BenchmarkBase {

public:
	using BenchmarkBase::BenchmarkBase;

	GeometricTransformLinearOffsetsBenchmark() = default;

	void Run(CppBenchmark::Context &aContext) override
	{
		for (const auto &[pointX, pointY] : points) {
			[[maybe_unused]] auto vectorX = std::asin(std::atan((xc - pointX) / d));
			[[maybe_unused]] auto vectorY = std::asin(std::atan((yc - pointY) / d));
		}
		aContext.metrics().AddItems(points.size());
	}

private:
	float d{0.23756};
	float xc{-0.002};
	float yc{-0.0302};
};

class GeometricTransformAllOffsetsBenchmark : public BenchmarkBase {

    public:
        using BenchmarkBase::BenchmarkBase;
    
        GeometricTransformAllOffsetsBenchmark() = default;
    
        void Run(CppBenchmark::Context &aContext) override
        {
            for (const auto &[pointX, pointY] : points) {
                auto xcRotated = std::cos(phi) * xc - std::sin(phi) * yc;
                auto ycRotated = std::sin(phi) * xc + std::cos(phi) * yc;

                [[maybe_unused]] auto vectorX = std::asin(std::atan(((xcRotated - pointX) / d) * std::cos(beta)) + alpha);
                [[maybe_unused]] auto vectorY = std::asin(std::atan(((ycRotated - pointY) / d) * std::cos(alpha)) + beta);
            }
            aContext.metrics().AddItems(points.size());
        }
    
    private:
        float d{0.23753};
        float xc{0.03063};
        float yc{-0.01097};
        float alpha{0.00285};
        float beta{0.00769};
        float phi{-1.3148};
    };

} // namespace Calibration

BENCHMARK_CLASS(Calibration::GeometricTransformNoOffsetsBenchmark, "Geometric transformation (no offsets)", Settings().Param(10).Duration(1).Attempts(5))

BENCHMARK_CLASS(Calibration::GeometricTransformLinearOffsetsBenchmark, "Geometric transformation (linear offsets)", Settings().Param(10).Duration(1).Attempts(5))

BENCHMARK_CLASS(Calibration::GeometricTransformAllOffsetsBenchmark, "Geometric transformation (all offsets)", Settings().Param(10).Duration(1).Attempts(5))

BENCHMARK_MAIN()
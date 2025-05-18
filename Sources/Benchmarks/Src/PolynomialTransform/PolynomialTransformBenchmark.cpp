#include "BenchmarkBase.hpp"

#include <array>
#include <vector>
#include <algorithm>

namespace Calibration {

class Polynomial2TransformBenchmark : public BenchmarkBase {

public:
	using BenchmarkBase::BenchmarkBase;

	Polynomial2TransformBenchmark() = default;

	void Run(CppBenchmark::Context &aContext) override
	{
		for (const auto &[pointX, pointY] : points) {
			[[maybe_unused]] auto vectorX = pointX * xCoefs[0] + pointY * xCoefs[1] + (pointX * pointX) * xCoefs[2]
				+ (pointX * pointY) * xCoefs[3] + (pointY * pointY) * xCoefs[4];

			[[maybe_unused]] auto vectorY = pointX * yCoefs[0] + pointY * yCoefs[1] + (pointX * pointX) * yCoefs[2]
				+ (pointX * pointY) * yCoefs[3] + (pointY * pointY) * yCoefs[4];
		}
		aContext.metrics().AddItems(points.size());
	}

protected:
	std::array<float, 5> xCoefs{-4.01425, -0.02549, 0.12995, 0.17799, 0.04911};
	std::array<float, 5> yCoefs{0.01573, -4.0119, 0.14508, -0.06436, 0.33313};
};

class Polynomial3TransformBenchmark : public BenchmarkBase {

public:
	using BenchmarkBase::BenchmarkBase;

	Polynomial3TransformBenchmark() = default;

	void Run(CppBenchmark::Context &aContext) override
	{
		for (const auto &[pointX, pointY] : points) {
			[[maybe_unused]] auto vectorX = pointX * xCoefs[0] + pointY * xCoefs[1] + (pointX * pointX) * xCoefs[2]
				+ (pointX * pointY) * xCoefs[3] + (pointY * pointY) * xCoefs[4] + (pointX * pointX * pointX) * xCoefs[5]
				+ (pointX * pointX * pointY) * xCoefs[6] + (pointX * pointY * pointY) * xCoefs[7]
				+ (pointY * pointY * pointY) * xCoefs[8];

			[[maybe_unused]] auto vectorY = pointX * yCoefs[0] + pointY * yCoefs[1] + (pointX * pointX) * yCoefs[2]
				+ (pointX * pointY) * yCoefs[3] + (pointY * pointY) * yCoefs[4] + (pointX * pointX * pointX) * yCoefs[5]
				+ (pointX * pointX * pointY) * yCoefs[6] + (pointX * pointY * pointY) * yCoefs[7]
				+ (pointY * pointY * pointY) * yCoefs[8];
		}
		aContext.metrics().AddItems(points.size());
	}

protected:
	std::array<float, 9> xCoefs{-4.40398, 0.01094, -1.19227, 0.85994, 1.5493, 11.63868, -44.13126, 14.47144, 15.88627};
	std::array<float, 9> yCoefs{0.00601, -4.37863, 1.0866, 0.07037, 0.7452, 0.39946, 30.57752, 0.5357, 4.98722};
};

class Polynomial4TransformBenchmark : public BenchmarkBase {

public:
	using BenchmarkBase::BenchmarkBase;

	Polynomial4TransformBenchmark() = default;

	void Run(CppBenchmark::Context &aContext) override
	{
		for (const auto &[pointX, pointY] : points) {
			[[maybe_unused]] auto vectorX = pointX * xCoefs[0] + pointY * xCoefs[1] + (pointX * pointX) * xCoefs[2]
				+ (pointX * pointY) * xCoefs[3] + (pointY * pointY) * xCoefs[4] + (pointX * pointX * pointX) * xCoefs[5]
				+ (pointX * pointX * pointY) * xCoefs[6] + (pointX * pointY * pointY) * xCoefs[7]
				+ (pointY * pointY * pointY) * xCoefs[8] + (pointX * pointX * pointX * pointX) * xCoefs[9]
				+ (pointX * pointX * pointX * pointY) * xCoefs[10] + (pointX * pointX * pointY * pointY) * xCoefs[11]
				+ (pointX * pointY * pointY * pointY) * xCoefs[12] + (pointY * pointY * pointY * pointY) * xCoefs[13];

			[[maybe_unused]] auto vectorY = pointX * yCoefs[0] + pointY * yCoefs[1] + (pointX * pointX) * yCoefs[2]
				+ (pointX * pointY) * yCoefs[3] + (pointY * pointY) * yCoefs[4] + (pointX * pointX * pointX) * yCoefs[5]
				+ (pointX * pointX * pointY) * yCoefs[6] + (pointX * pointY * pointY) * yCoefs[7]
				+ (pointY * pointY * pointY) * yCoefs[8] + (pointX * pointX * pointX * pointX) * yCoefs[9]
				+ (pointX * pointX * pointX * pointY) * yCoefs[10] + (pointX * pointX * pointY * pointY) * yCoefs[11]
				+ (pointX * pointY * pointY * pointY) * yCoefs[12] + (pointY * pointY * pointY * pointY) * yCoefs[13];
		}
		aContext.metrics().AddItems(points.size());
	}

protected:
	std::array<float, 14> xCoefs{-4.44223, 0.06282, -3.21493, 0.59672, 4.13414, 4.53784, -133.92454, 19.06615, 73.34086,
		10.61099, -260.11342, -1179.80744, 81.72613, 423.34796};
	std::array<float, 14> yCoefs{0.00042, -4.36339, 0.55898, 0.72942, 1.46812, -20.49006, 13.54365, 22.64342, 11.33291,
		7.34193, -695.08698, -94.15836, 241.4226, 21.12251};
};

} // namespace Calibration

BENCHMARK_CLASS(
	Calibration::Polynomial2TransformBenchmark, "Polynomial 2 benchmark", Settings().Param(10).Duration(1).Attempts(5))

BENCHMARK_CLASS(
	Calibration::Polynomial3TransformBenchmark, "Polynomial 3 benchmark", Settings().Param(10).Duration(1).Attempts(5))

BENCHMARK_CLASS(
	Calibration::Polynomial4TransformBenchmark, "Polynomial 4 benchmark", Settings().Param(10).Duration(1).Attempts(5))

BENCHMARK_MAIN()
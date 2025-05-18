#ifndef BENCHMARKBASE_HPP_
#define BENCHMARKBASE_HPP_

#include <SensorPoint.hpp>

#include "benchmark/cppbenchmark.h"

namespace Calibration {
class BenchmarkBase : public CppBenchmark::Benchmark {

public:
	using Benchmark::Benchmark;

	void Initialize(CppBenchmark::Context &aContext) override
	{
		points.resize(aContext.x());
	}

	void Cleanup([[maybe_unused]] CppBenchmark::Context &aContext) override
	{
		points.clear();
	}

protected:
	std::vector<SensorPoint> points;
};
} // namespace Calibration

#endif /* BENCHMARKBASE_HPP_ */

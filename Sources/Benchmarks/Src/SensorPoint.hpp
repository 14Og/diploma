#ifndef SENSORPOINT_HPP_
#define SENSORPOINT_HPP_

#include <random>

namespace Calibration {

class SensorPoint {

    static constexpr float kMin{-0.25};
    static constexpr float kMax{0.25};

public:
    float x;
    float y;
public:

    SensorPoint(): x(dist(engine)), y(dist(engine))
    {

    }

private:
    static inline std::random_device rd;
    static inline std::mt19937 engine{rd()};
    static inline std::uniform_real_distribution<float> dist{kMin, kMax};

};

}
#endif /* SENSORPOINT_HPP_ */

#pragma once

#include <vector>
#include <cmath>

using TVector = std::vector<float>;

inline float Dot(const TVector& a, const TVector& b) {
	float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

inline void Add(TVector& a, const TVector& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] += b[i];
    }
}

inline void Saxpy(TVector& a, float coeff, const TVector& b) {
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] += coeff * b[i];
    }
}

inline void Unit(TVector& a) {
	const float len = sqrt(Dot(a, a));
	if (len == 0.0f) {
        return;
    }

    for (size_t i = 0; i < a.size(); ++i) {
        a[i] /= len;
    }
}


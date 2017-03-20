#include <vector>
#include <functional>
#include <algorithm>
#include <random>
#include <array>
#include <string>
#include <chrono>
#include <iostream>
#include <complex>
#include <cassert>
#include <cstring>
#include <cmath>
#include <immintrin.h>
#include <boost/align/aligned_allocator.hpp>
#include <omp.h>

using fvec = std::vector<float, boost::alignment::aligned_allocator<float, 32>>;

#ifdef __FMA__
static void hconv_split_fma(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < iheight; i++) {
        for (int j = 0; j < length-1; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[std::abs(k)]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[std::abs(k)]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = length-1; j < iwidth-length+1; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                sum = std::fma(kernel[k], input[i*ipitch+j-k] +
                                          input[i*ipitch+j+k], sum);
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = iwidth-length+1; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[std::abs(k)]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[std::abs(k)]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}
#endif

static void hconv_split(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < iheight; i++) {
        for (int j = 0; j < length-1; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[k]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[k]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = length-1; j < iwidth-length+1; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                sum += kernel[k]*(input[i*ipitch+j-k] + input[i*ipitch+j+k]);
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = iwidth-length+1; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[k]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[k]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void hconv_split_sc(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    int iheightsc = iheight & (~3);
    for (int i = 0; i < iheightsc; i+=4) {
        for (int j = 0; j < length-1; j++) {
            float sum0 = kernel[0]*input[(i+0)*ipitch+j];
            float sum1 = kernel[0]*input[(i+1)*ipitch+j];
            float sum2 = kernel[0]*input[(i+2)*ipitch+j];
            float sum3 = kernel[0]*input[(i+3)*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0) {
                    sum0 += kernel[k]*input[(i+0)*ipitch+j-k];
                    sum1 += kernel[k]*input[(i+1)*ipitch+j-k];
                    sum2 += kernel[k]*input[(i+2)*ipitch+j-k];
                    sum3 += kernel[k]*input[(i+3)*ipitch+j-k];
                }
                if (j+k < iwidth) {
                    sum0 += kernel[k]*input[(i+0)*ipitch+j+k];
                    sum1 += kernel[k]*input[(i+1)*ipitch+j+k];
                    sum2 += kernel[k]*input[(i+2)*ipitch+j+k];
                    sum3 += kernel[k]*input[(i+3)*ipitch+j+k];
                }
            }
            (*output)[(i+0)*ipitch+j] = sum0;
            (*output)[(i+1)*ipitch+j] = sum1;
            (*output)[(i+2)*ipitch+j] = sum2;
            (*output)[(i+3)*ipitch+j] = sum3;
        }
        for (int j = length-1; j < iwidth-length+1; j++) {
            float sum0 = kernel[0]*input[(i+0)*ipitch+j];
            float sum1 = kernel[0]*input[(i+1)*ipitch+j];
            float sum2 = kernel[0]*input[(i+2)*ipitch+j];
            float sum3 = kernel[0]*input[(i+3)*ipitch+j];
            for (int k = 1; k < length; k++) {
                sum0 += kernel[k]*(input[(i+0)*ipitch+j-k] +
                    input[(i+0)*ipitch+j+k]);
                sum1 += kernel[k]*(input[(i+1)*ipitch+j-k] +
                    input[(i+1)*ipitch+j+k]);
                sum2 += kernel[k]*(input[(i+2)*ipitch+j-k] +
                    input[(i+2)*ipitch+j+k]);
                sum3 += kernel[k]*(input[(i+3)*ipitch+j-k] +
                    input[(i+3)*ipitch+j+k]);
            }
            (*output)[(i+0)*ipitch+j] = sum0;
            (*output)[(i+1)*ipitch+j] = sum1;
            (*output)[(i+2)*ipitch+j] = sum2;
            (*output)[(i+3)*ipitch+j] = sum3;
        }
        for (int j = iwidth-length+1; j < iwidth; j++) {
            float sum0 = kernel[0]*input[(i+0)*ipitch+j];
            float sum1 = kernel[0]*input[(i+1)*ipitch+j];
            float sum2 = kernel[0]*input[(i+2)*ipitch+j];
            float sum3 = kernel[0]*input[(i+3)*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0) {
                    sum0 += kernel[k]*input[(i+0)*ipitch+j-k];
                    sum1 += kernel[k]*input[(i+1)*ipitch+j-k];
                    sum2 += kernel[k]*input[(i+2)*ipitch+j-k];
                    sum3 += kernel[k]*input[(i+3)*ipitch+j-k];
                }
                if (j+k < iwidth) {
                    sum0 += kernel[k]*input[(i+0)*ipitch+j+k];
                    sum1 += kernel[k]*input[(i+1)*ipitch+j+k];
                    sum2 += kernel[k]*input[(i+2)*ipitch+j+k];
                    sum3 += kernel[k]*input[(i+3)*ipitch+j+k];
                }
            }
            (*output)[(i+0)*ipitch+j] = sum0;
            (*output)[(i+1)*ipitch+j] = sum1;
            (*output)[(i+2)*ipitch+j] = sum2;
            (*output)[(i+3)*ipitch+j] = sum3;
        }
    }
    for (int i = iheightsc; i < iheight; i++) {
        for (int j = 0; j < length-1; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[k]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[k]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = length-1; j < iwidth-length+1; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                sum += kernel[k]*(input[i*ipitch+j-k] + input[i*ipitch+j+k]);
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = iwidth-length+1; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[k]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[k]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void hconv_split_sc_avx(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    int iheightsc = iheight & (~0x1);
    for (int i = 0; i < iheightsc; i+=2) {
        for (int j = 0; j < length-1; j++) {
            float sum0 = kernel[0]*input[(i+0)*ipitch+j];
            float sum1 = kernel[0]*input[(i+1)*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0) {
                    sum0 += kernel[k]*input[(i+0)*ipitch+j-k];
                    sum1 += kernel[k]*input[(i+1)*ipitch+j-k];
                }
                if (j+k < iwidth) {
                    sum0 += kernel[k]*input[(i+0)*ipitch+j+k];
                    sum1 += kernel[k]*input[(i+1)*ipitch+j+k];
                }
            }
            (*output)[(i+0)*ipitch+j] = sum0;
            (*output)[(i+1)*ipitch+j] = sum1;
        }
        for (int j = length-1; j < iwidth-length+1; j++) {
            float sum0 = kernel[0]*input[(i+0)*ipitch+j];
            float sum1 = kernel[0]*input[(i+1)*ipitch+j];
            for (int k = 1; k < length; k++) {
                sum0 += kernel[k]*(input[(i+0)*ipitch+j-k] +
                    input[(i+0)*ipitch+j+k]);
                sum1 += kernel[k]*(input[(i+1)*ipitch+j-k] +
                    input[(i+1)*ipitch+j+k]);
            }
            (*output)[(i+0)*ipitch+j] = sum0;
            (*output)[(i+1)*ipitch+j] = sum1;
        }
        int iwidthavx = length-1 + ((iwidth-2*length+2) & (~7));
        for (int j = length-1; j < iwidthavx; j+=8) {
            auto sum0 = _mm256_mul_ps(
                _mm256_set1_ps(kernel[0]),
                _mm256_loadu_ps(&input[(i+0)*ipitch+j]));
            auto sum1 = _mm256_mul_ps(
                _mm256_set1_ps(kernel[0]),
                _mm256_loadu_ps(&input[(i+1)*ipitch+j]));
            for (int k = 1; k < length; k++) {
                sum0 = _mm256_add_ps(
                    sum0,
                    _mm256_mul_ps(
                        _mm256_set1_ps(kernel[k]),
                        _mm256_add_ps(
                            _mm256_loadu_ps(&input[(i+0)*ipitch+j-k]),
                            _mm256_loadu_ps(&input[(i+0)*ipitch+j+k]))));
                sum1 = _mm256_add_ps(
                    sum1,
                    _mm256_mul_ps(
                        _mm256_set1_ps(kernel[k]),
                        _mm256_add_ps(
                            _mm256_loadu_ps(&input[(i+1)*ipitch+j-k]),
                            _mm256_loadu_ps(&input[(i+1)*ipitch+j+k]))));
            }
            _mm256_storeu_ps(&(*output)[(i+0)*ipitch+j], sum0);
            _mm256_storeu_ps(&(*output)[(i+1)*ipitch+j], sum1);
        }
        for (int j = iwidthavx; j < iwidth; j++) {
            float sum0 = kernel[0]*input[(i+0)*ipitch+j];
            float sum1 = kernel[0]*input[(i+1)*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0) {
                    sum0 += kernel[k]*input[(i+0)*ipitch+j-k];
                    sum1 += kernel[k]*input[(i+1)*ipitch+j-k];
                }
                if (j+k < iwidth) {
                    sum0 += kernel[k]*input[(i+0)*ipitch+j+k];
                    sum1 += kernel[k]*input[(i+1)*ipitch+j+k];
                }
            }
            (*output)[(i+0)*ipitch+j] = sum0;
            (*output)[(i+1)*ipitch+j] = sum1;
        }
    }
    for (int i = iheightsc; i < iheight; i++) {
        for (int j = 0; j < length-1; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[k]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[k]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = length-1; j < iwidth-length+1; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                sum += kernel[k]*(input[i*ipitch+j-k] + input[i*ipitch+j+k]);
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = iwidth-length+1; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[k]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[k]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

#ifdef __FMA__
static void hconv_split_avx_fma(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < iheight; i++) {
        for (int j = 0; j < length-1; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[k]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[k]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
        int iwidthavx = length-1 + ((iwidth-2*length+2) & (~7));
        for (int j = length-1; j < iwidthavx; j+=8) {
            auto sum = _mm256_mul_ps(
                _mm256_set1_ps(kernel[0]),
                _mm256_loadu_ps(&input[i*ipitch+j]));
            for (int k = 1; k < length; k++) {
                sum = _mm256_fmadd_ps(
                    _mm256_set1_ps(kernel[k]),
                    _mm256_add_ps(
                        _mm256_loadu_ps(&input[i*ipitch+j-k]),
                        _mm256_loadu_ps(&input[i*ipitch+j+k])),
                    sum);
            }
            _mm256_storeu_ps(&(*output)[i*ipitch+j], sum);
        }
        for (int j = iwidthavx; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[k]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[k]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}
#endif

static void hconv_split_avx(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < iheight; i++) {
        for (int j = 0; j < length-1; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[k]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[k]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
        int iwidthavx = length-1 + ((iwidth-2*length+2) & (~7));
        for (int j = length-1; j < iwidthavx; j+=8) {
            auto sum = _mm256_mul_ps(
                _mm256_set1_ps(kernel[0]),
                _mm256_loadu_ps(&input[i*ipitch+j]));
            for (int k = 1; k < length; k++) {
                sum = _mm256_add_ps(
                    sum,
                    _mm256_mul_ps(
                        _mm256_set1_ps(kernel[k]),
                        _mm256_add_ps(
                            _mm256_loadu_ps(&input[i*ipitch+j-k]),
                            _mm256_loadu_ps(&input[i*ipitch+j+k]))));
            }
            _mm256_storeu_ps(&(*output)[i*ipitch+j], sum);
        }
        for (int j = iwidthavx; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum += kernel[k]*input[i*ipitch+j-k];
                if (j+k < iwidth)
                    sum += kernel[k]*input[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void hconv_groundtruth(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < iheight; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = 0.f;
            for (int k = -length+1; k < length; k++) {
                if (j-k >= 0 && j-k < iwidth)
                    sum += kernel[std::abs(k)]*input[i*ipitch+j-k];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void vconv_groundtruth(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int j = 0; j < iwidth; j++) {
        for (int i = 0; i < iheight; i++) {
            float sum = 0.f;
            for (int k = -length+1; k < length; k++) {
                if (i-k >= 0 && i-k < iheight)
                    sum += kernel[std::abs(k)]*input[(i-k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void vconv_split(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int j = 0; j < iwidth; j++) {
        for (int i = 0; i < length-1; i++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum += kernel[k]*input[(i-k)*ipitch+j];
                if (i+k < iheight)
                    sum += kernel[k]*input[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int i = length-1; i < iheight-length+1; i++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                sum += kernel[k]*(input[(i-k)*ipitch+j] +
                    input[(i+k)*ipitch+j]);
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int i = iheight-length+1; i < iheight; i++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum += kernel[k]*input[(i-k)*ipitch+j];
                if (i+k < iheight)
                    sum += kernel[k]*input[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void vconv_rowmajor(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < iheight; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = 0.f;
            for (int k = -length+1; k < length; k++) {
                if (i-k >= 0 && i-k < iheight)
                    sum += kernel[std::abs(k)]*input[(i-k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void vconv_rowmajor_split(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < length-1; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum += kernel[k]*input[(i-k)*ipitch+j];
                if (i+k < iheight)
                    sum += kernel[k]*input[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = length-1; i < iheight-length+1; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                sum += kernel[k]*(input[(i-k)*ipitch+j] +
                    input[(i+k)*ipitch+j]);
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = iheight-length+1; i < iheight; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum += kernel[k]*input[(i-k)*ipitch+j];
                if (i+k < iheight)
                    sum += kernel[k]*input[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void vconv_rowmajor_split_avx(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < length-1; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum += kernel[k]*input[(i-k)*ipitch+j];
                if (i+k < iheight)
                    sum += kernel[k]*input[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    int iwidthavx = iwidth & (~7);
    for (int i = length-1; i < iheight-length+1; i++) {
        for (int j = 0; j < iwidthavx; j+=8) {
            auto sum = _mm256_mul_ps(_mm256_set1_ps(kernel[0]),
                           _mm256_loadu_ps(&input[i*ipitch+j]));
            for (int k = 1; k < length; k++) {
                sum = _mm256_add_ps(
                        sum,
                        _mm256_mul_ps(
                            _mm256_set1_ps(kernel[k]),
                            _mm256_add_ps(
                                _mm256_loadu_ps(&input[(i-k)*ipitch+j]),
                                _mm256_loadu_ps(&input[(i+k)*ipitch+j]))));
            }
            _mm256_storeu_ps(&(*output)[i*ipitch+j], sum);
        }
        for (int j = iwidthavx; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                sum += kernel[k]*(input[(i-k)*ipitch+j] +
                    input[(i+k)*ipitch+j]);
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = iheight-length+1; i < iheight; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum += kernel[k]*input[(i-k)*ipitch+j];
                if (i+k < iheight)
                    sum += kernel[k]*input[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

#ifdef __FMA__
static void vconv_rowmajor_split_avx_fma(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    assert(ipitch % 8 == 0);
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < length-1; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum += kernel[k]*input[(i-k)*ipitch+j];
                if (i+k < iheight)
                    sum += kernel[k]*input[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    int iwidthavx = iwidth & (~7);
    for (int i = length-1; i < iheight-length+1; i++) {
        for (int j = 0; j < iwidthavx; j+=8) {
            auto sum = _mm256_mul_ps(_mm256_set1_ps(kernel[0]),
                           _mm256_loadu_ps(&input[i*ipitch+j]));
            for (int k = 1; k < length; k++) {
                sum = _mm256_fmadd_ps(
                    _mm256_set1_ps(kernel[k]),
                    _mm256_add_ps(
                        _mm256_loadu_ps(&input[(i-k)*ipitch+j]),
                        _mm256_loadu_ps(&input[(i+k)*ipitch+j])),
                    sum);
            }
            _mm256_storeu_ps(&(*output)[i*ipitch+j], sum);
        }
        for (int j = iwidthavx; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                sum = std::fma(
                    kernel[k],
                    input[(i-k)*ipitch+j] + input[(i+k)*ipitch+j],
                    sum);
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = iheight-length+1; i < iheight; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum += kernel[k]*input[(i-k)*ipitch+j];
                if (i+k < iheight)
                    sum += kernel[k]*input[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}
#endif

#ifdef __FMA__
static void vconv_rowmajor_split_fma(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < length-1; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum += kernel[k]*input[(i-k)*ipitch+j];
                if (i+k < iheight)
                    sum += kernel[k]*input[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = length-1; i < iheight-length+1; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                sum = std::fma(kernel[k],input[(i-k)*ipitch+j] +
                    input[(i+k)*ipitch+j], sum);
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = iheight-length+1; i < iheight; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum += kernel[k]*input[(i-k)*ipitch+j];
                if (i+k < iheight)
                    sum += kernel[k]*input[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}
#endif

static void hrec_groundtruth(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < iheight; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum -= kernel[k]*(*output)[i*ipitch+j-k];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = iwidth-1; j >= 0; j--) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j+k < iwidth)
                    sum -= kernel[k]*(*output)[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void hrec_split(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < iheight; i++) {
        for (int j = 0; j < length; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum -= kernel[k]*(*output)[i*ipitch+j-k];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = length; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++)
                sum -= kernel[k]*(*output)[i*ipitch+j-k];
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = iwidth-1; j >= iwidth-length; j--) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j+k < iwidth)
                    sum -= kernel[k]*(*output)[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = iwidth-length-1; j >= 0; j--) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++)
                sum -= kernel[k]*(*output)[i*ipitch+j+k];
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void hrec_sc4(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    int iheightsc = iheight & (~3);
    for (int i = 0; i < iheightsc; i+=4) {
        for (int j = 0; j < iwidth; j++) {
            float sum0 = kernel[0]*input[(i+0)*ipitch+j];
            float sum1 = kernel[0]*input[(i+1)*ipitch+j];
            float sum2 = kernel[0]*input[(i+2)*ipitch+j];
            float sum3 = kernel[0]*input[(i+3)*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0) {
                    sum0 -= kernel[k]*(*output)[(i+0)*ipitch+j-k];
                    sum1 -= kernel[k]*(*output)[(i+1)*ipitch+j-k];
                    sum2 -= kernel[k]*(*output)[(i+2)*ipitch+j-k];
                    sum3 -= kernel[k]*(*output)[(i+3)*ipitch+j-k];
                }
            }
            (*output)[(i+0)*ipitch+j] = sum0;
            (*output)[(i+1)*ipitch+j] = sum1;
            (*output)[(i+2)*ipitch+j] = sum2;
            (*output)[(i+3)*ipitch+j] = sum3;
        }
        for (int j = iwidth-1; j >= 0; j--) {
            float sum0 = kernel[0]*input[(i+0)*ipitch+j];
            float sum1 = kernel[0]*input[(i+1)*ipitch+j];
            float sum2 = kernel[0]*input[(i+2)*ipitch+j];
            float sum3 = kernel[0]*input[(i+3)*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j+k < iwidth) {
                    sum0 -= kernel[k]*(*output)[(i+0)*ipitch+j+k];
                    sum1 -= kernel[k]*(*output)[(i+1)*ipitch+j+k];
                    sum2 -= kernel[k]*(*output)[(i+2)*ipitch+j+k];
                    sum3 -= kernel[k]*(*output)[(i+3)*ipitch+j+k];
                }
            }
            (*output)[(i+0)*ipitch+j] = sum0;
            (*output)[(i+1)*ipitch+j] = sum1;
            (*output)[(i+2)*ipitch+j] = sum2;
            (*output)[(i+3)*ipitch+j] = sum3;
        }
    }
    for (int i = iheightsc; i < iheight; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum -= kernel[k]*(*output)[i*ipitch+j-k];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = iwidth-1; j >= 0; j--) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j+k < iwidth)
                    sum -= kernel[k]*(*output)[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void hrec_sc2(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    int iheightsc = iheight & (~1);
    for (int i = 0; i < iheightsc; i+=2) {
        for (int j = 0; j < iwidth; j++) {
            float sum0 = kernel[0]*input[(i+0)*ipitch+j];
            float sum1 = kernel[0]*input[(i+1)*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0) {
                    sum0 -= kernel[k]*(*output)[(i+0)*ipitch+j-k];
                    sum1 -= kernel[k]*(*output)[(i+1)*ipitch+j-k];
                }
            }
            (*output)[(i+0)*ipitch+j] = sum0;
            (*output)[(i+1)*ipitch+j] = sum1;
        }
        for (int j = iwidth-1; j >= 0; j--) {
            float sum0 = kernel[0]*input[(i+0)*ipitch+j];
            float sum1 = kernel[0]*input[(i+1)*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j+k < iwidth) {
                    sum0 -= kernel[k]*(*output)[(i+0)*ipitch+j+k];
                    sum1 -= kernel[k]*(*output)[(i+1)*ipitch+j+k];
                }
            }
            (*output)[(i+0)*ipitch+j] = sum0;
            (*output)[(i+1)*ipitch+j] = sum1;
        }
    }
    for (int i = iheightsc; i < iheight; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j-k >= 0)
                    sum -= kernel[k]*(*output)[i*ipitch+j-k];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int j = iwidth-1; j >= 0; j--) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (j+k < iwidth)
                    sum -= kernel[k]*(*output)[i*ipitch+j+k];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void vrec_split(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int j = 0; j < iwidth; j++) {
        for (int i = 0; i < length; i++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum -= kernel[k]*(*output)[(i-k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int i = length; i < iheight; i++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++)
                sum -= kernel[k]*(*output)[(i-k)*ipitch+j];
            (*output)[i*ipitch+j] = sum;
        }
        for (int i = iheight-1; i >= iheight-length; i--) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i+k < iheight)
                    sum -= kernel[k]*(*output)[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int i = iheight-length-1; i >= 0; i--) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i+k < iheight)
                    sum -= kernel[k]*(*output)[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void vrec_rowmajor_split_sc(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    int iwidthsc = iwidth & (~3);
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum -= kernel[k]*(*output)[(i-k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = length-1; i < iheight; i++) {
        for (int j = 0; j < iwidthsc; j+=4) {
            float sum0 = kernel[0]*input[i*ipitch+j+0];
            float sum1 = kernel[0]*input[i*ipitch+j+1];
            float sum2 = kernel[0]*input[i*ipitch+j+2];
            float sum3 = kernel[0]*input[i*ipitch+j+3];
            for (int k = 1; k < length; k++) {
                sum0 -= kernel[k]*(*output)[(i-k)*ipitch+j+0];
                sum1 -= kernel[k]*(*output)[(i-k)*ipitch+j+1];
                sum2 -= kernel[k]*(*output)[(i-k)*ipitch+j+2];
                sum3 -= kernel[k]*(*output)[(i-k)*ipitch+j+3];
            }
            (*output)[i*ipitch+j+0] = sum0;
            (*output)[i*ipitch+j+1] = sum1;
            (*output)[i*ipitch+j+2] = sum2;
            (*output)[i*ipitch+j+3] = sum3;
        }
        for (int j = iwidthsc; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++)
                sum -= kernel[k]*(*output)[(i-k)*ipitch+j];
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = iheight-1; i >= iheight-length; i--) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i+k < iheight)
                    sum -= kernel[k]*(*output)[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = iheight-length-1; i >= 0; i--) {
        for (int j = 0; j < iwidthsc; j+=4) {
            float sum0 = kernel[0]*input[i*ipitch+j+0];
            float sum1 = kernel[0]*input[i*ipitch+j+1];
            float sum2 = kernel[0]*input[i*ipitch+j+2];
            float sum3 = kernel[0]*input[i*ipitch+j+3];
            for (int k = 1; k < length; k++) {
                sum0 -= kernel[k]*(*output)[(i+k)*ipitch+j+0];
                sum1 -= kernel[k]*(*output)[(i+k)*ipitch+j+1];
                sum2 -= kernel[k]*(*output)[(i+k)*ipitch+j+2];
                sum3 -= kernel[k]*(*output)[(i+k)*ipitch+j+3];
            }
            (*output)[i*ipitch+j+0] = sum0;
            (*output)[i*ipitch+j+1] = sum1;
            (*output)[i*ipitch+j+2] = sum2;
            (*output)[i*ipitch+j+3] = sum3;
        }
        for (int j = iwidthsc; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++)
                sum -= kernel[k]*(*output)[(i+k)*ipitch+j];
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void vrec_rowmajor_split_avx(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum -= kernel[k]*(*output)[(i-k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    int iwidthavx = length-1 + ((iwidth-2*length+2) & (~7));
    for (int i = length-1; i < iheight; i++) {
        for (int j = length-1; j < iwidthavx; j+=8) {
            auto sum = _mm256_mul_ps(_mm256_set1_ps(kernel[0]),
                _mm256_loadu_ps(&input[i*ipitch+j]));
            for (int k = 1; k < length; k++) {
                sum = _mm256_sub_ps(
                    sum,
                    _mm256_mul_ps(
                        _mm256_set1_ps(kernel[k]),
                        _mm256_loadu_ps(&(*output)[(i-k)*ipitch+j])));
            }
            _mm256_storeu_ps(&(*output)[i*ipitch+j], sum);
        }
        for (int j = iwidthavx; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++)
                sum -= kernel[k]*(*output)[(i-k)*ipitch+j];
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = iheight-1; i >= iheight-length; i--) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i+k < iheight)
                    sum -= kernel[k]*(*output)[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = iheight-length-1; i >= 0; i--) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++)
                sum -= kernel[k]*(*output)[(i+k)*ipitch+j];
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void vrec_rowmajor_split(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum -= kernel[k]*(*output)[(i-k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = length-1; i < iheight; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++)
                sum -= kernel[k]*(*output)[(i-k)*ipitch+j];
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = iheight-1; i >= iheight-length; i--) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i+k < iheight)
                    sum -= kernel[k]*(*output)[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = iheight-length-1; i >= 0; i--) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++)
                sum -= kernel[k]*(*output)[(i+k)*ipitch+j];
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void vrec_rowmajor(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int i = 0; i < iheight; i++) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum -= kernel[k]*(*output)[(i-k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
    for (int i = iheight-1; i >= 0; i--) {
        for (int j = 0; j < iwidth; j++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i+k < iheight)
                    sum -= kernel[k]*(*output)[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

static void vrec_groundtruth(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch) {
    output->resize(iheight*ipitch);
    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
    for (int j = 0; j < iwidth; j++) {
        for (int i = 0; i < iheight; i++) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i-k >= 0)
                    sum -= kernel[k]*(*output)[(i-k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
        for (int i = iheight-1; i >= 0; i--) {
            float sum = kernel[0]*input[i*ipitch+j];
            for (int k = 1; k < length; k++) {
                if (i+k < iheight)
                    sum -= kernel[k]*(*output)[(i+k)*ipitch+j];
            }
            (*output)[i*ipitch+j] = sum;
        }
    }
}

#define match(s, o, p) \
    ((strncmp(s, o, sizeof(o)-1) == 0) && (p = s+sizeof(o)-1))

typedef void (*p_func)(
    const fvec &input,
    int iwidth,
    int iheight,
    int ipitch,
    const fvec &kernel,
    int length,
    fvec *output,
    int *owidth,
    int *oheight,
    int *opitch
);

typedef struct {
    const char *name;
    p_func func;
} t_variant;

typedef struct {
    const char *name;
    t_variant *variants;
} t_test;

static p_func findfunc(t_test *tests, const char *name, const char *variant) {
    if (!name || !variant) return nullptr;
    for (int i = 0; tests[i].name; i++) {
        if (strcmp(name, tests[i].name) == 0) {
            for (int j = 0; tests[i].variants[j].name; j++) {
                if (strcmp(variant, tests[i].variants[j].name) == 0) {
                    return tests[i].variants[j].func;
                }
            }
        }
    }
    return nullptr;
}

template <typename F>
constexpr F maxrel(void) {
    return static_cast<F>(8) * std::numeric_limits<F>::epsilon();
}

template <typename F>
constexpr F maxabs(void) {
    return static_cast<F>(8) * std::numeric_limits<F>::epsilon();
}

template <typename F>
bool is_almost_equal(F a, F b, F mr = maxrel<F>(), F ma = maxabs<F>()) {
    F diff = std::abs(a-b);
    if (diff < ma) return true;
    a = std::abs(a);
    b = std::abs(b);
    F largest = (b > a) ? b : a;
    return diff <= largest * mr;
}

static bool is_almost_equal(const fvec &a, int aw, int ah, int ap,
    const fvec &b, int bw, int bh, int bp) {
    if (aw != bw || ah != bh) return false;
    for (int i = 0; i < ah; i++) {
        for (int j = 0; j < aw; j++) {
            if (!is_almost_equal(a[i*ap+j], b[i*bp+j])) {
                std::cerr << "error in (" << j << ',' << i << ")\n";
                std::cerr << "  " << a[i*ap+j] << " vs " << b[i*bp+j] << '\n';
                return false;
            }
        }
    }
    return true;
}

t_variant hconv[] = {
    { "groundtruth", hconv_groundtruth },
    { "split", hconv_split },
    { "split_sc", hconv_split_sc },
    { "split_sc_avx", hconv_split_sc_avx },
    { "split_avx", hconv_split_avx },
#ifdef __FMA__
    { "split_fma", hconv_split_fma },
    { "split_avx_fma", hconv_split_avx_fma },
#endif
    { nullptr, nullptr }
};

t_variant vconv[] = {
    { "groundtruth", vconv_groundtruth },
    { "rowmajor", vconv_rowmajor },
    { "split", vconv_split },
    { "rowmajor_split", vconv_rowmajor_split },
    { "rowmajor_split_avx", vconv_rowmajor_split_avx },
#ifdef __FMA__
    { "rowmajor_split_fma", vconv_rowmajor_split_fma },
    { "rowmajor_split_avx_fma", vconv_rowmajor_split_avx_fma },
#endif
    { nullptr, nullptr }
};

t_variant hrec[] = {
    { "groundtruth", hrec_groundtruth },
    { "split", hrec_split },
    { "sc2", hrec_sc2 },
    { "sc4", hrec_sc4 },
    { nullptr, nullptr }
};

t_variant vrec[] = {
    { "groundtruth", vrec_groundtruth },
    { "split", vrec_split },
    { "rowmajor", vrec_rowmajor },
    { "rowmajor_split", vrec_rowmajor_split },
    { "rowmajor_split_sc", vrec_rowmajor_split_sc },
    { "rowmajor_split_avx", vrec_rowmajor_split_avx },
    { nullptr, nullptr }
};

t_test tests[] = {
    { "hconv", hconv },
    { "vconv", vconv },
    { "hrec", hrec },
    { "vrec", vrec },
    { nullptr, nullptr }
};

int main(int argc, char *argv[]) {
    unsigned int iwidth = 1024, iheight = 1024, ipitch = iwidth, rounds = 10;
    unsigned int length = 3, seed = 12345;
    const char *test = "hconv";
    const char *variant = "groundtruth";
    for (int i = 1; i < argc; i++) {
        const char *ptr = nullptr;
        int end = 0;
        if (match(argv[i], "-width:", ptr)) {
            if (sscanf(ptr, "%u%n", &iwidth, &end) == 1 && !ptr[end])
                continue;
        } else if (match(argv[i], "-height:", ptr)) {
            if (sscanf(ptr, "%u%n", &iheight, &end) == 1 && !ptr[end]) {
                if (ipitch < iwidth) ipitch = iwidth;
                continue;
            }
        } else if (match(argv[i], "-length:", ptr)) {
            if (sscanf(ptr, "%u%n", &length, &end) == 1 && !ptr[end]) {
                continue;
            }
        } else if (match(argv[i], "-pitch:", ptr)) {
            if (sscanf(ptr, "%u%n", &ipitch, &end) == 1 && !ptr[end]) {
                if (ipitch < iwidth) ipitch = iwidth;
                continue;
            }
        } else if (match(argv[i], "-seed:", ptr)) {
            if (sscanf(ptr, "%u%n", &seed, &end) == 1 && !ptr[end]) {
                continue;
            }
        } else if (match(argv[i], "-rounds:", ptr)) {
            if (sscanf(ptr, "%u%n", &rounds, &end) == 1 && !ptr[end]) {
                if (ipitch < iwidth) ipitch = iwidth;
                continue;
            }
        } else if (match(argv[i], "-test:list", ptr)) {
            for (int i = 0; tests[i].name; i++) {
                std::cout << tests[i].name << '\n';
            }
            exit(1);
        } else if (match(argv[i], "-test:", ptr)) {
            test = ptr;
        } else if (match(argv[i], "-variant:list", ptr)) {
            for (int i = 0; tests[i].name; i++) {
                if (strcmp(test, tests[i].name) == 0) {
                    for (int j = 0; tests[i].variants[j].name; j++) {
                        std::cout << tests[i].variants[j].name << '\n';
                    }
                    exit(1);
                }
            }
        } else if (match(argv[i], "-variant:", ptr)) {
            variant = ptr;
        } else {
            std::cerr << "invalid argument " << argv[i] << '\n';
            exit(1);
        }
    }

    if (iwidth <= 0 || iheight <= 0 || ipitch < iwidth) {
        std::cerr << "invalid width, height, pitch combination\n";
        exit(1);
    }

    std::cerr << "using input dimensions " << iwidth << 'x'
              << iheight << ':' << ipitch << '\n';

    p_func variant_func = findfunc(tests, test, variant);
    p_func groundtruth_func = findfunc(tests, test, "groundtruth");

    if (!variant_func || !groundtruth_func) {
        std::cerr << "invalid test '" << test <<
                     "' or variant '" << variant << "'\n";
        exit(1);
    }
    std::mt19937 engine(seed);
    std::uniform_real_distribution<> uniform;
    auto garbage = std::bind(uniform, engine);
    fvec kernel(length), input(iheight*ipitch), output(iheight*ipitch),
        truth(iheight*ipitch);
    std::generate(kernel.begin(), kernel.end(), garbage);
    std::generate(input.begin(), input.end(), garbage);
    std::generate(output.begin(), output.end(), garbage);
    std::generate(truth.begin(), truth.end(), garbage);

    int twidth, theight, tpitch;
    std::cerr << "running " << test << " groundtruth\n";
    groundtruth_func(input, iwidth, iheight, ipitch, kernel, length,
        &truth, &twidth, &theight, &tpitch);
    int owidth, oheight, opitch;
    std::cerr << "running " << test << ' ' << variant << '\n';
    variant_func(input, iwidth, iheight, ipitch, kernel, length,
        &output, &owidth, &oheight, &opitch);
    std::cerr << "comparing outputs: ";
    if (!is_almost_equal(output, owidth, oheight, opitch,
                 truth, twidth, theight, tpitch)) {
        std::cerr << "error!\n";
        exit(1);
    } else  {
        std::cerr << "ok\n";
    }
    std::cerr << "running " << rounds << " rounds\n";
    auto begin = std::chrono::high_resolution_clock::now();
//    #pragma omp parallel for
    for (unsigned i = 0; i < rounds; i++) {
        variant_func(input, iwidth, iheight, ipitch, kernel, length,
            &output, &owidth, &oheight, &opitch);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
    std::cout << test << '_' << variant << '\t' << iwidth << '\t' << iheight << '\t' << ipitch << '\t' << duration/rounds << "\n";
    return 0;
}

#include <vector>
#include <chrono>
#include <iostream>
#include <complex>
#include <cmath>

//static void hconv_split(const fvec &input,int iwidth,int iheight, int ipitch, const fvec &kernel, int length, fvec *output,int *owidth, int *oheight, int *opitch) {
//    output->resize(iheight*ipitch);
//    *owidth = iwidth; *oheight = iheight; *opitch = ipitch;
//    for (int i = 0; i < iheight; i++) {
//        for (int j = 0; j < length-1; j++) {
//            float sum = kernel[0]*input[i*ipitch+j];
//            for (int k = 1; k < length; k++) {
//                if (j-k >= 0)
//                    sum += kernel[k]*input[i*ipitch+j-k];
//                if (j+k < iwidth)
//                    sum += kernel[k]*input[i*ipitch+j+k];
//            }
//            (*output)[i*ipitch+j] = sum;
//        }
//        for (int j = length-1; j < iwidth-length+1; j++) {
//            float sum = kernel[0]*input[i*ipitch+j];
//            for (int k = 1; k < length; k++) {
//                sum += kernel[k]*(input[i*ipitch+j-k] + input[i*ipitch+j+k]);
//            }
//            (*output)[i*ipitch+j] = sum;
//        }
//        for (int j = iwidth-length+1; j < iwidth; j++) {
//            float sum = kernel[0]*input[i*ipitch+j];
//            for (int k = 1; k < length; k++) {
//                if (j-k >= 0)
//                    sum += kernel[k]*input[i*ipitch+j-k];
//                if (j+k < iwidth)
//                    sum += kernel[k]*input[i*ipitch+j+k];
//            }
//            (*output)[i*ipitch+j] = sum;
//        }
//    }
//}

static void hconv(float *in, int w, int h, int p, float *ker, int n, float *out) {
//	hconv_split(const fvec &input,int iwidth,int iheight, int ipitch, const fvec &kernel, int length, fvec *output,int *owidth, int *oheight, int *opitch
	int i,j,k;
// interior
	for(j=1;j<h-1;j++){
		for(i =1 ;i<w-1;i++)
			#pragma unroll
			for(k=0;k<n;k++){
// supos que 0 é o indice -1 do filtro
				out[i+w*j] += in[i+w*j-k+1]*ker[k];
		}
// exterior
		for(k=0;k<n;k++){
		out[j] += in[j-k+1]*ker[k-1]; 		  // caso 1:
		out[j+w*(h-1)] += in[j+w*(h-1)-k+1]*ker[k]; // caso 2:
		out[w*j] += in[w*j-k+1]*ker[k]; 		  // caso 1:
		out[w*j+(h-1)] += in[j+w*(h-1)-k+1]*ker[k]; // caso 2:
		}
	}

}

static void vconv(float *in, int w, int h, int p, float *ker, int n, float *out) {
	int i,j,k;
// interior
	for(j=1;j<h-1;j++){
		for(i =1 ;i<w-1;i++)
			#pragma unroll 
			for(k=0;k<n;k++){
				out[i+w*j] += in[i+w*(j-k+1)]*ker[k];
		}
// exterior	
		#pragma unroll 
		for(k=0;k<n-1;k++){
			out[j] += in[j-w*(k-1)]*ker[k]; 		  // caso 1:
			out[j+w*(h-1)] += in[j+w*(h-1-k)]*ker[k]; // caso 2:
			out[w*j] += in[j-w*(k-1)]*ker[k]; 		  // caso 1:
			out[w*j+(h-1)] += in[j+w*(h-1-k)]*ker[k]; // caso 2:}
		}
	}
}

static void hrec(float *in, int w, int h, int p, float *k, int n, float *out) {
    (void) in;
    (void) w;
    (void) h;
    (void) p;
    (void) k;
    (void) n;
    (void) out;
}

static void vrec(float *in, int w, int h, int p, float *k, int n, float *out) {
    (void) in;
    (void) w;
    (void) h;
    (void) p;
    (void) k;
    (void) n;
    (void) out;
}

static constexpr std::complex<double> d1(1.41650, 1.00829);
static constexpr double d3(1.86543);

static double qs(double s) {
    return .00399341 + .4715161*s;
}

static std::complex<double> ds(std::complex<double> d, double s) {
    double q = qs(s);
    return std::polar(std::pow(std::abs(d),1.0/q), std::arg(d)/q);
}

static double ds(double d, double s) {
    return std::pow(d, 1.0/qs(s));
}

static void weights1(double s, float *b0, float *a1) {
    double d = ds(d3, s);
    *b0 = static_cast<float>(-(1.0-d)/d);
    *a1 = static_cast<float>(-1.0/d);
}

static void weights2(double s, float *b0, float *a1, float *a2) { 
    std::complex<double> d = ds(d1, s);
    double n2 = abs(d);
    n2 *= n2;
    double re = real(d);
    *b0 = static_cast<float>((1-2*re+n2)/n2);
    *a1 = static_cast<float>(-2*re/n2);
    *a2 = static_cast<float>(1/n2);
}

static void recgaussian(float s, float *b, float *a1, float *a2, float *a3) {
    float b10, a11;
    weights1(s, &b10, &a11);
    float b20, a21, a22;
    weights2(s, &b20, &a21, &a22);
    *b = b20*b10;
    *a1 = a11+a21;
    *a2 = a21*a11 + a22;
    *a3 = a22*a11;
}

int main(void) {
    int reps = 100;
    // somehow define n, h, w, and p
    int n = 3, h = 1024, w = 1024, p = 1024;
    // allocate and somehow fill it
    std::vector<float> kernel(n, 0.f);
    // allocate and somehow fill them
    std::vector<float> input(h*p, 0.f), output(h*p, 0.f), temp(h*p, 0.f);
    auto begin = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < reps; i++) {
        hconv(&input[0], w, h, p, &kernel[0], n, &temp[0]);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    std::cout << "hconv time was " << duration/reps << "ns\n";
    begin = std::chrono::high_resolution_clock::now();
for (int i = 0; i < reps; i++) {
        vconv(&temp[0], w, h, p, &kernel[0], n, &output[0]);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    std::cout << "vconv time was " << duration/reps << "ns\n";
    begin = std::chrono::high_resolution_clock::now();
for (int i = 0; i < reps; i++) {
        hrec(&temp[0], w, h, p, &kernel[0], n, &output[0]);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    std::cout << "hrec time was " << duration/reps << "ns\n";
    begin = std::chrono::high_resolution_clock::now();
for (int i = 0; i < reps; i++) {
        vrec(&temp[0], w, h, p, &kernel[0], n, &output[0]);
    }
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count();
    std::cout << "vrec time was " << duration/reps << "ns\n";
    return 0;
}

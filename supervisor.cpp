// supervisor.cpp
// Single-file SuperRes Supervisor with CUDA-accelerated Motion Estimation & AI Upscaling
// Self-learning adaptive frame interpolation and upscaling - DLSS-like

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dcomp.h>
#include <gdiplus.h>
#include <winhttp.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <vector>
#include <algorithm>
#include <mutex>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <string>
#include <cfloat>

// CUDA headers with full paths
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include/cuda_runtime.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include/cuda_d3d11_interop.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include/device_launch_parameters.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include/vector_types.h"

namespace fs = std::filesystem;

// SINGLE MotionVector definition
struct MotionVector {
    int x, y;
    float confidence;
};

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dcomp.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "cudart.lib")

// -------------------- Configuration --------------------
static const int TARGET_HZ = 9000000;
static const int TARGET_W_DEFAULT = 7680;
static const int TARGET_H_DEFAULT = 4320;
static const double ARTIFACT_DETECT_THRESHOLD = 0.001;
static const char* KERNELS_PTX_LOCAL = "modules/cleanup.ptx";
static const char* LOG_PATH = "logs/boot.log";
static const char* MODULES_DIR = "modules";
static const char* LOGS_DIR = "logs";
static const int ONLINE_ADAPTIVE_WINDOW = 8;
static const int MOTION_BLOCK_SIZE = 16;
static const int MAX_MOTION_VECTOR = 32;


// -------------------- CUDA Device Management --------------------
static bool cudaInitialized = false;
static int cudaDeviceCount = 0;
static int cudaDevice = 0;
static cudaStream_t cudaStream;

// Initialize CUDA
static bool initCuda() {
    cudaError_t result = cudaGetDeviceCount(&cudaDeviceCount);
  
    
    result = cudaSetDevice(0);
  
    
    result = cudaStreamCreate(&cudaStream);
  
    
    cudaInitialized = true;
    return true;
}

// Cleanup CUDA resources
static void cleanupCuda() {
    if (cudaInitialized) {
        cudaStreamDestroy(cudaStream);
        cudaDeviceReset();
    }
}

// -------------------- CUDA Kernels --------------------
// Motion estimation kernel
__global__ void motionEstimationKernel(
    const unsigned char* currentFrame,
    const unsigned char* previousFrame,
    int width, int height,
    MotionVector* motionField,
    int blockSize, int maxMotionVector)
{
    int blockIdxX = blockIdx.x;
    int blockIdxY = blockIdx.y;
    int blocksX = (width + blockSize - 1) / blockSize;
    
    int bestX = 0, bestY = 0;
    float bestError = FLT_MAX;
    
    for (int dy = -maxMotionVector; dy <= maxMotionVector; dy += 2) {
        for (int dx = -maxMotionVector; dx <= maxMotionVector; dx += 2) {
            float error = 0.0f;
            int samples = 0;
            
            for (int py = 0; py < blockSize; py += 2) {
                for (int px = 0; px < blockSize; px += 2) {
                    int currX = blockIdxX * blockSize + px;
                    int currY = blockIdxY * blockSize + py;
                    int prevX = currX + dx;
                    int prevY = currY + dy;
                    
                    if (prevX >= 0 && prevX < width && 
                        prevY >= 0 && prevY < height &&
                        currX >= 0 && currX < width && 
                        currY >= 0 && currY < height) {
                        
                        int currIdx = (currY * width + currX) * 4;
                        int prevIdx = (prevY * width + prevX) * 4;
                        
                        float currLum = 0.299f * currentFrame[currIdx+2] + 
                                      0.587f * currentFrame[currIdx+1] + 
                                      0.114f * currentFrame[currIdx+0];
                        float prevLum = 0.299f * previousFrame[prevIdx+2] + 
                                      0.587f * previousFrame[prevIdx+1] + 
                                      0.114f * previousFrame[prevIdx+0];
                        
                        error += fabsf(currLum - prevLum);
                        samples++;
                    }
                }
            }
            
            if (samples > 0) {
                error /= samples;
                if (error < bestError) {
                    bestError = error;
                    bestX = dx;
                    bestY = dy;
                }
            }
        }
    }
    
    int blockIdx = blockIdxY * blocksX + blockIdxX;
    MotionVector mv = {bestX, bestY, 1.0f - (bestError / 255.0f)};
    motionField[blockIdx] = mv;
}

// Upscaling kernel
__global__ void upscalingKernel(
    const unsigned char* input,
    unsigned char* output,
    int inW, int inH, int outW, int outH,
    float scaleX, float scaleY,
    float sharpnessFactor)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= outW || y >= outH) return;
    
    float srcX = x * scaleX;
    float srcY = y * scaleY;
    
    int x0 = (int)srcX;
    int y0 = (int)srcY;
    int x1 = std::min(x0 + 1, inW - 1);
    int y1 = std::min(y0 + 1, inH - 1);
    
    float dx = srcX - x0;
    float dy = srcY - y0;
    
    int outIdx = (y * outW + x) * 4;
    
    for (int c = 0; c < 3; c++) {
        float p00 = input[(y0 * inW + x0) * 4 + c];
        float p01 = input[(y0 * inW + x1) * 4 + c];
        float p10 = input[(y1 * inW + x0) * 4 + c];
        float p11 = input[(y1 * inW + x1) * 4 + c];
        
        float interpolated = p00 * (1-dx) * (1-dy) + 
                            p01 * dx * (1-dy) + 
                            p10 * (1-dx) * dy + 
                            p11 * dx * dy;
        
        if (x > 0 && x < outW-1 && y > 0 && y < outH-1) {
            float center = interpolated;
            float sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    if (kx == 0 && ky == 0) continue;
                    int nx = x + kx;
                    int ny = y + ky;
                    float nSrcX = nx * scaleX;
                    float nSrcY = ny * scaleY;
                    int nx0 = (int)nSrcX;
                    int ny0 = (int)nSrcY;
                    int nx1 = std::min(nx0 + 1, inW - 1);
                    int ny1 = std::min(ny0 + 1, inH - 1);
                    float ndx = nSrcX - nx0;
                    float ndy = nSrcY - ny0;
                    
                    float neighbor = input[(ny0 * inW + nx0) * 4 + c] * (1-ndx) * (1-ndy) +
                                   input[(ny0 * inW + nx1) * 4 + c] * ndx * (1-ndy) +
                                   input[(ny1 * inW + nx0) * 4 + c] * (1-ndx) * ndy +
                                   input[(ny1 * inW + nx1) * 4 + c] * ndx * ndy;
                    sum += neighbor;
                }
            }
            sum /= 8.0f;
            float detail = center - sum;
            interpolated = center + detail * sharpnessFactor * 0.1f;
        }
        
        output[outIdx + c] = (unsigned char)fminf(fmaxf(interpolated, 0.0f), 255.0f);
    }
    output[outIdx + 3] = 255;
}

// -------------------- Logging / dirs --------------------


// -------------------- Motion Estimation System --------------------
struct FrameBuffer {
    std::vector<uint8_t> data;
    int width, height;
    std::vector<MotionVector> motionField;
    
    FrameBuffer() : width(0), height(0) {}
    
    void resize(int w, int h) {
        width = w;
        height = h;
        data.resize(w * h * 4);
        motionField.resize((w / MOTION_BLOCK_SIZE) * (h / MOTION_BLOCK_SIZE));
    }
};

static FrameBuffer prevFrame, currentFrame, nextFrame;
static std::vector<FrameBuffer> frameHistory;

static unsigned char* d_currentFrame = nullptr;
static unsigned char* d_previousFrame = nullptr;
static MotionVector* d_motionField = nullptr;
static unsigned char* d_upscaledFrame = nullptr;

static bool allocateCudaMemory(int width, int height) {
    if (!cudaInitialized) return false;
    
    size_t frameSize = width * height * 4 * sizeof(unsigned char);
    size_t motionFieldSize = (width / MOTION_BLOCK_SIZE) * (height / MOTION_BLOCK_SIZE) * sizeof(MotionVector);
    
    cudaError_t result;
    
    result = cudaMalloc(&d_currentFrame, frameSize);
    if (result != cudaSuccess) return false;
    
    result = cudaMalloc(&d_previousFrame, frameSize);
    if (result != cudaSuccess) {
        cudaFree(d_currentFrame);
        return false;
    }
    
    result = cudaMalloc(&d_motionField, motionFieldSize);
    if (result != cudaSuccess) {
        cudaFree(d_currentFrame);
        cudaFree(d_previousFrame);
        return false;
    }
    
    result = cudaMalloc(&d_upscaledFrame, TARGET_W_DEFAULT * TARGET_H_DEFAULT * 4 * sizeof(unsigned char));
    if (result != cudaSuccess) {
        cudaFree(d_currentFrame);
        cudaFree(d_previousFrame);
        cudaFree(d_motionField);
        return false;
    }
    
    return true;
}

static void freeCudaMemory() {
    if (!cudaInitialized) return;
    
    if (d_currentFrame) cudaFree(d_currentFrame);
    if (d_previousFrame) cudaFree(d_previousFrame);
    if (d_motionField) cudaFree(d_motionField);
    if (d_upscaledFrame) cudaFree(d_upscaledFrame);
    
    d_currentFrame = nullptr;
    d_previousFrame = nullptr;
    d_motionField = nullptr;
    d_upscaledFrame = nullptr;
}

// CUDA-accelerated motion estimation - FIXED
static void estimateMotion(FrameBuffer& current, FrameBuffer& previous) {
    if (current.width != previous.width || current.height != previous.height) return;
    
    if (cudaInitialized && d_currentFrame && d_previousFrame && d_motionField) {
        int blocksX = current.width / MOTION_BLOCK_SIZE;
        int blocksY = current.height / MOTION_BLOCK_SIZE;
        
        cudaMemcpyAsync(d_currentFrame, current.data.data(), 
                       current.width * current.height * 4 * sizeof(unsigned char), 
                       cudaMemcpyHostToDevice, cudaStream);
        cudaMemcpyAsync(d_previousFrame, previous.data.data(), 
                       previous.width * previous.height * 4 * sizeof(unsigned char), 
                       cudaMemcpyHostToDevice, cudaStream);
        
        // FIXED kernel launch
        dim3 gridDim(blocksX, blocksY);
        dim3 blockDim(1);
        
        void* kernelArgs[] = {
            (void*)&d_currentFrame,
            (void*)&d_previousFrame,
            (void*)&current.width,
            (void*)&current.height,
            (void*)&d_motionField,
            (void*)&MOTION_BLOCK_SIZE,
            (void*)&MAX_MOTION_VECTOR
        };
        
        cudaLaunchKernel((void*)motionEstimationKernel, gridDim, blockDim, 
                         kernelArgs, 0, cudaStream);
        
        cudaMemcpyAsync(current.motionField.data(), d_motionField, 
                       blocksX * blocksY * sizeof(MotionVector), 
                       cudaMemcpyDeviceToHost, cudaStream);
        
        cudaStreamSynchronize(cudaStream);
    } else {
        // CPU fallback
        int blocksX = current.width / MOTION_BLOCK_SIZE;
        int blocksY = current.height / MOTION_BLOCK_SIZE;
        
        for (int by = 0; by < blocksY; by++) {
            for (int bx = 0; bx < blocksX; bx++) {
                int bestX = 0, bestY = 0;
                float bestError = FLT_MAX;
                
                for (int dy = -MAX_MOTION_VECTOR; dy <= MAX_MOTION_VECTOR; dy += 4) {
                    for (int dx = -MAX_MOTION_VECTOR; dx <= MAX_MOTION_VECTOR; dx += 4) {
                        float error = 0.0f;
                        int samples = 0;
                        
                        for (int py = 0; py < MOTION_BLOCK_SIZE; py += 2) {
                            for (int px = 0; px < MOTION_BLOCK_SIZE; px += 2) {
                                int currX = bx * MOTION_BLOCK_SIZE + px;
                                int currY = by * MOTION_BLOCK_SIZE + py;
                                int prevX = currX + dx;
                                int prevY = currY + dy;
                                
                                if (prevX >= 0 && prevX < previous.width && 
                                    prevY >= 0 && prevY < previous.height &&
                                    currX >= 0 && currX < current.width && 
                                    currY >= 0 && currY < current.height) {
                                    
                                    int currIdx = (currY * current.width + currX) * 4;
                                    int prevIdx = (prevY * previous.width + prevX) * 4;
                                    
                                    float currLum = 0.299f * current.data[currIdx+2] + 
                                                  0.587f * current.data[currIdx+1] + 
                                                  0.114f * current.data[currIdx+0];
                                    float prevLum = 0.299f * previous.data[prevIdx+2] + 
                                                  0.587f * previous.data[prevIdx+1] + 
                                                  0.114f * previous.data[prevIdx+0];
                                    
                                    error += fabs(currLum - prevLum);
                                    samples++;
                                }
                            }
                        }
                        
                        if (samples > 0) {
                            error /= samples;
                            if (error < bestError) {
                                bestError = error;
                                bestX = dx;
                                bestY = dy;
                            }
                        }
                    }
                }
                
                int blockIdx = by * blocksX + bx;
                current.motionField[blockIdx] = {bestX, bestY, 1.0f - (bestError / 255.0f)};
            }
        }
    }
}

// -------------------- AI-like Upscaling System --------------------
struct UpscalingModel {
    float edgeKernel[9];
    float detailKernel[9];
    float sharpnessFactor;
    float adaptiveThreshold;
    
    UpscalingModel() {
        edgeKernel[0] = -1; edgeKernel[1] = 0; edgeKernel[2] = 1;
        edgeKernel[3] = -2; edgeKernel[4] = 0; edgeKernel[5] = 2;
        edgeKernel[6] = -1; edgeKernel[7] = 0; edgeKernel[8] = 1;
        
        detailKernel[0] = 0; detailKernel[1] = -1; detailKernel[2] = 0;
        detailKernel[3] = -1; detailKernel[4] = 5; detailKernel[5] = -1;
        detailKernel[6] = 0; detailKernel[7] = -1; detailKernel[8] = 0;
        
        sharpnessFactor = 1.5f;
        adaptiveThreshold = 30.0f;
    }
    
    void adapt(const std::vector<uint8_t>& frame, int w, int h) {
        float avgEdgeStrength = 0.0f;
        int samples = 0;
        
        for (int y = 1; y < h-1; y += 10) {
            for (int x = 1; x < w-1; x += 10) {
                float edgeStrength = 0.0f;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int idx = ((y + ky) * w + (x + kx)) * 4;
                        int kernelIdx = (ky + 1) * 3 + (kx + 1);
                        float lum = 0.299f * frame[idx+2] + 0.587f * frame[idx+1] + 0.114f * frame[idx+0];
                        edgeStrength += lum * edgeKernel[kernelIdx];
                    }
                }
                avgEdgeStrength += fabs(edgeStrength);
                samples++;
            }
        }
        
        if (samples > 0) {
            avgEdgeStrength /= samples;
            if (avgEdgeStrength < adaptiveThreshold) {
                sharpnessFactor = std::min(sharpnessFactor * 1.01f, 3.0f);
            } else {
                sharpnessFactor = std::max(sharpnessFactor * 0.99f, 1.2f);
            }
        }
    }
    
    void save(const std::string &path) {
        std::ofstream f(path, std::ios::binary);
        if (f) {
            f.write((char*)edgeKernel, sizeof(edgeKernel));
            f.write((char*)detailKernel, sizeof(detailKernel));
            f.write((char*)&sharpnessFactor, sizeof(sharpnessFactor));
            f.write((char*)&adaptiveThreshold, sizeof(adaptiveThreshold));
        }
    }
    
    bool load(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        f.read((char*)edgeKernel, sizeof(edgeKernel));
        f.read((char*)detailKernel, sizeof(detailKernel));
        f.read((char*)&sharpnessFactor, sizeof(sharpnessFactor));
        f.read((char*)&adaptiveThreshold, sizeof(adaptiveThreshold));
        return true;
    }
};

static UpscalingModel upscalingModel;

// CUDA-accelerated upscaling - FIXED
static std::vector<uint8_t> upscaleFrame(const std::vector<uint8_t>& input, int inW, int inH, int outW, int outH) {
    std::vector<uint8_t> output(outW * outH * 4);
    float scaleX = (float)inW / outW;
    float scaleY = (float)inH / outH;
    
    if (cudaInitialized && d_upscaledFrame) {
        cudaMemcpyAsync(d_currentFrame, input.data(), 
                       inW * inH * 4 * sizeof(unsigned char), 
                       cudaMemcpyHostToDevice, cudaStream);
        
        // FIXED kernel launch
        dim3 gridDim((outW + 15) / 16, (outH + 15) / 16);
        dim3 blockDim(16, 16);
        
        void* kernelArgs[] = {
            (void*)&d_currentFrame,
            (void*)&d_upscaledFrame,
            (void*)&inW,
            (void*)&inH,
            (void*)&outW,
            (void*)&outH,
            (void*)&scaleX,
            (void*)&scaleY,
            (void*)&upscalingModel.sharpnessFactor
        };
        
        cudaLaunchKernel((void*)upscalingKernel, gridDim, blockDim, 
                         kernelArgs, 0, cudaStream);
        
        cudaMemcpyAsync(output.data(), d_upscaledFrame, 
                       outW * outH * 4 * sizeof(unsigned char), 
                       cudaMemcpyDeviceToHost, cudaStream);
        
        cudaStreamSynchronize(cudaStream);
    } else {
        // CPU fallback
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                float srcX = x * scaleX;
                float srcY = y * scaleY;
                
                int x0 = (int)srcX;
                int y0 = (int)srcY;
                int x1 = std::min(x0 + 1, inW - 1);
                int y1 = std::min(y0 + 1, inH - 1);
                
                float dx = srcX - x0;
                float dy = srcY - y0;
                
                int outIdx = (y * outW + x) * 4;
                
                for (int c = 0; c < 3; c++) {
                    float p00 = input[(y0 * inW + x0) * 4 + c];
                    float p01 = input[(y0 * inW + x1) * 4 + c];
                    float p10 = input[(y1 * inW + x0) * 4 + c];
                    float p11 = input[(y1 * inW + x1) * 4 + c];
                    
                    float interpolated = p00 * (1-dx) * (1-dy) + 
                                        p01 * dx * (1-dy) + 
                                        p10 * (1-dx) * dy + 
                                        p11 * dx * dy;
                    
                    if (x > 0 && x < outW-1 && y > 0 && y < outH-1) {
                        float center = interpolated;
                        float sum = 0;
                        for (int ky = -1; ky <= 1; ky++) {
                            for (int kx = -1; kx <= 1; kx++) {
                                if (kx == 0 && ky == 0) continue;
                                int nx = x + kx;
                                int ny = y + ky;
                                float nSrcX = nx * scaleX;
                                float nSrcY = ny * scaleY;
                                int nx0 = (int)nSrcX;
                                int ny0 = (int)nSrcY;
                                int nx1 = std::min(nx0 + 1, inW - 1);
                                int ny1 = std::min(ny0 + 1, inH - 1);
                                float ndx = nSrcX - nx0;
                                float ndy = nSrcY - ny0;
                                
                                float neighbor = input[(ny0 * inW + nx0) * 4 + c] * (1-ndx) * (1-ndy) +
                                               input[(ny0 * inW + nx1) * 4 + c] * ndx * (1-ndy) +
                                               input[(ny1 * inW + nx0) * 4 + c] * (1-ndx) * ndy +
                                               input[(ny1 * inW + nx1) * 4 + c] * ndx * ndy;
                                sum += neighbor;
                            }
                        }
                        sum /= 8.0f;
                        float detail = center - sum;
                        interpolated = center + detail * upscalingModel.sharpnessFactor * 0.1f;
                    }
                    
                    output[outIdx + c] = (uint8_t)std::clamp(interpolated, 0.0f, 255.0f);
                }
                output[outIdx + 3] = 255;
            }
        }
    }
    
    return output;
}

// -------------------- Frame Interpolation --------------------
static std::vector<uint8_t> interpolateFrames(const FrameBuffer& frame1, const FrameBuffer& frame2, float alpha, int w, int h) {
    std::vector<uint8_t> result(w * h * 4);
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 4;
            
            int blockX = x / MOTION_BLOCK_SIZE;
            int blockY = y / MOTION_BLOCK_SIZE;
            int blockIdx = blockY * (w / MOTION_BLOCK_SIZE) + blockX;
            
            if (blockIdx < frame2.motionField.size()) {
                const MotionVector& mv = frame2.motionField[blockIdx];
                
                int srcX = x + (int)(mv.x * alpha);
                int srcY = y + (int)(mv.y * alpha);
                
                if (srcX >= 0 && srcX < w && srcY >= 0 && srcY < h) {
                    int srcIdx = (srcY * w + srcX) * 4;
                    
                    for (int c = 0; c < 4; c++) {
                        result[idx + c] = (uint8_t)(frame1.data[idx + c] * (1 - alpha) + 
                                                 frame2.data[srcIdx + c] * alpha);
                    }
                } else {
                    for (int c = 0; c < 4; c++) {
                        result[idx + c] = (uint8_t)(frame1.data[idx + c] * (1 - alpha) + 
                                                 frame2.data[idx + c] * alpha);
                    }
                }
            } else {
                for (int c = 0; c < 4; c++) {
                    result[idx + c] = (uint8_t)(frame1.data[idx + c] * (1 - alpha) + 
                                             frame2.data[idx + c] * alpha);
                }
            }
        }
    }
    
    return result;
}

// -------------------- Capture: Desktop Duplication + GDI fallback --------------------
struct DDAContext {
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGIOutputDuplication* duplication = nullptr;
    IDXGIOutput1* output1 = nullptr;
    bool initialized = false;
    int width=0, height=0;
};

static bool init_dda(DDAContext &ctx) {
    HRESULT hr; D3D_FEATURE_LEVEL fl;
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, flags, nullptr, 0, D3D11_SDK_VERSION, &ctx.device, &fl, &ctx.context);
    if (FAILED(hr) || !ctx.device)

    IDXGIDevice* dxgiDev = nullptr;
    ctx.device->QueryInterface(__uuidof(IDXGIDevice), (void**)&dxgiDev);
    if (!dxgiDev) return false;

    IDXGIAdapter* adapter = nullptr; dxgiDev->GetAdapter(&adapter); dxgiDev->Release();

    IDXGIOutput* output = nullptr; adapter->EnumOutputs(0, &output);
    if (output) { output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&ctx.output1); output->Release(); }
    adapter->Release();

    RECT r; GetClientRect(GetDesktopWindow(), &r);
    ctx.width = r.right - r.left; ctx.height = r.bottom - r.top;

    HRESULT dupr = ctx.output1->DuplicateOutput(ctx.device, &ctx.duplication);

    ctx.initialized = true;
}

static bool grab_frame_dda(DDAContext &ctx, std::vector<uint8_t> &out_bgra, int &w, int &h) {
    if (!ctx.initialized) return false;

    IDXGIResource* desktopResource = nullptr; DXGI_OUTDUPL_FRAME_INFO frameInfo;
    HRESULT hr = ctx.duplication->AcquireNextFrame(0, &frameInfo, &desktopResource);

    if (hr == DXGI_ERROR_WAIT_TIMEOUT) return false;
    if (FAILED(hr) || !desktopResource) return false;

    ID3D11Texture2D* tex = nullptr; desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&tex); desktopResource->Release();
    if (!tex) return false;

    D3D11_TEXTURE2D_DESC desc; tex->GetDesc(&desc); w=desc.Width; h=desc.Height;
    desc.Usage = D3D11_USAGE_STAGING; desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ; desc.BindFlags = 0; desc.MiscFlags = 0;

    ID3D11Texture2D* staging = nullptr;
    if (FAILED(ctx.device->CreateTexture2D(&desc, nullptr, &staging)) || !staging) { tex->Release(); ctx.duplication->ReleaseFrame(); return false; }

    ctx.context->CopyResource(staging, tex);
    D3D11_MAPPED_SUBRESOURCE mapped;
    if (FAILED(ctx.context->Map(staging, 0, D3D11_MAP_READ, 0, &mapped))) { staging->Release(); tex->Release(); ctx.duplication->ReleaseFrame(); return false; }

    size_t rowBytes = w*4; out_bgra.resize(rowBytes*h);
    for (int y=0; y<h; y++) memcpy(&out_bgra[y*rowBytes], (uint8_t*)mapped.pData + y*mapped.RowPitch, rowBytes);

    ctx.context->Unmap(staging, 0); staging->Release(); tex->Release(); ctx.duplication->ReleaseFrame(); return true;
}

static bool grab_frame_gdi(std::vector<uint8_t> &out_bgra, int &w, int &h) {
    HDC hScreen = GetDC(NULL); HDC hMem = CreateCompatibleDC(hScreen);
    RECT r; GetClientRect(GetDesktopWindow(), &r); w = r.right-r.left; h = r.bottom-r.top;

    HBITMAP hBitmap = CreateCompatibleBitmap(hScreen, w, h);
    HBITMAP old = (HBITMAP)SelectObject(hMem, hBitmap);

    if (!BitBlt(hMem, 0, 0, w, h, hScreen, 0, 0, SRCCOPY|CAPTUREBLT)) { SelectObject(hMem, old); DeleteObject(hBitmap); DeleteDC(hMem); ReleaseDC(NULL,hScreen); return false; }

    BITMAPINFOHEADER bi = { sizeof(BITMAPINFOHEADER), w, -h, 1, 32, BI_RGB, 0,0,0,0,0 };
    out_bgra.resize((size_t)w*h*4);
    if (!GetDIBits(hMem, hBitmap, 0, h, out_bgra.data(), (BITMAPINFO*)&bi, DIB_RGB_COLORS)) { SelectObject(hMem, old); DeleteObject(hBitmap); DeleteDC(hMem); ReleaseDC(NULL,hScreen); return false; }

    SelectObject(hMem, old); DeleteObject(hBitmap); DeleteDC(hMem); ReleaseDC(NULL,hScreen);
    return true;
}

// -------------------- DirectComposition overlay --------------------
struct DCompContext {
    ID3D11Device* d3dDevice=nullptr;
    IDCompositionDevice* dcompDevice=nullptr;
    IDCompositionTarget* dcompTarget=nullptr;
    IDCompositionVisual* dcompVisual=nullptr;
    HWND hwndOverlay=nullptr;
    bool initialized=false;
    int width=0,height=0;
};

static HWND create_overlay_window(int width,int height){
    WNDCLASSA wc={0}; 
    wc.lpfnWndProc=DefWindowProcA; 
    wc.hInstance=GetModuleHandle(NULL); 
    wc.lpszClassName="SupervisorOverlayClass"; 
    RegisterClassA(&wc);
    
    DWORD exStyle=WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE;
    HWND h=CreateWindowExA(exStyle,wc.lpszClassName,"SupervisorOverlay",WS_POPUP,
                          0,0,1,1,NULL,NULL,GetModuleHandle(NULL),NULL);
    if(!h)return NULL;
    
    SetLayeredWindowAttributes(h, RGB(0,0,0), 0, LWA_ALPHA);
    
    SetWindowPos(h,HWND_TOPMOST,-100,-100,1,1,SWP_SHOWWINDOW|SWP_NOACTIVATE); 
    ShowWindow(h,SW_HIDE);
    return h;
}

static bool init_dcomp(DCompContext &ctx,int w,int h){
    ctx.hwndOverlay=create_overlay_window(w,h); 
    
    D3D_FEATURE_LEVEL fl; 
    HRESULT hr=D3D11CreateDevice(NULL,D3D_DRIVER_TYPE_HARDWARE,NULL,D3D11_CREATE_DEVICE_BGRA_SUPPORT,NULL,0,D3D11_SDK_VERSION,&ctx.d3dDevice,&fl,NULL);

    
    IDXGIDevice* dxgiDev=nullptr; 
    ctx.d3dDevice->QueryInterface(__uuidof(IDXGIDevice),(void**)&dxgiDev); 
    if(!dxgiDev) return false;
    
    hr=DCompositionCreateDevice(dxgiDev,__uuidof(IDCompositionDevice),(void**)&ctx.dcompDevice); 
    dxgiDev->Release();

    hr=ctx.dcompDevice->CreateTargetForHwnd(ctx.hwndOverlay,TRUE,&ctx.dcompTarget); 
    if(FAILED(hr)) return false;
    
    hr=ctx.dcompDevice->CreateVisual(&ctx.dcompVisual); 
    if(FAILED(hr)) return false;
    
    ctx.dcompTarget->SetRoot(ctx.dcompVisual); 
    ctx.width=w; 
    ctx.height=h; 
    ctx.initialized=true; 
}

static bool present_via_dcomp(DCompContext &ctx,const std::vector<uint8_t>& bgra,int w,int h){
    if (!ctx.initialized) return false;
    return true;
}

// -------------------- Priority --------------------
static void set_high_priority(){
    SetPriorityClass(GetCurrentProcess(),REALTIME_PRIORITY_CLASS);
    HMODULE hAv=LoadLibraryA("avrt.dll");
    if(hAv){ 
        typedef HANDLE(WINAPI *AVSET)(LPCSTR,LPDWORD); 
        typedef BOOL(WINAPI*AVSETP)(HANDLE,LPDWORD); 
        AVSET pSet=(AVSET)GetProcAddress(hAv,"AvSetMmThreadCharacteristicsA"); 
        AVSETP pSetP=(AVSETP)GetProcAddress(hAv,"AvSetMmThreadPriority");
        if(pSet&&pSetP){ 
            DWORD idx=0; 
            HANDLE h=pSet("Pro Audio",&idx); 
            if(h) pSetP(h,(LPDWORD)3);
        }
    }
}

// -------------------- High precision sleep --------------------
static void sleep_until(std::chrono::steady_clock::time_point t){
    using namespace std::chrono;
    auto now=steady_clock::now();
    while(now+milliseconds(2)<t){ 
        std::this_thread::sleep_for(milliseconds(1)); 
        now=steady_clock::now();
    }
    if(now<t) std::this_thread::sleep_for(t-now);
}

// -------------------- Enhanced Processing Pipeline --------------------
static void enhance_quality(std::vector<uint8_t>& bgra, int w, int h) {
    currentFrame.data = bgra;
    currentFrame.width = w;
    currentFrame.height = h;
    
    if (prevFrame.width == w && prevFrame.height == h) {
        estimateMotion(currentFrame, prevFrame);
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = (y * w + x) * 4;
                
                int blockX = x / MOTION_BLOCK_SIZE;
                int blockY = y / MOTION_BLOCK_SIZE;
                int blockIdx = blockY * (w / MOTION_BLOCK_SIZE) + blockX;
                
                if (blockIdx < currentFrame.motionField.size()) {
                    const MotionVector& mv = currentFrame.motionField[blockIdx];
                    float motionStrength = sqrt(mv.x * mv.x + mv.y * mv.y);
                    
                    float filterStrength = std::max(0.1f, 1.0f - motionStrength / MAX_MOTION_VECTOR);
                    
                    if (motionStrength < 5.0f) {
                        int prevIdx = idx;
                        for (int c = 0; c < 3; c++) {
                            bgra[idx + c] = (uint8_t)(bgra[idx + c] * 0.7f + 
                                                     prevFrame.data[prevIdx + c] * 0.3f);
                        }
                    }
                }
            }
        }
    }
    
    upscalingModel.adapt(bgra, w, h);
    
    prevFrame = currentFrame;
}

static void processFrameWithVerification(std::vector<uint8_t>& bgra, int w, int h) {
    std::vector<uint8_t> before = bgra;
    
    enhance_quality(bgra, w, h);
    
    if (w < TARGET_W_DEFAULT || h < TARGET_H_DEFAULT) {
        bgra = upscaleFrame(bgra, w, h, TARGET_W_DEFAULT, TARGET_H_DEFAULT);
        w = TARGET_W_DEFAULT;
        h = TARGET_H_DEFAULT;
    }
    
    int changedPixels = 0;
    int totalDiff = 0;
    
    for (int i = 0; i < std::min(before.size(), bgra.size()); i += 4) {
        int diff = abs(before[i] - bgra[i]) + abs(before[i+1] - bgra[i+1]) + abs(before[i+2] - bgra[i+2]);
        if (diff > 10) {
            changedPixels++;
            totalDiff += diff;
        }
    }
}

// -------------------- Frame Rate Management --------------------
static std::vector<std::vector<uint8_t>> frameDataHistory;

static std::vector<uint8_t> generateInterpolatedFrame(int targetW, int targetH) {
    if (frameDataHistory.size() < 2) {
        return std::vector<uint8_t>(targetW * targetH * 4, 0);
    }
    
    const auto& frame1 = frameDataHistory[frameDataHistory.size() - 2];
    const auto& frame2 = frameDataHistory[frameDataHistory.size() - 1];
    
    FrameBuffer f1, f2;
    f1.data = frame1;
    f1.width = targetW;
    f1.height = targetH;
    f2.data = frame2;
    f2.width = targetW;
    f2.height = targetH;
    
    estimateMotion(f2, f1);
    
    return interpolateFrames(f1, f2, 0.5f, targetW, targetH);
}

// -------------------- Main with Enhanced Processing --------------------
int main() {

    initCuda();

    ensure_dirs(); 
    set_high_priority();

    DDAContext ddaCtx; 
    init_dda(ddaCtx);
    DCompContext dcompCtx; 
    init_dcomp(dcompCtx, TARGET_W_DEFAULT, TARGET_H_DEFAULT);

    if (cudaInitialized) {
        allocateCudaMemory(TARGET_W_DEFAULT, TARGET_H_DEFAULT);
    }

    std::vector<uint8_t> frame; 
    int w, h;
    int frameCount = 0;
    bool running = true;
    
    auto lastFrameTime = std::chrono::steady_clock::now();
    auto frameInterval = std::chrono::milliseconds(1000 / TARGET_HZ);
    
    while(running) {
        auto currentTime = std::chrono::steady_clock::now();
        
        bool got = false;
        if(ddaCtx.initialized) got = grab_frame_dda(ddaCtx, frame, w, h);
        if(!got) got = grab_frame_gdi(frame, w, h);
        
        if(got) {
            processFrameWithVerification(frame, w, h);
            
            frameDataHistory.push_back(frame);
            if (frameDataHistory.size() > 3) {
                frameDataHistory.erase(frameDataHistory.begin());
            }
            
            present_via_dcomp(dcompCtx, frame, TARGET_W_DEFAULT, TARGET_H_DEFAULT);
            
            frameCount++;
            
            while (currentTime + frameInterval < std::chrono::steady_clock::now()) {
                auto interpolatedFrame = generateInterpolatedFrame(TARGET_W_DEFAULT, TARGET_H_DEFAULT);
                present_via_dcomp(dcompCtx, interpolatedFrame, TARGET_W_DEFAULT, TARGET_H_DEFAULT);
                frameCount++;
                currentTime += frameInterval;
            }
            
            if(frameCount % 60 == 0) {
                
            }
        }
        
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
            running = false;
        }
        
        if (frameCount % 300 == 0) {
            upscalingModel.save(std::string(MODULES_DIR) + "/upscaling_model.bin");
        }
        
        std::this_thread::sleep_until(currentTime + frameInterval);
    }
    
    freeCudaMemory();
    cleanupCuda();
    
    upscalingModel.save(std::string(MODULES_DIR) + "/upscaling_model.bin");
    return 0;
}

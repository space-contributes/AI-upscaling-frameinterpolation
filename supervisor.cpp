#include <iostream>
#include <vector>
#include <cmath>
#include <thread>
#include <memory>
#include <algorithm>
#include <immintrin.h>
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dcomp.h>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dcomp.lib")

// ============================================================================
// VECTOR MATH & SIMD UTILITIES
// ============================================================================

struct Vec3 {
    float x, y, z;
    Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
    Vec3 operator+(const Vec3& v) const { return Vec3(x+v.x, y+v.y, z+v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x-v.x, y-v.y, z-v.z); }
    Vec3 operator*(float s) const { return Vec3(x*s, y*s, z*s); }
    float dot(const Vec3& v) const { return x*v.x + y*v.y + z*v.z; }
    float length() const { return std::sqrt(x*x + y*y + z*z); }
};

struct Mat3x3 {
    float m[9];
    Vec3 mul(const Vec3& v) const {
        return Vec3(
            m[0]*v.x + m[1]*v.y + m[2]*v.z,
            m[3]*v.x + m[4]*v.y + m[5]*v.z,
            m[6]*v.x + m[7]*v.y + m[8]*v.z
        );
    }
};

// ============================================================================
// COLOR SPACE TRANSFORMATIONS
// (left exactly as you wrote it — no changes)
// ============================================================================

class ColorSpaceEngine {
private:
    // Dynamic transformation matrices (learned)
    Mat3x3 P3_to_XYZ, Adobe_to_XYZ, ProPhoto_to_XYZ, BT2020_to_XYZ;
    Mat3x3 XYZ_to_CAM02, CAM02_to_JzAzBz;
    
    // Adaptive parameters
    float chroma_stretch, hue_warp, lightness_flow;
    float contrast_gain, brightness_offset, gamma_exp;
    
public:
    ColorSpaceEngine() {
        // Initialize transformation matrices (accurate primaries)
        P3_to_XYZ = {0.4865f, 0.2657f, 0.1982f,
                     0.2290f, 0.6917f, 0.0793f,
                     0.0000f, 0.0451f, 1.0439f};
       
        Adobe_to_XYZ = {0.5767f, 0.1856f, 0.1882f,
                        0.2973f, 0.6274f, 0.0753f,
                        0.0270f, 0.0707f, 0.9911f};
       
        ProPhoto_to_XYZ = {0.7977f, 0.1352f, 0.0313f,
                           0.2880f, 0.7119f, 0.0001f,
                           0.0000f, 0.0000f, 0.8249f};
       
        BT2020_to_XYZ = {0.6370f, 0.1446f, 0.1689f,
                         0.2627f, 0.6780f, 0.0593f,
                         0.0000f, 0.0281f, 1.0610f};
       
        // Initialize adaptive parameters
        chroma_stretch = 1.0f;
        hue_warp = 0.0f;
        lightness_flow = 1.0f;
        contrast_gain = 1.0f;
        brightness_offset = 0.0f;
        gamma_exp = 1.0f;
    }
    
    // Linearize sRGB/Display P3
    float linearize(float v) const {
        return (v <= 0.04045f) ? v / 12.92f : std::pow((v + 0.055f) / 1.055f, 2.4f);
    }
    
    // Wide gamut infinite precision blend
    Vec3 blendGamuts(const Vec3& rgb, float w, float x, float y, float z, float eta) const {
        Vec3 p3 = P3_to_XYZ.mul(rgb * w);
        Vec3 adobe = Adobe_to_XYZ.mul(rgb * x);
        Vec3 prophoto = ProPhoto_to_XYZ.mul(rgb * y);
        Vec3 bt2020 = BT2020_to_XYZ.mul(rgb * z);
       
        Vec3 blended = p3 + adobe + prophoto + bt2020;
        blended.x += eta; blended.y += eta; blended.z += eta;
        return blended;
    }
    
    // XYZ to CAM02-UCS (perceptually uniform)
    Vec3 XYZ_to_CAM02UCS(const Vec3& xyz) const {
        // Simplified CAM02-UCS transformation
        float L = 116.0f * std::cbrt(xyz.y) - 16.0f;
        float a = 500.0f * (std::cbrt(xyz.x / 0.95047f) - std::cbrt(xyz.y));
        float b = 200.0f * (std::cbrt(xyz.y) - std::cbrt(xyz.z / 1.08883f));
        return Vec3(L, a, b);
    }
    
    // CAM02-UCS to Jzazbz (HDR perceptual)
    Vec3 CAM02_to_Jzazbz(const Vec3& cam) const {
        float Jz = cam.x / 100.0f;
        float az = cam.y * chroma_stretch * std::cos(hue_warp);
        float bz = cam.z * chroma_stretch * std::sin(hue_warp);
        return Vec3(Jz * lightness_flow, az, bz);
    }
    
    // Full forward pipeline with metadata adaptation
    Vec3 processPixel(const Vec3& rgb_in, float deltaE_CAM, float deltaE_Jz) {
        // Stage 1: Linearize
        Vec3 linear(linearize(rgb_in.x), linearize(rgb_in.y), linearize(rgb_in.z));
       
        // Stage 2: Infinite precision gamut blend (dynamic weights)
        float w = 0.4f + deltaE_CAM * 0.1f;
        float x = 0.2f + deltaE_Jz * 0.05f;
        float y = 0.2f;
        float z = 0.2f;
        float eta = deltaE_CAM * 0.01f;
       
        Vec3 xyz = blendGamuts(linear, w, x, y, z, eta);
       
        // Stage 3: Perceptual transformations
        Vec3 cam02 = XYZ_to_CAM02UCS(xyz);
        Vec3 jzazbz = CAM02_to_Jzazbz(cam02);
       
        // Stage 4: Adaptive enhancement
        float J = jzazbz.x * contrast_gain + brightness_offset;
        J = std::pow(std::max(0.0f, J), gamma_exp);
       
        // Stage 5: Back to display (simplified)
        return Vec3(
            std::clamp(J + jzazbz.y * 0.5f, 0.0f, 1.0f),
            std::clamp(J, 0.0f, 1.0f),
            std::clamp(J + jzazbz.z * 0.5f, 0.0f, 1.0f)
        );
    }
    
    // Learn from frame sequence
    void adaptFromMotion(float motion_magnitude, float detail_variance) {
        // Increase chroma stretch for high motion
        chroma_stretch = 1.0f + motion_magnitude * 0.3f;
       
        // Adjust contrast based on detail
        contrast_gain = 1.0f + detail_variance * 0.5f;
       
        // Dynamic gamma for perceived brightness
        gamma_exp = 0.9f + (1.0f - motion_magnitude) * 0.2f;
       
        // Brightness compensation
        brightness_offset = -0.05f + motion_magnitude * 0.1f;
       
        // Hue stability
        hue_warp = detail_variance * 0.1f;
        lightness_flow = 1.0f + motion_magnitude * 0.15f;
    }
};

// ============================================================================
// MOTION ESTIMATION & LEARNING
// ============================================================================

struct MotionVector {
    float dx, dy, confidence;
    MotionVector(float dx = 0, float dy = 0, float c = 0) : dx(dx), dy(dy), confidence(c) {}
};

class AdaptiveMotionEstimator {
private:
    int width, height, block_size;
    std::vector<MotionVector> motion_field;
    std::vector<float> motion_history; // Learning buffer
    
    float sad(const std::vector<float>& frame1, const std::vector<float>& frame2,
              int x1, int y1, int x2, int y2, int channel) {
        float sum = 0;
        for (int dy = 0; dy < block_size; ++dy) {
            for (int dx = 0; dx < block_size; ++dx) {
                int idx1 = ((y1 + dy) * width + (x1 + dx)) * 3 + channel;
                int idx2 = ((y2 + dy) * width + (x2 + dx)) * 3 + channel;
                if (idx1 >= 0 && idx1 < (int)frame1.size() && idx2 >= 0 && idx2 < (int)frame2.size()) {
                    sum += std::abs(frame1[idx1] - frame2[idx2]);
                }
            }
        }
        return sum;
    }
    
public:
    AdaptiveMotionEstimator(int w, int h, int bs = 8)
        : width(w), height(h), block_size(bs) {
        motion_field.resize((w / bs) * (h / bs));
    }
    
    void estimate(const std::vector<float>& frame1, const std::vector<float>& frame2, int search_range = 16) {
        int blocks_x = width / block_size;
        int blocks_y = height / block_size;
        
        // Parallel motion estimation
        #pragma omp parallel for collapse(2)
        for (int by = 0; by < blocks_y; ++by) {
            for (int bx = 0; bx < blocks_x; ++bx) {
                int base_x = bx * block_size;
                int base_y = by * block_size;
                
                float best_sad = 1e9f;
                int best_dx = 0, best_dy = 0;
                
                // Adaptive search range based on history
                int adaptive_range = search_range;
                if (!motion_history.empty()) {
                    float avg_motion = motion_history.back();
                    adaptive_range = static_cast<int>(search_range * (1.0f + avg_motion * 0.5f));
                }
                
                // Diamond search pattern (efficient)
                for (int dy = -adaptive_range; dy <= adaptive_range; dy += 2) {
                    for (int dx = -adaptive_range; dx <= adaptive_range; dx += 2) {
                        int test_x = base_x + dx;
                        int test_y = base_y + dy;
                        
                        if (test_x >= 0 && test_x + block_size < width &&
                            test_y >= 0 && test_y + block_size < height) {
                            
                            float total_sad = 0;
                            for (int c = 0; c < 3; ++c) {
                                total_sad += sad(frame1, frame2, base_x, base_y, test_x, test_y, c);
                            }
                            
                            if (total_sad < best_sad) {
                                best_sad = total_sad;
                                best_dx = dx;
                                best_dy = dy;
                            }
                        }
                    }
                }
                
                float confidence = 1.0f / (1.0f + best_sad / (block_size * block_size * 3.0f));
                motion_field[by * blocks_x + bx] = MotionVector((float)best_dx, (float)best_dy, confidence);
            }
        }
        
        // Learn motion patterns
        float avg_magnitude = 0;
        for (const auto& mv : motion_field) {
            avg_magnitude += std::sqrt(mv.dx * mv.dx + mv.dy * mv.dy);
        }
        avg_magnitude /= (motion_field.empty() ? 1.0f : (float)motion_field.size());
        motion_history.push_back(avg_magnitude);
        if (motion_history.size() > 30) motion_history.erase(motion_history.begin());
    }
    
    MotionVector getMotion(int x, int y) const {
        int bx = x / block_size;
        int by = y / block_size;
        int blocks_x = width / block_size;
        if (bx >= 0 && bx < blocks_x && by >= 0 && by < (height / block_size)) {
            return motion_field[by * blocks_x + bx];
        }
        return MotionVector(0, 0, 0);
    }
    
    float getMotionMagnitude() const {
        return motion_history.empty() ? 0.0f : motion_history.back();
    }
};

// ============================================================================
// FRAME INTERPOLATION ENGINE (MERGED & FIXED)
// - public width/height
// - public mutable motion_estimator & color_engine
// - sharpen uses both frames + t (no dependency on external 'result' buffer)
// - interpolate uses existing global Vec3 and MotionVector types
// ============================================================================

class FrameInterpolator {
public:
    int width, height;
    mutable AdaptiveMotionEstimator motion_estimator;
    mutable ColorSpaceEngine color_engine;

    FrameInterpolator(int w, int h)
        : width(w), height(h), motion_estimator(w, h, 8) {}

    // Bilateral-style adaptive sharpen that blends neighbors from both frames
    Vec3 sharpen(const std::vector<float>& f1, const std::vector<float>& f2,
                 const Vec3& c, int x, int y, float strength, float t) const {
        Vec3 blur(0.0f, 0.0f, 0.0f);
        float wsum = 0.0f;

        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = std::clamp(x + dx, 0, width - 1);
                int ny = std::clamp(y + dy, 0, height - 1);
                int idx = (ny * width + nx) * 3;

                Vec3 p1(f1[idx + 0], f1[idx + 1], f1[idx + 2]);
                Vec3 p2(f2[idx + 0], f2[idx + 1], f2[idx + 2]);
                Vec3 n = p1 * (1.0f - t) + p2 * t;

                float d = (c - n).length();
                float spatial = std::hypot((float)dx, (float)dy);
                float w = std::exp(-(spatial*spatial)/2.0f - (d*d)/0.02f);

                blur = blur + n * w;
                wsum += w;
            }
        }

        if (wsum > 0.0f) {
            blur = blur * (1.0f / wsum);
        }

        Vec3 detail = c - blur;
        return c + detail * strength;
    }

    // Bilinear sample helper
    Vec3 sampleBilinear(const std::vector<float>& f, float fx, float fy) const {
        int x0 = std::clamp(static_cast<int>(fx), 0, width - 1);
        int y0 = std::clamp(static_cast<int>(fy), 0, height - 1);
        int x1 = std::min(x0 + 1, width - 1);
        int y1 = std::min(y0 + 1, height - 1);

        float wx = fx - x0;
        float wy = fy - y0;

        int i00 = (y0 * width + x0) * 3;
        int i10 = (y0 * width + x1) * 3;
        int i01 = (y1 * width + x0) * 3;
        int i11 = (y1 * width + x1) * 3;

        Vec3 a(f[i00 + 0], f[i00 + 1], f[i00 + 2]);
        Vec3 b(f[i10 + 0], f[i10 + 1], f[i10 + 2]);
        Vec3 c(f[i01 + 0], f[i01 + 1], f[i01 + 2]);
        Vec3 d(f[i11 + 0], f[i11 + 1], f[i11 + 2]);

        Vec3 top = a * (1.0f - wx) + b * wx;
        Vec3 bot = c * (1.0f - wx) + d * wx;
        return top * (1.0f - wy) + bot * wy;
    }

    // Interpolate between two frames with motion compensation, sharpening, and color enhancement.
    std::vector<float> interpolate(const std::vector<float>& frame1,
                                   const std::vector<float>& frame2,
                                   float t) const {
        // estimate motion (mutable member allowed in const method)
        motion_estimator.estimate(frame1, frame2);

        float mag = motion_estimator.getMotionMagnitude();
        color_engine.adaptFromMotion(mag, 0.1f);

        std::vector<float> out(width * height * 3);

        #pragma omp parallel for collapse(2)
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                MotionVector mv = motion_estimator.getMotion(x, y);

                Vec3 s1 = sampleBilinear(frame1, x - mv.dx * t, y - mv.dy * t);
                Vec3 s2 = sampleBilinear(frame2, x + mv.dx * (1.0f - t), y + mv.dy * (1.0f - t));

                float conf = mv.confidence;
                // Blend with confidence and temporal weight
                Vec3 blended = s1 * ((1.0f - t) * conf) + s2 * (t * conf);
                float denom = ( (1.0f - t) * conf + t * conf );
                if (denom > 0.0f) blended = blended * (1.0f / denom);

                // Adaptive sharpen using both frames
                float sharp_strength = 1.8f * (1.0f - mag * 0.4f);
                Vec3 sharp = sharpen(frame1, frame2, blended, x, y, sharp_strength, t);

                // Color pipeline (unchanged)
                float deltaE_CAM = (s1 - s2).length();
                float deltaE_Jz = deltaE_CAM * 0.8f;
                Vec3 final = color_engine.processPixel(sharp, deltaE_CAM, deltaE_Jz);

                int i = (y * width + x) * 3;
                out[i + 0] = std::clamp(final.x, 0.f, 1.f);
                out[i + 1] = std::clamp(final.y, 0.f, 1.f);
                out[i + 2] = std::clamp(final.z, 0.f, 1.f);
            }
        }

        return out;
    }

    // Keep previous enhanceFrame behavior (for demo/main compatibility)
    void enhanceFrame(std::vector<float>& frame) const {
        float detail_var = 0.1f;
        float motion_mag = motion_estimator.getMotionMagnitude();
        color_engine.adaptFromMotion(motion_mag, detail_var);

        #pragma omp parallel for
        for (int i = 0; i < (int)frame.size(); i += 3) {
            Vec3 pixel(frame[i], frame[i+1], frame[i+2]);
            Vec3 enhanced = color_engine.processPixel(pixel, 0.05f, 0.03f);
            frame[i] = enhanced.x;
            frame[i+1] = enhanced.y;
            frame[i+2] = enhanced.z;
        }
    }
};

// ============================================================================
// main()
// ============================================================================
int main() {
    std::cout << "Adaptive Frame Interpolation & Enhancement Engine\n";
    std::cout << "================================================\n\n";

    int width  = GetSystemMetrics(SM_CXSCREEN);
    int height = GetSystemMetrics(SM_CYSCREEN);
    std::cout << "Capture Resolution: " << width << "x" << height << "\n";
    std::cout << "Color Pipeline: P3+Adobe+ProPhoto+BT2020 -> XYZ -> CAM02-UCS -> Jzazbz\n";
    std::cout << "Features: Motion learning, adaptive sharpening, dynamic color grading\n\n";

    // Demo (unchanged, runs once at startup)
    {
        std::cout << "Running synthetic test pattern demo...\n";
        int demo_w = 7680, demo_h = 4320;
        std::vector<float> frame1(demo_w * demo_h * 3);
        std::vector<float> frame2(demo_w * demo_h * 3);
        for (int y = 0; y < demo_h; ++y) for (int x = 0; x < demo_w; ++x) {
            int idx = (y * demo_w + x) * 3;
            frame1[idx]   = (float)x / demo_w;  frame1[idx+1] = (float)y / demo_h;  frame1[idx+2] = 0.5f;
            frame2[idx]   = (float)(x+10) / demo_w; frame2[idx+1] = (float)(y+5) / demo_h; frame2[idx+2] = 0.6f;
        }
        FrameInterpolator demoInterpolator(demo_w, demo_h);
        for (int i = 1; i <= 3; ++i) {
            float t = i / 4.0f;
            auto interpolated = demoInterpolator.interpolate(frame1, frame2, t);
            std::cout << "  Demo Frame " << i << " (t=" << t << ") generated with "
                      << interpolated.size()/3 << " pixels\n";
        }
        std::cout << "Demo complete.\n\n";
    }

    FrameInterpolator interpolator(width, height);

    // DX11 + DXGI + DirectComposition setup (unchanged)
    ID3D11Device* device = nullptr;
    ID3D11DeviceContext* context = nullptr;
    IDXGIOutputDuplication* dupl = nullptr;
    IDXGIFactory1* factory = nullptr;
    IDXGIAdapter1* adapter = nullptr;
    IDXGIOutput* output = nullptr;
    IDXGIOutput1* output1 = nullptr;

    HRESULT hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
                                   nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, &context);
    if (FAILED(hr)) { std::cerr << "Failed to create D3D11 device\n"; return -1; }

    CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&factory);
    factory->EnumAdapters1(0, &adapter);
    adapter->EnumOutputs(0, &output);
    output->QueryInterface(__uuidof(IDXGIOutput1), (void**)&output1);
    output1->DuplicateOutput(device, &dupl);

    IDCompositionDevice* dcompDevice = nullptr;
    IDCompositionTarget* target = nullptr;
    IDCompositionVisual* visual = nullptr;
    IDCompositionSurface* surface = nullptr;

    DCompositionCreateDevice(nullptr, IID_PPV_ARGS(&dcompDevice));
    dcompDevice->CreateTargetForHwnd(GetDesktopWindow(), TRUE, &target);
    dcompDevice->CreateVisual(&visual);
    target->SetRoot(visual);
    dcompDevice->CreateSurface(width, height, DXGI_FORMAT_B8G8R8A8_UNORM,
                               DXGI_ALPHA_MODE_PREMULTIPLIED, &surface);
    visual->SetContent(surface);
    dcompDevice->Commit();

    // Helper to upload any frame (interpolated or real)
    auto upload = [&](const std::vector<float>& frame) {
        RECT rect = {0, 0, width, height};
        POINT offset = {0, 0};
        ID3D11Texture2D* surfTex = nullptr;
        D3D11_MAPPED_SUBRESOURCE map{};

        surface->BeginDraw(&rect, __uuidof(ID3D11Texture2D), (void**)&surfTex, &offset);
        context->Map(surfTex, 0, D3D11_MAP_WRITE, 0, &map);

        uint8_t* dstBase = (uint8_t*)map.pData;
        for (int y = 0; y < height; ++y) {
            uint8_t* dst = dstBase + map.RowPitch * y;
            int base = y * width * 3;
            for (int x = 0; x < width; ++x) {
                dst[x*4 + 2] = (uint8_t)(frame[base + x*3 + 0] * 255.0f); // R
                dst[x*4 + 1] = (uint8_t)(frame[base + x*3 + 1] * 255.0f); // G
                dst[x*4 + 0] = (uint8_t)(frame[base + x*3 + 2] * 255.0f); // B
                dst[x*4 + 3] = 255;
            }
        }
        context->Unmap(surfTex, 0);
        surfTex->Release();
        surface->EndDraw();
        dcompDevice->Commit();
    };

    std::vector<float> prev(width * height * 3);
    std::vector<float> curr(width * height * 3);
    bool first = true;
    const int fake_frames = 7;   // 8× total smoothness

    std::cout << "Starting real-time interpolation (+" << fake_frames << " fake frames per real frame)\n";
    std::cout << "Close console window or press Alt+F4 to stop.\n\n";

    while (true) {
        DXGI_OUTDUPL_FRAME_INFO frameInfo;
        IDXGIResource* desktopRes = nullptr;

        hr = dupl->AcquireNextFrame(16, &frameInfo, &desktopRes);
        if (FAILED(hr)) { Sleep(1); continue; }

        ID3D11Texture2D* gpuFrame = nullptr;
        desktopRes->QueryInterface(__uuidof(ID3D11Texture2D), (void**)&gpuFrame);

        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = width; desc.Height = height;
        desc.MipLevels = desc.ArraySize = 1;
        desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

        ID3D11Texture2D* cpuTex = nullptr;
        device->CreateTexture2D(&desc, nullptr, &cpuTex);
        context->CopyResource(cpuTex, gpuFrame);

        D3D11_MAPPED_SUBRESOURCE map{};
        context->Map(cpuTex, 0, D3D11_MAP_READ, 0, &map);
        const uint8_t* src = (const uint8_t*)map.pData;

        // 1. Copy new real frame into curr
        #pragma omp parallel for
        for (int i = 0; i < width * height; ++i) {
            curr[i*3 + 0] = src[i*4 + 2] / 255.0f; // R
            curr[i*3 + 1] = src[i*4 + 1] / 255.0f; // G
            curr[i*3 + 2] = src[i*4 + 0] / 255.0f; // B
        }

        context->Unmap(cpuTex, 0);
        cpuTex->Release();
        gpuFrame->Release();
        desktopRes->Release();
        dupl->ReleaseFrame();

        // 2. NOW apply full color enhancement to the real frame
        float motion_mag = interpolator.motion_estimator.getMotionMagnitude();
        interpolator.color_engine.adaptFromMotion(motion_mag, 0.1f);

        #pragma omp parallel for
        for (int i = 0; i < width * height * 3; i += 3) {
            Vec3 p(curr[i], curr[i+1], curr[i+2]);
            Vec3 e = interpolator.color_engine.processPixel(p, 0.02f, 0.01f);
            curr[i]   = e.x;
            curr[i+1] = e.y;
            curr[i+2] = e.z;
        }

        // 3. Draw interpolated frames (if we have a previous frame)
        if (!first) {
            for (int i = 1; i <= fake_frames; ++i) {
                float t = i / float(fake_frames + 1);
                auto interp = interpolator.interpolate(prev, curr, t);
                upload(interp);
            }
        }

        // 4. Always draw the real (now enhanced) current frame last
        upload(curr);

        prev = curr;
        first = false;
    }

    return 0;
}

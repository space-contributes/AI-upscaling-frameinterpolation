nvcc -arch=sm_89 -O3 -lineinfo supervisor.obj supervisor_kernels.obj ^
  "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/lib/x64/cudart.lib" ^
  -Xlinker "/NODEFAULTLIB:libcmt.lib" ^
  -Xlinker "/NODEFAULTLIB:libcpmt.lib" ^
  -Xlinker "/NODEFAULTLIB:libucrt.lib" ^
  -Xlinker "/NODEFAULTLIB:libvcruntime.lib" ^
  -ld3d11 -ldxgi -ldcomp -ldwmapi -lshcore -lwinmm -lgdi32 -luser32 -luuid -lole32 ^
  -o SuperResSupervisor.exe

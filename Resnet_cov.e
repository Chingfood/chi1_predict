2020-02-07 18:27:21.095492: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcuda.so.1
2020-02-07 18:27:21.190288: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:05:00.0
2020-02-07 18:27:21.313619: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-02-07 18:27:21.709590: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-02-07 18:27:22.656555: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-02-07 18:27:22.837537: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-02-07 18:27:23.260116: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-02-07 18:27:23.524075: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-02-07 18:27:24.092826: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-02-07 18:27:24.096632: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-02-07 18:27:24.098542: I tensorflow/core/platform/cpu_feature_guard.cc:142] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA
2020-02-07 18:27:24.626348: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2499720000 Hz
2020-02-07 18:27:24.627221: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x559453462be0 executing computations on platform Host. Devices:
2020-02-07 18:27:24.627626: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
2020-02-07 18:27:24.651344: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1618] Found device 0 with properties: 
name: Tesla K80 major: 3 minor: 7 memoryClockRate(GHz): 0.8235
pciBusID: 0000:05:00.0
2020-02-07 18:27:24.651394: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-02-07 18:27:24.651419: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-02-07 18:27:24.651441: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcufft.so.10.0
2020-02-07 18:27:24.651463: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcurand.so.10.0
2020-02-07 18:27:24.651486: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusolver.so.10.0
2020-02-07 18:27:24.651507: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcusparse.so.10.0
2020-02-07 18:27:24.651529: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-02-07 18:27:24.654486: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1746] Adding visible gpu devices: 0
2020-02-07 18:27:24.655193: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudart.so.10.0
2020-02-07 18:27:24.733958: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1159] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-02-07 18:27:24.734012: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1165]      0 
2020-02-07 18:27:24.734320: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1178] 0:   N 
2020-02-07 18:27:24.742602: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1304] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 11532 MB memory) -> physical GPU (device: 0, name: Tesla K80, pci bus id: 0000:05:00.0, compute capability: 3.7)
2020-02-07 18:27:24.764152: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x5594539dfbb0 executing computations on platform CUDA. Devices:
2020-02-07 18:27:24.764184: I tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Tesla K80, Compute Capability 3.7
2020-02-07 18:27:59.887253: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcublas.so.10.0
2020-02-07 18:28:00.245228: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library libcudnn.so.7
2020-02-07 18:28:01.997098: W tensorflow/core/common_runtime/bfc_allocator.cc:239] Allocator (GPU_0_bfc) ran out of memory trying to allocate 124.59MiB with freed_by_count=0. The caller indicates that this is not a failure, but may mean that there could be performance gains if more memory were available.
2020-02-07 18:28:12.016577: W tensorflow/core/common_runtime/bfc_allocator.cc:419] Allocator (GPU_0_bfc) ran out of memory trying to allocate 26.59MiB (rounded to 27878400).  Current allocation summary follows.
2020-02-07 18:28:12.016699: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (256): 	Total Chunks: 538, Chunks in use: 538. 134.5KiB allocated for chunks. 134.5KiB in use in bin. 133.8KiB client-requested in use in bin.
2020-02-07 18:28:12.016721: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (512): 	Total Chunks: 1, Chunks in use: 1. 768B allocated for chunks. 768B in use in bin. 768B client-requested in use in bin.
2020-02-07 18:28:12.016737: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (1024): 	Total Chunks: 1, Chunks in use: 1. 1.2KiB allocated for chunks. 1.2KiB in use in bin. 1.0KiB client-requested in use in bin.
2020-02-07 18:28:12.016752: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (2048): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016765: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (4096): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016780: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (8192): 	Total Chunks: 1, Chunks in use: 0. 10.2KiB allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016794: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (16384): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016807: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (32768): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016820: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (65536): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016835: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (131072): 	Total Chunks: 74, Chunks in use: 73. 10.39MiB allocated for chunks. 10.25MiB in use in bin. 10.25MiB client-requested in use in bin.
2020-02-07 18:28:12.016849: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (262144): 	Total Chunks: 1, Chunks in use: 0. 262.8KiB allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016863: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (524288): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016876: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (1048576): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016889: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (2097152): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016924: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (4194304): 	Total Chunks: 1, Chunks in use: 0. 6.65MiB allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016941: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (8388608): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.016956: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (16777216): 	Total Chunks: 409, Chunks in use: 409. 10.69GiB allocated for chunks. 10.69GiB in use in bin. 10.69GiB client-requested in use in bin.
2020-02-07 18:28:12.016972: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (33554432): 	Total Chunks: 10, Chunks in use: 10. 350.47MiB allocated for chunks. 350.47MiB in use in bin. 317.67MiB client-requested in use in bin.
2020-02-07 18:28:12.016986: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (67108864): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.017001: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (134217728): 	Total Chunks: 1, Chunks in use: 1. 218.51MiB allocated for chunks. 218.51MiB in use in bin. 218.51MiB client-requested in use in bin.
2020-02-07 18:28:12.017014: I tensorflow/core/common_runtime/bfc_allocator.cc:869] Bin (268435456): 	Total Chunks: 0, Chunks in use: 0. 0B allocated for chunks. 0B in use in bin. 0B client-requested in use in bin.
2020-02-07 18:28:12.017032: I tensorflow/core/common_runtime/bfc_allocator.cc:885] Bin for 26.59MiB was 16.00MiB, Chunk State: 
2020-02-07 18:28:12.017047: I tensorflow/core/common_runtime/bfc_allocator.cc:898] Next region of size 12092604416
2020-02-07 18:28:12.017063: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303b60000 next 1 of size 1280
2020-02-07 18:28:12.017076: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303b60500 next 2 of size 256
2020-02-07 18:28:12.017089: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303b60600 next 5 of size 256
2020-02-07 18:28:12.017101: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303b60700 next 4 of size 256
2020-02-07 18:28:12.017113: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303b60800 next 7 of size 256
2020-02-07 18:28:12.017125: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x2303b60900 next 3 of size 269056
2020-02-07 18:28:12.017138: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303ba2400 next 6 of size 134656
2020-02-07 18:28:12.017150: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3200 next 9 of size 256
2020-02-07 18:28:12.017162: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3300 next 11 of size 256
2020-02-07 18:28:12.017174: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3400 next 12 of size 256
2020-02-07 18:28:12.017186: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3500 next 14 of size 256
2020-02-07 18:28:12.017198: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3600 next 16 of size 256
2020-02-07 18:28:12.017210: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3700 next 17 of size 256
2020-02-07 18:28:12.017231: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3800 next 19 of size 256
2020-02-07 18:28:12.017248: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3900 next 20 of size 256
2020-02-07 18:28:12.017260: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3a00 next 21 of size 256
2020-02-07 18:28:12.017272: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3b00 next 23 of size 256
2020-02-07 18:28:12.017284: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3c00 next 24 of size 256
2020-02-07 18:28:12.017305: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3d00 next 25 of size 256
2020-02-07 18:28:12.017319: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3e00 next 27 of size 256
2020-02-07 18:28:12.017331: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc3f00 next 28 of size 256
2020-02-07 18:28:12.017343: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4000 next 29 of size 256
2020-02-07 18:28:12.017355: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4100 next 31 of size 256
2020-02-07 18:28:12.017367: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4200 next 32 of size 256
2020-02-07 18:28:12.017379: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4300 next 33 of size 256
2020-02-07 18:28:12.017391: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4400 next 35 of size 256
2020-02-07 18:28:12.017402: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4500 next 36 of size 256
2020-02-07 18:28:12.017414: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4600 next 37 of size 256
2020-02-07 18:28:12.017426: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4700 next 39 of size 256
2020-02-07 18:28:12.017438: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4800 next 40 of size 256
2020-02-07 18:28:12.017450: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4900 next 41 of size 256
2020-02-07 18:28:12.017462: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4a00 next 43 of size 256
2020-02-07 18:28:12.017473: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4b00 next 44 of size 256
2020-02-07 18:28:12.017485: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4c00 next 45 of size 256
2020-02-07 18:28:12.017497: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4d00 next 47 of size 256
2020-02-07 18:28:12.017509: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4e00 next 48 of size 256
2020-02-07 18:28:12.017521: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc4f00 next 49 of size 256
2020-02-07 18:28:12.017533: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5000 next 51 of size 256
2020-02-07 18:28:12.017544: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5100 next 52 of size 256
2020-02-07 18:28:12.017556: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5200 next 53 of size 256
2020-02-07 18:28:12.017568: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5300 next 55 of size 256
2020-02-07 18:28:12.017580: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5400 next 56 of size 256
2020-02-07 18:28:12.017592: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5500 next 57 of size 256
2020-02-07 18:28:12.017604: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5600 next 59 of size 256
2020-02-07 18:28:12.017615: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5700 next 60 of size 256
2020-02-07 18:28:12.017627: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5800 next 61 of size 256
2020-02-07 18:28:12.017639: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5900 next 63 of size 256
2020-02-07 18:28:12.017651: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5a00 next 64 of size 256
2020-02-07 18:28:12.017663: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5b00 next 65 of size 256
2020-02-07 18:28:12.017674: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5c00 next 67 of size 256
2020-02-07 18:28:12.017686: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5d00 next 68 of size 256
2020-02-07 18:28:12.017705: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5e00 next 69 of size 256
2020-02-07 18:28:12.017718: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc5f00 next 71 of size 256
2020-02-07 18:28:12.017730: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6000 next 72 of size 256
2020-02-07 18:28:12.017742: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6100 next 73 of size 256
2020-02-07 18:28:12.017754: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6200 next 75 of size 256
2020-02-07 18:28:12.017766: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6300 next 76 of size 256
2020-02-07 18:28:12.017778: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6400 next 77 of size 256
2020-02-07 18:28:12.017789: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6500 next 79 of size 256
2020-02-07 18:28:12.017801: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6600 next 80 of size 256
2020-02-07 18:28:12.017813: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6700 next 81 of size 256
2020-02-07 18:28:12.017825: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6800 next 83 of size 256
2020-02-07 18:28:12.017837: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6900 next 84 of size 256
2020-02-07 18:28:12.017849: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6a00 next 85 of size 256
2020-02-07 18:28:12.017860: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6b00 next 87 of size 256
2020-02-07 18:28:12.017872: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6c00 next 88 of size 256
2020-02-07 18:28:12.017884: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6d00 next 89 of size 256
2020-02-07 18:28:12.017896: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6e00 next 91 of size 256
2020-02-07 18:28:12.017908: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc6f00 next 92 of size 256
2020-02-07 18:28:12.017920: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7000 next 93 of size 256
2020-02-07 18:28:12.017932: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7100 next 95 of size 256
2020-02-07 18:28:12.017943: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7200 next 96 of size 256
2020-02-07 18:28:12.017955: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7300 next 97 of size 256
2020-02-07 18:28:12.017967: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7400 next 99 of size 256
2020-02-07 18:28:12.017979: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7500 next 100 of size 256
2020-02-07 18:28:12.017991: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7600 next 101 of size 256
2020-02-07 18:28:12.018003: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7700 next 103 of size 256
2020-02-07 18:28:12.018015: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7800 next 104 of size 256
2020-02-07 18:28:12.018027: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7900 next 105 of size 256
2020-02-07 18:28:12.018039: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7a00 next 107 of size 256
2020-02-07 18:28:12.018050: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7b00 next 108 of size 256
2020-02-07 18:28:12.018062: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7c00 next 109 of size 256
2020-02-07 18:28:12.018074: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7d00 next 111 of size 256
2020-02-07 18:28:12.018086: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7e00 next 112 of size 256
2020-02-07 18:28:12.018104: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc7f00 next 113 of size 256
2020-02-07 18:28:12.018117: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8000 next 115 of size 256
2020-02-07 18:28:12.018130: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8100 next 116 of size 256
2020-02-07 18:28:12.018141: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8200 next 117 of size 256
2020-02-07 18:28:12.018153: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8300 next 119 of size 256
2020-02-07 18:28:12.018165: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8400 next 120 of size 256
2020-02-07 18:28:12.018177: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8500 next 121 of size 256
2020-02-07 18:28:12.018189: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8600 next 123 of size 256
2020-02-07 18:28:12.018201: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8700 next 124 of size 256
2020-02-07 18:28:12.018213: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8800 next 125 of size 256
2020-02-07 18:28:12.018229: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8900 next 127 of size 256
2020-02-07 18:28:12.018243: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8a00 next 128 of size 256
2020-02-07 18:28:12.018255: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8b00 next 129 of size 256
2020-02-07 18:28:12.018267: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8c00 next 131 of size 256
2020-02-07 18:28:12.018279: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8d00 next 132 of size 256
2020-02-07 18:28:12.018291: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8e00 next 133 of size 256
2020-02-07 18:28:12.018303: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc8f00 next 135 of size 256
2020-02-07 18:28:12.018315: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9000 next 136 of size 256
2020-02-07 18:28:12.018327: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9100 next 137 of size 256
2020-02-07 18:28:12.018339: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9200 next 139 of size 256
2020-02-07 18:28:12.018351: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9300 next 140 of size 256
2020-02-07 18:28:12.018363: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9400 next 141 of size 256
2020-02-07 18:28:12.018375: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9500 next 143 of size 256
2020-02-07 18:28:12.018387: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9600 next 144 of size 256
2020-02-07 18:28:12.018399: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9700 next 145 of size 256
2020-02-07 18:28:12.018411: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9800 next 147 of size 256
2020-02-07 18:28:12.018422: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9900 next 148 of size 256
2020-02-07 18:28:12.018434: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9a00 next 149 of size 256
2020-02-07 18:28:12.018446: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9b00 next 151 of size 256
2020-02-07 18:28:12.018458: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9c00 next 152 of size 256
2020-02-07 18:28:12.018470: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9d00 next 153 of size 256
2020-02-07 18:28:12.018482: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9e00 next 155 of size 256
2020-02-07 18:28:12.018501: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bc9f00 next 156 of size 256
2020-02-07 18:28:12.018514: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bca000 next 157 of size 256
2020-02-07 18:28:12.018526: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bca100 next 159 of size 256
2020-02-07 18:28:12.018538: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bca200 next 160 of size 256
2020-02-07 18:28:12.018550: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bca300 next 161 of size 256
2020-02-07 18:28:12.018562: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bca400 next 163 of size 256
2020-02-07 18:28:12.018574: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bca500 next 164 of size 256
2020-02-07 18:28:12.018586: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bca600 next 165 of size 256
2020-02-07 18:28:12.018598: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bca700 next 167 of size 256
2020-02-07 18:28:12.018610: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bca800 next 168 of size 256
2020-02-07 18:28:12.018622: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bca900 next 169 of size 256
2020-02-07 18:28:12.018633: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcaa00 next 171 of size 256
2020-02-07 18:28:12.018645: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcab00 next 172 of size 256
2020-02-07 18:28:12.018657: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcac00 next 173 of size 256
2020-02-07 18:28:12.018669: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcad00 next 175 of size 256
2020-02-07 18:28:12.018681: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcae00 next 176 of size 256
2020-02-07 18:28:12.018693: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcaf00 next 177 of size 256
2020-02-07 18:28:12.018704: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcb000 next 179 of size 256
2020-02-07 18:28:12.018716: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcb100 next 180 of size 256
2020-02-07 18:28:12.018728: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcb200 next 181 of size 256
2020-02-07 18:28:12.018740: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcb300 next 183 of size 256
2020-02-07 18:28:12.018752: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcb400 next 184 of size 256
2020-02-07 18:28:12.018774: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcb500 next 185 of size 256
2020-02-07 18:28:12.018787: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcb600 next 187 of size 256
2020-02-07 18:28:12.018799: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcb700 next 188 of size 256
2020-02-07 18:28:12.018812: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcb800 next 189 of size 256
2020-02-07 18:28:12.018825: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcb900 next 191 of size 256
2020-02-07 18:28:12.018837: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcba00 next 192 of size 256
2020-02-07 18:28:12.018850: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcbb00 next 193 of size 256
2020-02-07 18:28:12.018872: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcbc00 next 195 of size 256
2020-02-07 18:28:12.018884: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcbd00 next 196 of size 256
2020-02-07 18:28:12.018896: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcbe00 next 197 of size 256
2020-02-07 18:28:12.018907: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcbf00 next 199 of size 256
2020-02-07 18:28:12.018926: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcc000 next 200 of size 256
2020-02-07 18:28:12.018939: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcc100 next 201 of size 256
2020-02-07 18:28:12.018951: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcc200 next 203 of size 256
2020-02-07 18:28:12.018963: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcc300 next 204 of size 256
2020-02-07 18:28:12.018975: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcc400 next 205 of size 256
2020-02-07 18:28:12.018987: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcc500 next 207 of size 256
2020-02-07 18:28:12.018999: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcc600 next 208 of size 256
2020-02-07 18:28:12.019011: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcc700 next 209 of size 256
2020-02-07 18:28:12.019023: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcc800 next 211 of size 256
2020-02-07 18:28:12.019035: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcc900 next 212 of size 256
2020-02-07 18:28:12.019047: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcca00 next 213 of size 256
2020-02-07 18:28:12.019059: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bccb00 next 215 of size 256
2020-02-07 18:28:12.019082: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bccc00 next 216 of size 256
2020-02-07 18:28:12.019095: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bccd00 next 217 of size 256
2020-02-07 18:28:12.019108: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcce00 next 219 of size 256
2020-02-07 18:28:12.019130: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bccf00 next 220 of size 256
2020-02-07 18:28:12.019142: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcd000 next 221 of size 256
2020-02-07 18:28:12.019154: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcd100 next 223 of size 256
2020-02-07 18:28:12.019166: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcd200 next 224 of size 256
2020-02-07 18:28:12.019178: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcd300 next 225 of size 256
2020-02-07 18:28:12.019201: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcd400 next 227 of size 256
2020-02-07 18:28:12.019214: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcd500 next 228 of size 256
2020-02-07 18:28:12.019238: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcd600 next 229 of size 256
2020-02-07 18:28:12.019252: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcd700 next 231 of size 256
2020-02-07 18:28:12.019265: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcd800 next 232 of size 256
2020-02-07 18:28:12.019287: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcd900 next 233 of size 256
2020-02-07 18:28:12.019299: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcda00 next 235 of size 256
2020-02-07 18:28:12.019549: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcdb00 next 236 of size 256
2020-02-07 18:28:12.019563: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcdc00 next 237 of size 256
2020-02-07 18:28:12.019576: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcdd00 next 239 of size 256
2020-02-07 18:28:12.019588: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcde00 next 240 of size 256
2020-02-07 18:28:12.019599: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcdf00 next 241 of size 256
2020-02-07 18:28:12.019621: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bce000 next 243 of size 256
2020-02-07 18:28:12.019635: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bce100 next 244 of size 256
2020-02-07 18:28:12.019659: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bce200 next 245 of size 256
2020-02-07 18:28:12.019672: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bce300 next 247 of size 256
2020-02-07 18:28:12.019684: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bce400 next 248 of size 256
2020-02-07 18:28:12.019697: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bce500 next 249 of size 256
2020-02-07 18:28:12.019723: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bce600 next 251 of size 256
2020-02-07 18:28:12.019734: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bce700 next 252 of size 256
2020-02-07 18:28:12.019746: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bce800 next 253 of size 256
2020-02-07 18:28:12.019758: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bce900 next 255 of size 256
2020-02-07 18:28:12.019782: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcea00 next 256 of size 256
2020-02-07 18:28:12.019794: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bceb00 next 257 of size 256
2020-02-07 18:28:12.019807: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcec00 next 259 of size 256
2020-02-07 18:28:12.019819: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bced00 next 260 of size 256
2020-02-07 18:28:12.019832: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcee00 next 261 of size 256
2020-02-07 18:28:12.019844: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcef00 next 263 of size 256
2020-02-07 18:28:12.019857: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcf000 next 264 of size 256
2020-02-07 18:28:12.019869: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcf100 next 265 of size 256
2020-02-07 18:28:12.019882: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcf200 next 267 of size 256
2020-02-07 18:28:12.019894: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcf300 next 268 of size 256
2020-02-07 18:28:12.019907: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcf400 next 269 of size 256
2020-02-07 18:28:12.019919: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcf500 next 271 of size 256
2020-02-07 18:28:12.019932: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcf600 next 272 of size 256
2020-02-07 18:28:12.019944: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcf700 next 273 of size 256
2020-02-07 18:28:12.019957: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcf800 next 275 of size 256
2020-02-07 18:28:12.019970: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcf900 next 276 of size 256
2020-02-07 18:28:12.019982: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcfa00 next 277 of size 256
2020-02-07 18:28:12.019995: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcfb00 next 279 of size 256
2020-02-07 18:28:12.020008: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcfc00 next 280 of size 256
2020-02-07 18:28:12.020020: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcfd00 next 281 of size 256
2020-02-07 18:28:12.020033: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcfe00 next 283 of size 256
2020-02-07 18:28:12.020045: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bcff00 next 284 of size 256
2020-02-07 18:28:12.020058: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0000 next 285 of size 256
2020-02-07 18:28:12.020087: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0100 next 287 of size 256
2020-02-07 18:28:12.020101: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0200 next 288 of size 256
2020-02-07 18:28:12.020113: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0300 next 289 of size 256
2020-02-07 18:28:12.020125: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0400 next 291 of size 256
2020-02-07 18:28:12.020138: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0500 next 292 of size 256
2020-02-07 18:28:12.020150: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0600 next 293 of size 256
2020-02-07 18:28:12.020162: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0700 next 295 of size 256
2020-02-07 18:28:12.020174: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0800 next 296 of size 256
2020-02-07 18:28:12.020186: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0900 next 297 of size 256
2020-02-07 18:28:12.020198: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0a00 next 298 of size 256
2020-02-07 18:28:12.020210: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0b00 next 300 of size 256
2020-02-07 18:28:12.020226: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0c00 next 302 of size 256
2020-02-07 18:28:12.020253: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0d00 next 303 of size 256
2020-02-07 18:28:12.020266: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0e00 next 308 of size 256
2020-02-07 18:28:12.020279: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd0f00 next 309 of size 256
2020-02-07 18:28:12.020292: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1000 next 310 of size 256
2020-02-07 18:28:12.020314: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1100 next 301 of size 256
2020-02-07 18:28:12.020327: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1200 next 299 of size 768
2020-02-07 18:28:12.020339: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1500 next 312 of size 256
2020-02-07 18:28:12.020351: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1600 next 316 of size 256
2020-02-07 18:28:12.020363: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1700 next 317 of size 256
2020-02-07 18:28:12.020375: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1800 next 318 of size 256
2020-02-07 18:28:12.020387: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1900 next 319 of size 256
2020-02-07 18:28:12.020399: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1a00 next 320 of size 256
2020-02-07 18:28:12.020411: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1b00 next 322 of size 256
2020-02-07 18:28:12.020423: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1c00 next 323 of size 256
2020-02-07 18:28:12.020435: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1d00 next 332 of size 256
2020-02-07 18:28:12.020447: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1e00 next 333 of size 256
2020-02-07 18:28:12.020459: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd1f00 next 334 of size 256
2020-02-07 18:28:12.020471: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2000 next 335 of size 256
2020-02-07 18:28:12.020483: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2100 next 336 of size 256
2020-02-07 18:28:12.020496: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2200 next 338 of size 256
2020-02-07 18:28:12.020516: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2300 next 339 of size 256
2020-02-07 18:28:12.020530: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2400 next 347 of size 256
2020-02-07 18:28:12.020543: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2500 next 349 of size 256
2020-02-07 18:28:12.020555: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2600 next 350 of size 256
2020-02-07 18:28:12.020568: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2700 next 351 of size 256
2020-02-07 18:28:12.020580: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2800 next 352 of size 256
2020-02-07 18:28:12.020592: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2900 next 354 of size 256
2020-02-07 18:28:12.020604: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2a00 next 355 of size 256
2020-02-07 18:28:12.020616: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2b00 next 365 of size 256
2020-02-07 18:28:12.020629: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2c00 next 367 of size 256
2020-02-07 18:28:12.020641: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2d00 next 368 of size 256
2020-02-07 18:28:12.020653: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2e00 next 369 of size 256
2020-02-07 18:28:12.020665: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd2f00 next 370 of size 256
2020-02-07 18:28:12.020677: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3000 next 372 of size 256
2020-02-07 18:28:12.020689: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3100 next 373 of size 256
2020-02-07 18:28:12.020702: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3200 next 381 of size 256
2020-02-07 18:28:12.020714: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3300 next 383 of size 256
2020-02-07 18:28:12.020726: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3400 next 384 of size 256
2020-02-07 18:28:12.020738: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3500 next 385 of size 256
2020-02-07 18:28:12.020750: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3600 next 386 of size 256
2020-02-07 18:28:12.020762: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3700 next 388 of size 256
2020-02-07 18:28:12.020774: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3800 next 389 of size 256
2020-02-07 18:28:12.020786: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3900 next 399 of size 256
2020-02-07 18:28:12.020798: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3a00 next 401 of size 256
2020-02-07 18:28:12.020810: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3b00 next 402 of size 256
2020-02-07 18:28:12.020823: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3c00 next 403 of size 256
2020-02-07 18:28:12.020835: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3d00 next 404 of size 256
2020-02-07 18:28:12.020847: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3e00 next 406 of size 256
2020-02-07 18:28:12.020859: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd3f00 next 407 of size 256
2020-02-07 18:28:12.020871: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4000 next 415 of size 256
2020-02-07 18:28:12.020883: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4100 next 417 of size 256
2020-02-07 18:28:12.020895: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4200 next 418 of size 256
2020-02-07 18:28:12.020907: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4300 next 419 of size 256
2020-02-07 18:28:12.020926: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4400 next 420 of size 256
2020-02-07 18:28:12.020940: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4500 next 422 of size 256
2020-02-07 18:28:12.020953: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4600 next 423 of size 256
2020-02-07 18:28:12.020965: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4700 next 433 of size 256
2020-02-07 18:28:12.020977: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4800 next 435 of size 256
2020-02-07 18:28:12.020989: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4900 next 436 of size 256
2020-02-07 18:28:12.021001: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4a00 next 437 of size 256
2020-02-07 18:28:12.021014: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4b00 next 438 of size 256
2020-02-07 18:28:12.021026: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4c00 next 440 of size 256
2020-02-07 18:28:12.021038: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4d00 next 441 of size 256
2020-02-07 18:28:12.021050: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4e00 next 449 of size 256
2020-02-07 18:28:12.021062: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd4f00 next 451 of size 256
2020-02-07 18:28:12.021075: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5000 next 452 of size 256
2020-02-07 18:28:12.021087: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5100 next 453 of size 256
2020-02-07 18:28:12.021099: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5200 next 454 of size 256
2020-02-07 18:28:12.021111: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5300 next 456 of size 256
2020-02-07 18:28:12.021123: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5400 next 457 of size 256
2020-02-07 18:28:12.021136: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5500 next 467 of size 256
2020-02-07 18:28:12.021148: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5600 next 469 of size 256
2020-02-07 18:28:12.021160: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5700 next 470 of size 256
2020-02-07 18:28:12.021173: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5800 next 471 of size 256
2020-02-07 18:28:12.021185: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5900 next 472 of size 256
2020-02-07 18:28:12.021197: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5a00 next 474 of size 256
2020-02-07 18:28:12.021209: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5b00 next 475 of size 256
2020-02-07 18:28:12.021225: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5c00 next 481 of size 256
2020-02-07 18:28:12.021251: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5d00 next 482 of size 256
2020-02-07 18:28:12.021264: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5e00 next 483 of size 256
2020-02-07 18:28:12.021277: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd5f00 next 484 of size 256
2020-02-07 18:28:12.021290: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6000 next 485 of size 256
2020-02-07 18:28:12.021312: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6100 next 487 of size 256
2020-02-07 18:28:12.021324: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6200 next 488 of size 256
2020-02-07 18:28:12.021336: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6300 next 496 of size 256
2020-02-07 18:28:12.021356: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6400 next 497 of size 256
2020-02-07 18:28:12.021369: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6500 next 498 of size 256
2020-02-07 18:28:12.021382: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6600 next 499 of size 256
2020-02-07 18:28:12.021394: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6700 next 500 of size 256
2020-02-07 18:28:12.021406: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6800 next 502 of size 256
2020-02-07 18:28:12.021418: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6900 next 503 of size 256
2020-02-07 18:28:12.021430: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6a00 next 511 of size 256
2020-02-07 18:28:12.021441: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6b00 next 513 of size 256
2020-02-07 18:28:12.021453: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6c00 next 514 of size 256
2020-02-07 18:28:12.021465: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6d00 next 515 of size 256
2020-02-07 18:28:12.021477: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6e00 next 516 of size 256
2020-02-07 18:28:12.021489: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd6f00 next 518 of size 256
2020-02-07 18:28:12.021501: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7000 next 519 of size 256
2020-02-07 18:28:12.021513: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7100 next 529 of size 256
2020-02-07 18:28:12.021525: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7200 next 531 of size 256
2020-02-07 18:28:12.021536: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7300 next 532 of size 256
2020-02-07 18:28:12.021548: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7400 next 533 of size 256
2020-02-07 18:28:12.021560: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7500 next 534 of size 256
2020-02-07 18:28:12.021572: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7600 next 536 of size 256
2020-02-07 18:28:12.021584: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7700 next 537 of size 256
2020-02-07 18:28:12.021596: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7800 next 545 of size 256
2020-02-07 18:28:12.021608: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7900 next 547 of size 256
2020-02-07 18:28:12.021620: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7a00 next 548 of size 256
2020-02-07 18:28:12.021631: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7b00 next 549 of size 256
2020-02-07 18:28:12.021643: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7c00 next 550 of size 256
2020-02-07 18:28:12.021655: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7d00 next 552 of size 256
2020-02-07 18:28:12.021667: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7e00 next 553 of size 256
2020-02-07 18:28:12.021679: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd7f00 next 563 of size 256
2020-02-07 18:28:12.021691: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8000 next 565 of size 256
2020-02-07 18:28:12.021703: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8100 next 566 of size 256
2020-02-07 18:28:12.021715: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8200 next 567 of size 256
2020-02-07 18:28:12.021726: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8300 next 568 of size 256
2020-02-07 18:28:12.021738: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8400 next 570 of size 256
2020-02-07 18:28:12.021756: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8500 next 571 of size 256
2020-02-07 18:28:12.021770: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8600 next 579 of size 256
2020-02-07 18:28:12.021782: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8700 next 581 of size 256
2020-02-07 18:28:12.021794: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8800 next 582 of size 256
2020-02-07 18:28:12.021806: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8900 next 583 of size 256
2020-02-07 18:28:12.021818: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8a00 next 584 of size 256
2020-02-07 18:28:12.021830: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8b00 next 586 of size 256
2020-02-07 18:28:12.021842: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8c00 next 587 of size 256
2020-02-07 18:28:12.021854: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8d00 next 597 of size 256
2020-02-07 18:28:12.021866: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8e00 next 599 of size 256
2020-02-07 18:28:12.021877: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd8f00 next 600 of size 256
2020-02-07 18:28:12.021889: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9000 next 601 of size 256
2020-02-07 18:28:12.021901: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9100 next 602 of size 256
2020-02-07 18:28:12.021913: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9200 next 604 of size 256
2020-02-07 18:28:12.021924: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9300 next 605 of size 256
2020-02-07 18:28:12.021936: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9400 next 613 of size 256
2020-02-07 18:28:12.021948: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9500 next 615 of size 256
2020-02-07 18:28:12.021960: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9600 next 616 of size 256
2020-02-07 18:28:12.021972: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9700 next 617 of size 256
2020-02-07 18:28:12.021983: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9800 next 618 of size 256
2020-02-07 18:28:12.021995: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9900 next 620 of size 256
2020-02-07 18:28:12.022007: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9a00 next 621 of size 256
2020-02-07 18:28:12.022019: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9b00 next 631 of size 256
2020-02-07 18:28:12.022031: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9c00 next 633 of size 256
2020-02-07 18:28:12.022042: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9d00 next 634 of size 256
2020-02-07 18:28:12.022054: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9e00 next 635 of size 256
2020-02-07 18:28:12.022066: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bd9f00 next 636 of size 256
2020-02-07 18:28:12.022078: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bda000 next 638 of size 256
2020-02-07 18:28:12.022090: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bda100 next 639 of size 256
2020-02-07 18:28:12.022102: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bda200 next 645 of size 256
2020-02-07 18:28:12.022113: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bda300 next 646 of size 256
2020-02-07 18:28:12.022125: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bda400 next 647 of size 256
2020-02-07 18:28:12.022143: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bda500 next 648 of size 256
2020-02-07 18:28:12.022157: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bda600 next 649 of size 256
2020-02-07 18:28:12.022169: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bda700 next 651 of size 256
2020-02-07 18:28:12.022181: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bda800 next 652 of size 256
2020-02-07 18:28:12.022193: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bda900 next 660 of size 256
2020-02-07 18:28:12.022205: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdaa00 next 661 of size 256
2020-02-07 18:28:12.022217: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdab00 next 662 of size 256
2020-02-07 18:28:12.022245: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdac00 next 663 of size 256
2020-02-07 18:28:12.022259: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdad00 next 664 of size 256
2020-02-07 18:28:12.022272: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdae00 next 666 of size 256
2020-02-07 18:28:12.022284: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdaf00 next 667 of size 256
2020-02-07 18:28:12.022307: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdb000 next 675 of size 256
2020-02-07 18:28:12.022319: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdb100 next 677 of size 256
2020-02-07 18:28:12.022331: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdb200 next 678 of size 256
2020-02-07 18:28:12.022343: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdb300 next 679 of size 256
2020-02-07 18:28:12.022355: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdb400 next 680 of size 256
2020-02-07 18:28:12.022367: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdb500 next 682 of size 256
2020-02-07 18:28:12.022379: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdb600 next 683 of size 256
2020-02-07 18:28:12.022391: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdb700 next 693 of size 256
2020-02-07 18:28:12.022403: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdb800 next 695 of size 256
2020-02-07 18:28:12.022416: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdb900 next 696 of size 256
2020-02-07 18:28:12.022427: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdba00 next 697 of size 256
2020-02-07 18:28:12.022439: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdbb00 next 698 of size 256
2020-02-07 18:28:12.022451: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdbc00 next 700 of size 256
2020-02-07 18:28:12.022464: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdbd00 next 701 of size 256
2020-02-07 18:28:12.022475: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdbe00 next 709 of size 256
2020-02-07 18:28:12.022488: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdbf00 next 711 of size 256
2020-02-07 18:28:12.022500: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdc000 next 712 of size 256
2020-02-07 18:28:12.022512: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdc100 next 713 of size 256
2020-02-07 18:28:12.022524: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdc200 next 714 of size 256
2020-02-07 18:28:12.022536: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdc300 next 716 of size 256
2020-02-07 18:28:12.022548: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdc400 next 717 of size 256
2020-02-07 18:28:12.022560: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdc500 next 727 of size 256
2020-02-07 18:28:12.022581: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdc600 next 729 of size 256
2020-02-07 18:28:12.022595: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdc700 next 730 of size 256
2020-02-07 18:28:12.022608: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdc800 next 731 of size 256
2020-02-07 18:28:12.022620: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdc900 next 732 of size 256
2020-02-07 18:28:12.022632: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdca00 next 734 of size 256
2020-02-07 18:28:12.022645: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdcb00 next 735 of size 256
2020-02-07 18:28:12.022657: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdcc00 next 743 of size 256
2020-02-07 18:28:12.022669: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdcd00 next 745 of size 256
2020-02-07 18:28:12.022681: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdce00 next 746 of size 256
2020-02-07 18:28:12.022693: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdcf00 next 747 of size 256
2020-02-07 18:28:12.022706: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdd000 next 748 of size 256
2020-02-07 18:28:12.022718: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdd100 next 750 of size 256
2020-02-07 18:28:12.022730: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdd200 next 751 of size 256
2020-02-07 18:28:12.022743: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdd300 next 761 of size 256
2020-02-07 18:28:12.022755: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdd400 next 763 of size 256
2020-02-07 18:28:12.022767: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdd500 next 764 of size 256
2020-02-07 18:28:12.022779: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdd600 next 765 of size 256
2020-02-07 18:28:12.022791: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdd700 next 766 of size 256
2020-02-07 18:28:12.022804: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdd800 next 768 of size 256
2020-02-07 18:28:12.022816: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdd900 next 769 of size 256
2020-02-07 18:28:12.022828: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdda00 next 777 of size 256
2020-02-07 18:28:12.022840: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bddb00 next 779 of size 256
2020-02-07 18:28:12.022853: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bddc00 next 780 of size 256
2020-02-07 18:28:12.022865: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bddd00 next 781 of size 256
2020-02-07 18:28:12.022877: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdde00 next 782 of size 256
2020-02-07 18:28:12.022889: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bddf00 next 784 of size 256
2020-02-07 18:28:12.022901: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bde000 next 785 of size 256
2020-02-07 18:28:12.022914: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bde100 next 795 of size 256
2020-02-07 18:28:12.022927: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bde200 next 797 of size 256
2020-02-07 18:28:12.022939: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bde300 next 798 of size 256
2020-02-07 18:28:12.022952: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bde400 next 799 of size 256
2020-02-07 18:28:12.022964: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bde500 next 800 of size 256
2020-02-07 18:28:12.022982: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bde600 next 802 of size 256
2020-02-07 18:28:12.022996: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bde700 next 803 of size 256
2020-02-07 18:28:12.023009: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bde800 next 809 of size 256
2020-02-07 18:28:12.023021: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bde900 next 810 of size 256
2020-02-07 18:28:12.023033: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdea00 next 811 of size 256
2020-02-07 18:28:12.023044: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdeb00 next 812 of size 256
2020-02-07 18:28:12.023056: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdec00 next 813 of size 256
2020-02-07 18:28:12.023068: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bded00 next 815 of size 256
2020-02-07 18:28:12.023091: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdee00 next 816 of size 256
2020-02-07 18:28:12.023115: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdef00 next 824 of size 256
2020-02-07 18:28:12.023138: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdf000 next 825 of size 256
2020-02-07 18:28:12.023168: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdf100 next 826 of size 256
2020-02-07 18:28:12.023181: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdf200 next 827 of size 256
2020-02-07 18:28:12.023204: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdf300 next 828 of size 256
2020-02-07 18:28:12.023217: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdf400 next 830 of size 256
2020-02-07 18:28:12.023244: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdf500 next 831 of size 256
2020-02-07 18:28:12.023257: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdf600 next 839 of size 256
2020-02-07 18:28:12.023269: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdf700 next 841 of size 256
2020-02-07 18:28:12.023281: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdf800 next 842 of size 256
2020-02-07 18:28:12.023293: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdf900 next 843 of size 256
2020-02-07 18:28:12.023305: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdfa00 next 844 of size 256
2020-02-07 18:28:12.023317: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdfb00 next 846 of size 256
2020-02-07 18:28:12.023329: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdfc00 next 847 of size 256
2020-02-07 18:28:12.023341: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdfd00 next 857 of size 256
2020-02-07 18:28:12.023353: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdfe00 next 859 of size 256
2020-02-07 18:28:12.023365: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303bdff00 next 860 of size 256
2020-02-07 18:28:12.023377: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0000 next 861 of size 256
2020-02-07 18:28:12.023389: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0100 next 862 of size 256
2020-02-07 18:28:12.023401: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0200 next 864 of size 256
2020-02-07 18:28:12.023413: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0300 next 865 of size 256
2020-02-07 18:28:12.023425: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0400 next 873 of size 256
2020-02-07 18:28:12.023437: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0500 next 875 of size 256
2020-02-07 18:28:12.023449: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0600 next 876 of size 256
2020-02-07 18:28:12.023467: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0700 next 877 of size 256
2020-02-07 18:28:12.023481: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0800 next 878 of size 256
2020-02-07 18:28:12.023493: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0900 next 880 of size 256
2020-02-07 18:28:12.023506: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0a00 next 881 of size 256
2020-02-07 18:28:12.023518: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0b00 next 891 of size 256
2020-02-07 18:28:12.023530: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0c00 next 893 of size 256
2020-02-07 18:28:12.023542: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0d00 next 894 of size 256
2020-02-07 18:28:12.023554: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0e00 next 895 of size 256
2020-02-07 18:28:12.023566: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be0f00 next 896 of size 256
2020-02-07 18:28:12.023578: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1000 next 898 of size 256
2020-02-07 18:28:12.023590: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1100 next 899 of size 256
2020-02-07 18:28:12.023602: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1200 next 907 of size 256
2020-02-07 18:28:12.023614: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1300 next 909 of size 256
2020-02-07 18:28:12.023626: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1400 next 910 of size 256
2020-02-07 18:28:12.023638: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1500 next 911 of size 256
2020-02-07 18:28:12.023650: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1600 next 912 of size 256
2020-02-07 18:28:12.023662: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1700 next 914 of size 256
2020-02-07 18:28:12.023674: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1800 next 915 of size 256
2020-02-07 18:28:12.023686: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1900 next 925 of size 256
2020-02-07 18:28:12.023698: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1a00 next 927 of size 256
2020-02-07 18:28:12.023710: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1b00 next 928 of size 256
2020-02-07 18:28:12.023722: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1c00 next 929 of size 256
2020-02-07 18:28:12.023734: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1d00 next 930 of size 256
2020-02-07 18:28:12.023746: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1e00 next 932 of size 256
2020-02-07 18:28:12.023758: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be1f00 next 933 of size 256
2020-02-07 18:28:12.023770: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2000 next 941 of size 256
2020-02-07 18:28:12.023782: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2100 next 943 of size 256
2020-02-07 18:28:12.023793: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2200 next 944 of size 256
2020-02-07 18:28:12.023805: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2300 next 945 of size 256
2020-02-07 18:28:12.023817: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2400 next 946 of size 256
2020-02-07 18:28:12.023829: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2500 next 948 of size 256
2020-02-07 18:28:12.023841: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2600 next 949 of size 256
2020-02-07 18:28:12.023859: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2700 next 959 of size 256
2020-02-07 18:28:12.023873: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2800 next 961 of size 256
2020-02-07 18:28:12.023885: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2900 next 962 of size 256
2020-02-07 18:28:12.023897: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2a00 next 963 of size 256
2020-02-07 18:28:12.023909: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2b00 next 964 of size 256
2020-02-07 18:28:12.023933: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2c00 next 966 of size 256
2020-02-07 18:28:12.023945: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2d00 next 967 of size 256
2020-02-07 18:28:12.023958: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2e00 next 973 of size 256
2020-02-07 18:28:12.023981: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be2f00 next 974 of size 256
2020-02-07 18:28:12.023993: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3000 next 975 of size 256
2020-02-07 18:28:12.024005: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3100 next 976 of size 256
2020-02-07 18:28:12.024017: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3200 next 977 of size 256
2020-02-07 18:28:12.024029: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3300 next 979 of size 256
2020-02-07 18:28:12.024041: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3400 next 980 of size 256
2020-02-07 18:28:12.024053: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3500 next 988 of size 256
2020-02-07 18:28:12.024065: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3600 next 989 of size 256
2020-02-07 18:28:12.024077: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3700 next 990 of size 256
2020-02-07 18:28:12.024090: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3800 next 991 of size 256
2020-02-07 18:28:12.024102: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3900 next 992 of size 256
2020-02-07 18:28:12.024114: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3a00 next 994 of size 256
2020-02-07 18:28:12.024126: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3b00 next 995 of size 256
2020-02-07 18:28:12.024138: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3c00 next 1003 of size 256
2020-02-07 18:28:12.024150: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3d00 next 1005 of size 256
2020-02-07 18:28:12.024162: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3e00 next 1006 of size 256
2020-02-07 18:28:12.024174: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be3f00 next 1007 of size 256
2020-02-07 18:28:12.024186: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be4000 next 1008 of size 256
2020-02-07 18:28:12.024198: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be4100 next 1010 of size 256
2020-02-07 18:28:12.024210: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be4200 next 1011 of size 256
2020-02-07 18:28:12.024249: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be4300 next 1021 of size 256
2020-02-07 18:28:12.024284: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be4400 next 1023 of size 256
2020-02-07 18:28:12.024297: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be4500 next 1024 of size 256
2020-02-07 18:28:12.024333: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be4600 next 1025 of size 256
2020-02-07 18:28:12.024355: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be4700 next 1026 of size 256
2020-02-07 18:28:12.024397: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be4800 next 1028 of size 256
2020-02-07 18:28:12.024412: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be4900 next 1029 of size 256
2020-02-07 18:28:12.024435: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be4a00 next 1036 of size 256
2020-02-07 18:28:12.024458: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x2303be4b00 next 10 of size 10496
2020-02-07 18:28:12.024470: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303be7400 next 8 of size 147456
2020-02-07 18:28:12.024482: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x2303c0b400 next 15 of size 147456
2020-02-07 18:28:12.024495: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303c2f400 next 13 of size 147456
2020-02-07 18:28:12.024507: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303c53400 next 18 of size 147456
2020-02-07 18:28:12.024519: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303c77400 next 22 of size 147456
2020-02-07 18:28:12.024531: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303c9b400 next 26 of size 147456
2020-02-07 18:28:12.024543: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303cbf400 next 30 of size 147456
2020-02-07 18:28:12.024555: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303ce3400 next 34 of size 147456
2020-02-07 18:28:12.024567: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303d07400 next 38 of size 147456
2020-02-07 18:28:12.024579: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303d2b400 next 42 of size 147456
2020-02-07 18:28:12.024591: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303d4f400 next 46 of size 147456
2020-02-07 18:28:12.024603: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303d73400 next 50 of size 147456
2020-02-07 18:28:12.024615: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303d97400 next 54 of size 147456
2020-02-07 18:28:12.024627: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303dbb400 next 58 of size 147456
2020-02-07 18:28:12.024639: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303ddf400 next 62 of size 147456
2020-02-07 18:28:12.024651: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303e03400 next 66 of size 147456
2020-02-07 18:28:12.024663: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303e27400 next 70 of size 147456
2020-02-07 18:28:12.024675: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303e4b400 next 74 of size 147456
2020-02-07 18:28:12.024687: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303e6f400 next 78 of size 147456
2020-02-07 18:28:12.024699: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303e93400 next 82 of size 147456
2020-02-07 18:28:12.024711: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303eb7400 next 86 of size 147456
2020-02-07 18:28:12.024723: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303edb400 next 90 of size 147456
2020-02-07 18:28:12.024735: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303eff400 next 94 of size 147456
2020-02-07 18:28:12.024747: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303f23400 next 98 of size 147456
2020-02-07 18:28:12.024759: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303f47400 next 102 of size 147456
2020-02-07 18:28:12.024771: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303f6b400 next 106 of size 147456
2020-02-07 18:28:12.024783: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303f8f400 next 110 of size 147456
2020-02-07 18:28:12.024795: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303fb3400 next 114 of size 147456
2020-02-07 18:28:12.024813: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303fd7400 next 118 of size 147456
2020-02-07 18:28:12.024828: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2303ffb400 next 122 of size 147456
2020-02-07 18:28:12.024841: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230401f400 next 126 of size 147456
2020-02-07 18:28:12.024853: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304043400 next 130 of size 147456
2020-02-07 18:28:12.024865: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304067400 next 134 of size 147456
2020-02-07 18:28:12.024877: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230408b400 next 138 of size 147456
2020-02-07 18:28:12.024889: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23040af400 next 142 of size 147456
2020-02-07 18:28:12.024901: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23040d3400 next 146 of size 147456
2020-02-07 18:28:12.024913: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23040f7400 next 150 of size 147456
2020-02-07 18:28:12.024925: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230411b400 next 154 of size 147456
2020-02-07 18:28:12.024937: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230413f400 next 158 of size 147456
2020-02-07 18:28:12.024949: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304163400 next 162 of size 147456
2020-02-07 18:28:12.024961: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304187400 next 166 of size 147456
2020-02-07 18:28:12.024973: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23041ab400 next 170 of size 147456
2020-02-07 18:28:12.024985: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23041cf400 next 174 of size 147456
2020-02-07 18:28:12.024997: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23041f3400 next 178 of size 147456
2020-02-07 18:28:12.025009: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304217400 next 182 of size 147456
2020-02-07 18:28:12.025021: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230423b400 next 186 of size 147456
2020-02-07 18:28:12.025033: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230425f400 next 190 of size 147456
2020-02-07 18:28:12.025045: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304283400 next 194 of size 147456
2020-02-07 18:28:12.025066: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23042a7400 next 198 of size 147456
2020-02-07 18:28:12.025078: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23042cb400 next 202 of size 147456
2020-02-07 18:28:12.025091: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23042ef400 next 206 of size 147456
2020-02-07 18:28:12.025103: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304313400 next 210 of size 147456
2020-02-07 18:28:12.025115: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304337400 next 214 of size 147456
2020-02-07 18:28:12.025128: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230435b400 next 218 of size 147456
2020-02-07 18:28:12.025140: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230437f400 next 222 of size 147456
2020-02-07 18:28:12.025152: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23043a3400 next 226 of size 147456
2020-02-07 18:28:12.025164: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23043c7400 next 230 of size 147456
2020-02-07 18:28:12.025177: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23043eb400 next 234 of size 147456
2020-02-07 18:28:12.025189: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230440f400 next 238 of size 147456
2020-02-07 18:28:12.025207: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304433400 next 242 of size 147456
2020-02-07 18:28:12.025225: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304457400 next 246 of size 147456
2020-02-07 18:28:12.025250: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230447b400 next 250 of size 147456
2020-02-07 18:28:12.025263: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230449f400 next 254 of size 147456
2020-02-07 18:28:12.025276: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23044c3400 next 258 of size 147456
2020-02-07 18:28:12.025289: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23044e7400 next 262 of size 147456
2020-02-07 18:28:12.025311: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230450b400 next 266 of size 147456
2020-02-07 18:28:12.025324: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230452f400 next 270 of size 147456
2020-02-07 18:28:12.025336: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304553400 next 274 of size 147456
2020-02-07 18:28:12.025349: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304577400 next 278 of size 147456
2020-02-07 18:28:12.025361: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230459b400 next 282 of size 147456
2020-02-07 18:28:12.025373: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23045bf400 next 286 of size 147456
2020-02-07 18:28:12.025386: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23045e3400 next 290 of size 147456
2020-02-07 18:28:12.025398: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2304607400 next 294 of size 147456
2020-02-07 18:28:12.025410: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230462b400 next 306 of size 27878400
2020-02-07 18:28:12.025423: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23060c1800 next 307 of size 27878400
2020-02-07 18:28:12.025435: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2307b57c00 next 311 of size 27878400
2020-02-07 18:28:12.025448: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23095ee000 next 304 of size 30927616
2020-02-07 18:28:12.025461: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x230b36cb00 next 305 of size 229125632
2020-02-07 18:28:12.025473: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2318def900 next 313 of size 27878400
2020-02-07 18:28:12.025486: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x231a885d00 next 314 of size 27878400
2020-02-07 18:28:12.025498: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x231c31c100 next 315 of size 27878400
2020-02-07 18:28:12.025510: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x231ddb2500 next 321 of size 27878400
2020-02-07 18:28:12.025523: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x231f848900 next 324 of size 27878400
2020-02-07 18:28:12.025535: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23212ded00 next 325 of size 27878400
2020-02-07 18:28:12.025547: I tensorflow/core/common_runtime/bfc_allocator.cc:905] Free  at 0x2322d75100 next 328 of size 6969600
2020-02-07 18:28:12.025560: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x232341aa00 next 327 of size 48787200
2020-02-07 18:28:12.025573: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23262a1900 next 329 of size 27878400
2020-02-07 18:28:12.025585: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2327d37d00 next 326 of size 27878400
2020-02-07 18:28:12.025598: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23297ce100 next 330 of size 27878400
2020-02-07 18:28:12.025610: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x232b264500 next 331 of size 27878400
2020-02-07 18:28:12.025622: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x232ccfa900 next 337 of size 27878400
2020-02-07 18:28:12.025642: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x232e790d00 next 340 of size 27878400
2020-02-07 18:28:12.025656: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2330227100 next 341 of size 27878400
2020-02-07 18:28:12.025669: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2331cbd500 next 342 of size 27878400
2020-02-07 18:28:12.025682: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2333753900 next 343 of size 28558336
2020-02-07 18:28:12.025694: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x233528fd00 next 344 of size 27878400
2020-02-07 18:28:12.025706: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2336d26100 next 345 of size 27878400
2020-02-07 18:28:12.025719: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23387bc500 next 346 of size 27878400
2020-02-07 18:28:12.025731: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x233a252900 next 348 of size 27878400
2020-02-07 18:28:12.025743: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x233bce8d00 next 353 of size 27878400
2020-02-07 18:28:12.025756: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x233d77f100 next 356 of size 27878400
2020-02-07 18:28:12.025768: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x233f215500 next 357 of size 27878400
2020-02-07 18:28:12.025780: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2340cab900 next 360 of size 27878400
2020-02-07 18:28:12.025793: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2342741d00 next 359 of size 27878400
2020-02-07 18:28:12.025805: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23441d8100 next 361 of size 27878400
2020-02-07 18:28:12.025817: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2345c6e500 next 358 of size 27878400
2020-02-07 18:28:12.025830: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2347704900 next 362 of size 28558336
2020-02-07 18:28:12.025842: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2349240d00 next 363 of size 27878400
2020-02-07 18:28:12.025854: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x234acd7100 next 364 of size 27878400
2020-02-07 18:28:12.025867: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x234c76d500 next 366 of size 27878400
2020-02-07 18:28:12.025879: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x234e203900 next 371 of size 27878400
2020-02-07 18:28:12.025891: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x234fc99d00 next 374 of size 27878400
2020-02-07 18:28:12.025904: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2351730100 next 375 of size 27878400
2020-02-07 18:28:12.025916: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23531c6500 next 376 of size 27878400
2020-02-07 18:28:12.025928: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2354c5c900 next 377 of size 29593600
2020-02-07 18:28:12.025941: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2356895900 next 378 of size 28217344
2020-02-07 18:28:12.025954: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x235837e900 next 379 of size 27878400
2020-02-07 18:28:12.025966: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2359e14d00 next 380 of size 27878400
2020-02-07 18:28:12.025979: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x235b8ab100 next 382 of size 27878400
2020-02-07 18:28:12.025991: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x235d341500 next 387 of size 27878400
2020-02-07 18:28:12.026003: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x235edd7900 next 390 of size 27878400
2020-02-07 18:28:12.026015: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x236086dd00 next 391 of size 27878400
2020-02-07 18:28:12.026034: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2362304100 next 394 of size 27878400
2020-02-07 18:28:12.026048: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2363d9a500 next 393 of size 27878400
2020-02-07 18:28:12.026060: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2365830900 next 395 of size 27878400
2020-02-07 18:28:12.026073: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23672c6d00 next 392 of size 27878400
2020-02-07 18:28:12.026085: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2368d5d100 next 396 of size 29593600
2020-02-07 18:28:12.026097: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x236a996100 next 397 of size 28217344
2020-02-07 18:28:12.026118: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x236c47f100 next 398 of size 27878400
2020-02-07 18:28:12.026130: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x236df15500 next 400 of size 27878400
2020-02-07 18:28:12.026142: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x236f9ab900 next 405 of size 27878400
2020-02-07 18:28:12.026154: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2371441d00 next 408 of size 27878400
2020-02-07 18:28:12.026166: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2372ed8100 next 409 of size 27878400
2020-02-07 18:28:12.026178: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x237496e500 next 410 of size 27878400
2020-02-07 18:28:12.026190: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2376404900 next 411 of size 31719424
2020-02-07 18:28:12.026202: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2378244900 next 412 of size 28901376
2020-02-07 18:28:12.026215: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2379dd4900 next 413 of size 27878400
2020-02-07 18:28:12.026243: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x237b86ad00 next 414 of size 27878400
2020-02-07 18:28:12.026257: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x237d301100 next 416 of size 27878400
2020-02-07 18:28:12.026269: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x237ed97500 next 421 of size 27878400
2020-02-07 18:28:12.026282: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x238082d900 next 424 of size 27878400
2020-02-07 18:28:12.026304: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23822c3d00 next 425 of size 27878400
2020-02-07 18:28:12.026316: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2383d5a100 next 428 of size 27878400
2020-02-07 18:28:12.026328: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23857f0500 next 427 of size 27878400
2020-02-07 18:28:12.026340: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2387286900 next 429 of size 27878400
2020-02-07 18:28:12.026352: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2388d1cd00 next 426 of size 27878400
2020-02-07 18:28:12.026364: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x238a7b3100 next 430 of size 31719424
2020-02-07 18:28:12.026375: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x238c5f3100 next 431 of size 28901376
2020-02-07 18:28:12.026387: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x238e183100 next 432 of size 27878400
2020-02-07 18:28:12.026399: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x238fc19500 next 434 of size 27878400
2020-02-07 18:28:12.026411: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23916af900 next 439 of size 27878400
2020-02-07 18:28:12.026423: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2393145d00 next 442 of size 27878400
2020-02-07 18:28:12.026434: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2394bdc100 next 443 of size 27878400
2020-02-07 18:28:12.026453: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2396672500 next 444 of size 27878400
2020-02-07 18:28:12.026467: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2398108900 next 445 of size 34668544
2020-02-07 18:28:12.026480: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x239a218900 next 446 of size 28901376
2020-02-07 18:28:12.026491: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x239bda8900 next 447 of size 27878400
2020-02-07 18:28:12.026503: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x239d83ed00 next 448 of size 27878400
2020-02-07 18:28:12.026515: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x239f2d5100 next 450 of size 27878400
2020-02-07 18:28:12.026527: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23a0d6b500 next 455 of size 27878400
2020-02-07 18:28:12.026539: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23a2801900 next 458 of size 27878400
2020-02-07 18:28:12.026550: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23a4297d00 next 459 of size 27878400
2020-02-07 18:28:12.026562: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23a5d2e100 next 462 of size 27878400
2020-02-07 18:28:12.026574: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23a77c4500 next 461 of size 27878400
2020-02-07 18:28:12.026586: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23a925a900 next 463 of size 27878400
2020-02-07 18:28:12.026598: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23aacf0d00 next 460 of size 27878400
2020-02-07 18:28:12.026609: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23ac787100 next 464 of size 34668544
2020-02-07 18:28:12.026621: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23ae897100 next 465 of size 28901376
2020-02-07 18:28:12.026633: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23b0427100 next 466 of size 27878400
2020-02-07 18:28:12.026645: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23b1ebd500 next 468 of size 27878400
2020-02-07 18:28:12.026657: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23b3953900 next 473 of size 27878400
2020-02-07 18:28:12.026668: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23b53e9d00 next 476 of size 27878400
2020-02-07 18:28:12.026680: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23b6e80100 next 477 of size 27878400
2020-02-07 18:28:12.026692: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23b8916500 next 478 of size 27878400
2020-02-07 18:28:12.026704: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23ba3ac900 next 479 of size 27878400
2020-02-07 18:28:12.026716: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23bbe42d00 next 480 of size 27878400
2020-02-07 18:28:12.026728: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23bd8d9100 next 486 of size 27878400
2020-02-07 18:28:12.026739: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23bf36f500 next 489 of size 27878400
2020-02-07 18:28:12.026751: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23c0e05900 next 490 of size 27878400
2020-02-07 18:28:12.026763: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23c289bd00 next 493 of size 27878400
2020-02-07 18:28:12.026775: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23c4332100 next 492 of size 27878400
2020-02-07 18:28:12.026787: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23c5dc8500 next 494 of size 27878400
2020-02-07 18:28:12.026799: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23c785e900 next 491 of size 27878400
2020-02-07 18:28:12.026810: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23c92f4d00 next 495 of size 27878400
2020-02-07 18:28:12.026822: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23cad8b100 next 501 of size 27878400
2020-02-07 18:28:12.026840: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23cc821500 next 504 of size 27878400
2020-02-07 18:28:12.026854: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23ce2b7900 next 505 of size 27878400
2020-02-07 18:28:12.026866: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23cfd4dd00 next 506 of size 27878400
2020-02-07 18:28:12.026878: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23d17e4100 next 507 of size 28558336
2020-02-07 18:28:12.026890: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23d3320500 next 508 of size 27878400
2020-02-07 18:28:12.026902: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23d4db6900 next 509 of size 27878400
2020-02-07 18:28:12.026914: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23d684cd00 next 510 of size 27878400
2020-02-07 18:28:12.026925: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23d82e3100 next 512 of size 27878400
2020-02-07 18:28:12.026937: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23d9d79500 next 517 of size 27878400
2020-02-07 18:28:12.026949: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23db80f900 next 520 of size 27878400
2020-02-07 18:28:12.026961: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23dd2a5d00 next 521 of size 27878400
2020-02-07 18:28:12.026973: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23ded3c100 next 524 of size 27878400
2020-02-07 18:28:12.026985: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23e07d2500 next 523 of size 27878400
2020-02-07 18:28:12.026996: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23e2268900 next 525 of size 27878400
2020-02-07 18:28:12.027008: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23e3cfed00 next 522 of size 27878400
2020-02-07 18:28:12.027020: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23e5795100 next 526 of size 28558336
2020-02-07 18:28:12.027032: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23e72d1500 next 527 of size 27878400
2020-02-07 18:28:12.027044: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23e8d67900 next 528 of size 27878400
2020-02-07 18:28:12.027055: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23ea7fdd00 next 530 of size 27878400
2020-02-07 18:28:12.027067: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23ec294100 next 535 of size 27878400
2020-02-07 18:28:12.027079: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23edd2a500 next 538 of size 27878400
2020-02-07 18:28:12.027091: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23ef7c0900 next 539 of size 27878400
2020-02-07 18:28:12.027103: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23f1256d00 next 540 of size 27878400
2020-02-07 18:28:12.027115: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23f2ced100 next 541 of size 29593600
2020-02-07 18:28:12.027126: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23f4926100 next 542 of size 28217344
2020-02-07 18:28:12.027138: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23f640f100 next 543 of size 27878400
2020-02-07 18:28:12.027150: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23f7ea5500 next 544 of size 27878400
2020-02-07 18:28:12.027162: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23f993b900 next 546 of size 27878400
2020-02-07 18:28:12.027174: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23fb3d1d00 next 551 of size 27878400
2020-02-07 18:28:12.027186: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23fce68100 next 554 of size 27878400
2020-02-07 18:28:12.027197: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x23fe8fe500 next 555 of size 27878400
2020-02-07 18:28:12.027215: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2400394900 next 558 of size 27878400
2020-02-07 18:28:12.027234: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2401e2ad00 next 557 of size 27878400
2020-02-07 18:28:12.027247: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24038c1100 next 559 of size 27878400
2020-02-07 18:28:12.027259: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2405357500 next 556 of size 27878400
2020-02-07 18:28:12.027271: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2406ded900 next 560 of size 29593600
2020-02-07 18:28:12.027283: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2408a26900 next 561 of size 28217344
2020-02-07 18:28:12.027295: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x240a50f900 next 562 of size 27878400
2020-02-07 18:28:12.027307: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x240bfa5d00 next 564 of size 27878400
2020-02-07 18:28:12.027318: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x240da3c100 next 569 of size 27878400
2020-02-07 18:28:12.027330: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x240f4d2500 next 572 of size 27878400
2020-02-07 18:28:12.027342: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2410f68900 next 573 of size 27878400
2020-02-07 18:28:12.027354: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24129fed00 next 574 of size 27878400
2020-02-07 18:28:12.027366: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2414495100 next 575 of size 31719424
2020-02-07 18:28:12.027378: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24162d5100 next 576 of size 28901376
2020-02-07 18:28:12.027390: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2417e65100 next 577 of size 27878400
2020-02-07 18:28:12.027402: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24198fb500 next 578 of size 27878400
2020-02-07 18:28:12.027413: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x241b391900 next 580 of size 27878400
2020-02-07 18:28:12.027425: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x241ce27d00 next 585 of size 27878400
2020-02-07 18:28:12.027437: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x241e8be100 next 588 of size 27878400
2020-02-07 18:28:12.027449: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2420354500 next 589 of size 27878400
2020-02-07 18:28:12.027461: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2421dea900 next 592 of size 27878400
2020-02-07 18:28:12.027473: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2423880d00 next 591 of size 27878400
2020-02-07 18:28:12.027485: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2425317100 next 593 of size 27878400
2020-02-07 18:28:12.027497: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2426dad500 next 590 of size 27878400
2020-02-07 18:28:12.027508: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2428843900 next 594 of size 31719424
2020-02-07 18:28:12.027520: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x242a683900 next 595 of size 28901376
2020-02-07 18:28:12.027532: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x242c213900 next 596 of size 27878400
2020-02-07 18:28:12.027544: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x242dca9d00 next 598 of size 27878400
2020-02-07 18:28:12.027556: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x242f740100 next 603 of size 27878400
2020-02-07 18:28:12.027568: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24311d6500 next 606 of size 27878400
2020-02-07 18:28:12.027579: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2432c6c900 next 607 of size 27878400
2020-02-07 18:28:12.027599: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2434702d00 next 608 of size 27878400
2020-02-07 18:28:12.027613: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2436199100 next 609 of size 34668544
2020-02-07 18:28:12.027625: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24382a9100 next 610 of size 28901376
2020-02-07 18:28:12.027637: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2439e39100 next 611 of size 27878400
2020-02-07 18:28:12.027649: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x243b8cf500 next 612 of size 27878400
2020-02-07 18:28:12.027660: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x243d365900 next 614 of size 27878400
2020-02-07 18:28:12.027672: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x243edfbd00 next 619 of size 27878400
2020-02-07 18:28:12.027684: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2440892100 next 622 of size 27878400
2020-02-07 18:28:12.027696: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2442328500 next 623 of size 27878400
2020-02-07 18:28:12.027708: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2443dbe900 next 626 of size 27878400
2020-02-07 18:28:12.027719: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2445854d00 next 625 of size 27878400
2020-02-07 18:28:12.027731: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24472eb100 next 627 of size 27878400
2020-02-07 18:28:12.027743: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2448d81500 next 624 of size 27878400
2020-02-07 18:28:12.027755: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x244a817900 next 628 of size 34668544
2020-02-07 18:28:12.027767: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x244c927900 next 629 of size 28901376
2020-02-07 18:28:12.027778: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x244e4b7900 next 630 of size 27878400
2020-02-07 18:28:12.027790: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x244ff4dd00 next 632 of size 27878400
2020-02-07 18:28:12.027802: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24519e4100 next 637 of size 27878400
2020-02-07 18:28:12.027814: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x245347a500 next 640 of size 27878400
2020-02-07 18:28:12.027825: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2454f10900 next 641 of size 27878400
2020-02-07 18:28:12.027837: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24569a6d00 next 642 of size 27878400
2020-02-07 18:28:12.027849: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x245843d100 next 643 of size 27878400
2020-02-07 18:28:12.027861: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2459ed3500 next 644 of size 27878400
2020-02-07 18:28:12.027873: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x245b969900 next 650 of size 27878400
2020-02-07 18:28:12.027885: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x245d3ffd00 next 653 of size 27878400
2020-02-07 18:28:12.027896: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x245ee96100 next 654 of size 27878400
2020-02-07 18:28:12.027908: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x246092c500 next 657 of size 27878400
2020-02-07 18:28:12.027920: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24623c2900 next 656 of size 27878400
2020-02-07 18:28:12.027932: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2463e58d00 next 658 of size 27878400
2020-02-07 18:28:12.027944: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24658ef100 next 655 of size 27878400
2020-02-07 18:28:12.027955: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2467385500 next 659 of size 27878400
2020-02-07 18:28:12.027974: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2468e1b900 next 665 of size 27878400
2020-02-07 18:28:12.027987: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x246a8b1d00 next 668 of size 27878400
2020-02-07 18:28:12.027999: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x246c348100 next 669 of size 27878400
2020-02-07 18:28:12.028011: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x246ddde500 next 670 of size 27878400
2020-02-07 18:28:12.028023: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x246f874900 next 671 of size 28558336
2020-02-07 18:28:12.028035: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24713b0d00 next 672 of size 27878400
2020-02-07 18:28:12.028046: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2472e47100 next 673 of size 27878400
2020-02-07 18:28:12.028058: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24748dd500 next 674 of size 27878400
2020-02-07 18:28:12.028070: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2476373900 next 676 of size 27878400
2020-02-07 18:28:12.028081: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2477e09d00 next 681 of size 27878400
2020-02-07 18:28:12.028093: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24798a0100 next 684 of size 27878400
2020-02-07 18:28:12.028105: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x247b336500 next 685 of size 27878400
2020-02-07 18:28:12.028117: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x247cdcc900 next 688 of size 27878400
2020-02-07 18:28:12.028128: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x247e862d00 next 687 of size 27878400
2020-02-07 18:28:12.028140: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24802f9100 next 689 of size 27878400
2020-02-07 18:28:12.028152: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2481d8f500 next 686 of size 27878400
2020-02-07 18:28:12.028164: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2483825900 next 690 of size 28558336
2020-02-07 18:28:12.028176: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2485361d00 next 691 of size 27878400
2020-02-07 18:28:12.028187: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2486df8100 next 692 of size 27878400
2020-02-07 18:28:12.028199: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x248888e500 next 694 of size 27878400
2020-02-07 18:28:12.028211: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x248a324900 next 699 of size 27878400
2020-02-07 18:28:12.028227: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x248bdbad00 next 702 of size 27878400
2020-02-07 18:28:12.028252: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x248d851100 next 703 of size 27878400
2020-02-07 18:28:12.028265: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x248f2e7500 next 704 of size 27878400
2020-02-07 18:28:12.028278: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2490d7d900 next 705 of size 29593600
2020-02-07 18:28:12.028290: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24929b6900 next 706 of size 28217344
2020-02-07 18:28:12.028312: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x249449f900 next 707 of size 27878400
2020-02-07 18:28:12.028324: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2495f35d00 next 708 of size 27878400
2020-02-07 18:28:12.028336: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24979cc100 next 710 of size 27878400
2020-02-07 18:28:12.028347: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2499462500 next 715 of size 27878400
2020-02-07 18:28:12.028359: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x249aef8900 next 718 of size 27878400
2020-02-07 18:28:12.028371: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x249c98ed00 next 719 of size 27878400
2020-02-07 18:28:12.028390: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x249e425100 next 722 of size 27878400
2020-02-07 18:28:12.028404: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x249febb500 next 721 of size 27878400
2020-02-07 18:28:12.028417: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24a1951900 next 723 of size 27878400
2020-02-07 18:28:12.028429: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24a33e7d00 next 720 of size 27878400
2020-02-07 18:28:12.028440: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24a4e7e100 next 724 of size 29593600
2020-02-07 18:28:12.028452: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24a6ab7100 next 725 of size 28217344
2020-02-07 18:28:12.028464: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24a85a0100 next 726 of size 27878400
2020-02-07 18:28:12.028476: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24aa036500 next 728 of size 27878400
2020-02-07 18:28:12.028488: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24abacc900 next 733 of size 27878400
2020-02-07 18:28:12.028499: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24ad562d00 next 736 of size 27878400
2020-02-07 18:28:12.028511: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24aeff9100 next 737 of size 27878400
2020-02-07 18:28:12.028523: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24b0a8f500 next 738 of size 27878400
2020-02-07 18:28:12.028535: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24b2525900 next 739 of size 31719424
2020-02-07 18:28:12.028547: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24b4365900 next 740 of size 28901376
2020-02-07 18:28:12.028559: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24b5ef5900 next 741 of size 27878400
2020-02-07 18:28:12.028570: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24b798bd00 next 742 of size 27878400
2020-02-07 18:28:12.028582: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24b9422100 next 744 of size 27878400
2020-02-07 18:28:12.028594: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24baeb8500 next 749 of size 27878400
2020-02-07 18:28:12.028606: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24bc94e900 next 752 of size 27878400
2020-02-07 18:28:12.028617: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24be3e4d00 next 753 of size 27878400
2020-02-07 18:28:12.028629: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24bfe7b100 next 756 of size 27878400
2020-02-07 18:28:12.028641: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24c1911500 next 755 of size 27878400
2020-02-07 18:28:12.028653: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24c33a7900 next 757 of size 27878400
2020-02-07 18:28:12.028664: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24c4e3dd00 next 754 of size 27878400
2020-02-07 18:28:12.028676: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24c68d4100 next 758 of size 31719424
2020-02-07 18:28:12.028688: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24c8714100 next 759 of size 28901376
2020-02-07 18:28:12.028700: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24ca2a4100 next 760 of size 27878400
2020-02-07 18:28:12.028712: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24cbd3a500 next 762 of size 27878400
2020-02-07 18:28:12.028724: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24cd7d0900 next 767 of size 27878400
2020-02-07 18:28:12.028736: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24cf266d00 next 770 of size 27878400
2020-02-07 18:28:12.028747: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24d0cfd100 next 771 of size 27878400
2020-02-07 18:28:12.028766: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24d2793500 next 772 of size 27878400
2020-02-07 18:28:12.028780: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24d4229900 next 773 of size 34668544
2020-02-07 18:28:12.028792: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24d6339900 next 774 of size 28901376
2020-02-07 18:28:12.028804: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24d7ec9900 next 775 of size 27878400
2020-02-07 18:28:12.028816: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24d995fd00 next 776 of size 27878400
2020-02-07 18:28:12.028828: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24db3f6100 next 778 of size 27878400
2020-02-07 18:28:12.028839: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24dce8c500 next 783 of size 27878400
2020-02-07 18:28:12.028851: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24de922900 next 786 of size 27878400
2020-02-07 18:28:12.028863: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24e03b8d00 next 787 of size 27878400
2020-02-07 18:28:12.028875: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24e1e4f100 next 790 of size 27878400
2020-02-07 18:28:12.028887: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24e38e5500 next 789 of size 27878400
2020-02-07 18:28:12.028898: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24e537b900 next 791 of size 27878400
2020-02-07 18:28:12.028910: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24e6e11d00 next 788 of size 27878400
2020-02-07 18:28:12.028922: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24e88a8100 next 792 of size 34668544
2020-02-07 18:28:12.028934: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24ea9b8100 next 793 of size 28901376
2020-02-07 18:28:12.028945: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24ec548100 next 794 of size 27878400
2020-02-07 18:28:12.028957: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24edfde500 next 796 of size 27878400
2020-02-07 18:28:12.028969: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24efa74900 next 801 of size 27878400
2020-02-07 18:28:12.028981: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24f150ad00 next 804 of size 27878400
2020-02-07 18:28:12.028993: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24f2fa1100 next 805 of size 27878400
2020-02-07 18:28:12.029004: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24f4a37500 next 806 of size 27878400
2020-02-07 18:28:12.029016: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24f64cd900 next 807 of size 27878400
2020-02-07 18:28:12.029028: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24f7f63d00 next 808 of size 27878400
2020-02-07 18:28:12.029040: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24f99fa100 next 814 of size 27878400
2020-02-07 18:28:12.029051: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24fb490500 next 817 of size 27878400
2020-02-07 18:28:12.029063: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24fcf26900 next 818 of size 27878400
2020-02-07 18:28:12.029075: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x24fe9bcd00 next 821 of size 27878400
2020-02-07 18:28:12.029087: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2500453100 next 820 of size 27878400
2020-02-07 18:28:12.029099: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2501ee9500 next 822 of size 27878400
2020-02-07 18:28:12.029110: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x250397f900 next 819 of size 27878400
2020-02-07 18:28:12.029122: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2505415d00 next 823 of size 27878400
2020-02-07 18:28:12.029140: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2506eac100 next 829 of size 27878400
2020-02-07 18:28:12.029154: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2508942500 next 832 of size 27878400
2020-02-07 18:28:12.029166: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x250a3d8900 next 833 of size 27878400
2020-02-07 18:28:12.029178: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x250be6ed00 next 834 of size 27878400
2020-02-07 18:28:12.029190: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x250d905100 next 835 of size 28558336
2020-02-07 18:28:12.029202: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x250f441500 next 836 of size 27878400
2020-02-07 18:28:12.029214: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2510ed7900 next 837 of size 27878400
2020-02-07 18:28:12.029229: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x251296dd00 next 838 of size 27878400
2020-02-07 18:28:12.029254: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2514404100 next 840 of size 27878400
2020-02-07 18:28:12.029267: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2515e9a500 next 845 of size 27878400
2020-02-07 18:28:12.029280: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2517930900 next 848 of size 27878400
2020-02-07 18:28:12.029302: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25193c6d00 next 849 of size 27878400
2020-02-07 18:28:12.029314: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x251ae5d100 next 852 of size 27878400
2020-02-07 18:28:12.029326: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x251c8f3500 next 851 of size 27878400
2020-02-07 18:28:12.029338: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x251e389900 next 853 of size 27878400
2020-02-07 18:28:12.029350: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x251fe1fd00 next 850 of size 27878400
2020-02-07 18:28:12.029361: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25218b6100 next 854 of size 28558336
2020-02-07 18:28:12.029373: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25233f2500 next 855 of size 27878400
2020-02-07 18:28:12.029385: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2524e88900 next 856 of size 27878400
2020-02-07 18:28:12.029397: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x252691ed00 next 858 of size 27878400
2020-02-07 18:28:12.029409: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25283b5100 next 863 of size 27878400
2020-02-07 18:28:12.029421: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2529e4b500 next 866 of size 27878400
2020-02-07 18:28:12.029432: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x252b8e1900 next 867 of size 27878400
2020-02-07 18:28:12.029444: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x252d377d00 next 868 of size 27878400
2020-02-07 18:28:12.029456: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x252ee0e100 next 869 of size 29593600
2020-02-07 18:28:12.029468: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2530a47100 next 870 of size 28217344
2020-02-07 18:28:12.029480: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2532530100 next 871 of size 27878400
2020-02-07 18:28:12.029491: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2533fc6500 next 872 of size 27878400
2020-02-07 18:28:12.029503: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2535a5c900 next 874 of size 27878400
2020-02-07 18:28:12.029515: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25374f2d00 next 879 of size 27878400
2020-02-07 18:28:12.029527: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2538f89100 next 882 of size 27878400
2020-02-07 18:28:12.029539: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x253aa1f500 next 883 of size 27878400
2020-02-07 18:28:12.029557: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x253c4b5900 next 886 of size 27878400
2020-02-07 18:28:12.029571: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x253df4bd00 next 885 of size 27878400
2020-02-07 18:28:12.029583: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x253f9e2100 next 887 of size 27878400
2020-02-07 18:28:12.029594: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2541478500 next 884 of size 27878400
2020-02-07 18:28:12.029606: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2542f0e900 next 888 of size 29593600
2020-02-07 18:28:12.029618: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2544b47900 next 889 of size 28217344
2020-02-07 18:28:12.029629: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2546630900 next 890 of size 27878400
2020-02-07 18:28:12.029641: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25480c6d00 next 892 of size 27878400
2020-02-07 18:28:12.029653: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2549b5d100 next 897 of size 27878400
2020-02-07 18:28:12.029665: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x254b5f3500 next 900 of size 27878400
2020-02-07 18:28:12.029676: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x254d089900 next 901 of size 27878400
2020-02-07 18:28:12.029688: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x254eb1fd00 next 902 of size 27878400
2020-02-07 18:28:12.029699: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25505b6100 next 903 of size 31719424
2020-02-07 18:28:12.029711: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25523f6100 next 904 of size 28901376
2020-02-07 18:28:12.029723: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2553f86100 next 905 of size 27878400
2020-02-07 18:28:12.029734: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2555a1c500 next 906 of size 27878400
2020-02-07 18:28:12.029746: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25574b2900 next 908 of size 27878400
2020-02-07 18:28:12.029758: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2558f48d00 next 913 of size 27878400
2020-02-07 18:28:12.029769: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x255a9df100 next 916 of size 27878400
2020-02-07 18:28:12.029781: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x255c475500 next 917 of size 27878400
2020-02-07 18:28:12.029793: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x255df0b900 next 920 of size 27878400
2020-02-07 18:28:12.029804: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x255f9a1d00 next 919 of size 27878400
2020-02-07 18:28:12.029816: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2561438100 next 921 of size 27878400
2020-02-07 18:28:12.029828: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2562ece500 next 918 of size 27878400
2020-02-07 18:28:12.029839: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2564964900 next 922 of size 31719424
2020-02-07 18:28:12.029851: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25667a4900 next 923 of size 28901376
2020-02-07 18:28:12.029863: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2568334900 next 924 of size 27878400
2020-02-07 18:28:12.029874: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2569dcad00 next 926 of size 27878400
2020-02-07 18:28:12.029886: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x256b861100 next 931 of size 27878400
2020-02-07 18:28:12.029897: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x256d2f7500 next 934 of size 27878400
2020-02-07 18:28:12.029909: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x256ed8d900 next 935 of size 27878400
2020-02-07 18:28:12.029927: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2570823d00 next 936 of size 27878400
2020-02-07 18:28:12.029940: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25722ba100 next 937 of size 34668544
2020-02-07 18:28:12.029952: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25743ca100 next 938 of size 28901376
2020-02-07 18:28:12.029964: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2575f5a100 next 939 of size 27878400
2020-02-07 18:28:12.029975: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25779f0500 next 940 of size 27878400
2020-02-07 18:28:12.029987: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2579486900 next 942 of size 27878400
2020-02-07 18:28:12.029999: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x257af1cd00 next 947 of size 27878400
2020-02-07 18:28:12.030010: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x257c9b3100 next 950 of size 27878400
2020-02-07 18:28:12.030022: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x257e449500 next 951 of size 27878400
2020-02-07 18:28:12.030034: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x257fedf900 next 954 of size 27878400
2020-02-07 18:28:12.030046: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2581975d00 next 953 of size 27878400
2020-02-07 18:28:12.030057: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x258340c100 next 955 of size 27878400
2020-02-07 18:28:12.030069: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2584ea2500 next 952 of size 27878400
2020-02-07 18:28:12.030081: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2586938900 next 956 of size 34668544
2020-02-07 18:28:12.030092: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2588a48900 next 957 of size 28901376
2020-02-07 18:28:12.030104: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x258a5d8900 next 958 of size 27878400
2020-02-07 18:28:12.030116: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x258c06ed00 next 960 of size 27878400
2020-02-07 18:28:12.030127: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x258db05100 next 965 of size 27878400
2020-02-07 18:28:12.030139: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x258f59b500 next 968 of size 27878400
2020-02-07 18:28:12.030151: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2591031900 next 969 of size 27878400
2020-02-07 18:28:12.030162: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2592ac7d00 next 970 of size 27878400
2020-02-07 18:28:12.030174: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x259455e100 next 971 of size 27878400
2020-02-07 18:28:12.030186: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2595ff4500 next 972 of size 27878400
2020-02-07 18:28:12.030197: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2597a8a900 next 978 of size 27878400
2020-02-07 18:28:12.030209: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x2599520d00 next 981 of size 27878400
2020-02-07 18:28:12.030221: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x259afb7100 next 982 of size 27878400
2020-02-07 18:28:12.030238: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x259ca4d500 next 985 of size 27878400
2020-02-07 18:28:12.030251: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x259e4e3900 next 984 of size 27878400
2020-02-07 18:28:12.030263: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x259ff79d00 next 986 of size 27878400
2020-02-07 18:28:12.030274: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25a1a10100 next 983 of size 27878400
2020-02-07 18:28:12.030286: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25a34a6500 next 987 of size 27878400
2020-02-07 18:28:12.030305: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25a4f3c900 next 993 of size 27878400
2020-02-07 18:28:12.030319: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25a69d2d00 next 996 of size 27878400
2020-02-07 18:28:12.030331: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25a8469100 next 997 of size 27878400
2020-02-07 18:28:12.030342: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25a9eff500 next 998 of size 27878400
2020-02-07 18:28:12.030354: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25ab995900 next 999 of size 28558336
2020-02-07 18:28:12.030366: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25ad4d1d00 next 1000 of size 27878400
2020-02-07 18:28:12.030378: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25aef68100 next 1001 of size 27878400
2020-02-07 18:28:12.030389: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25b09fe500 next 1002 of size 27878400
2020-02-07 18:28:12.030401: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25b2494900 next 1004 of size 27878400
2020-02-07 18:28:12.030413: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25b3f2ad00 next 1009 of size 27878400
2020-02-07 18:28:12.030424: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25b59c1100 next 1012 of size 27878400
2020-02-07 18:28:12.030436: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25b7457500 next 1013 of size 27878400
2020-02-07 18:28:12.030448: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25b8eed900 next 1016 of size 27878400
2020-02-07 18:28:12.030460: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25ba983d00 next 1015 of size 27878400
2020-02-07 18:28:12.030471: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25bc41a100 next 1017 of size 27878400
2020-02-07 18:28:12.030483: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25bdeb0500 next 1014 of size 27878400
2020-02-07 18:28:12.030495: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25bf946900 next 1018 of size 28558336
2020-02-07 18:28:12.030506: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25c1482d00 next 1019 of size 27878400
2020-02-07 18:28:12.030518: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25c2f19100 next 1020 of size 27878400
2020-02-07 18:28:12.030530: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25c49af500 next 1022 of size 27878400
2020-02-07 18:28:12.030542: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25c6445900 next 1027 of size 27878400
2020-02-07 18:28:12.030553: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25c7edbd00 next 1030 of size 27878400
2020-02-07 18:28:12.030565: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25c9972100 next 1031 of size 27878400
2020-02-07 18:28:12.030577: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25cb408500 next 1032 of size 27878400
2020-02-07 18:28:12.030588: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25cce9e900 next 1033 of size 29593600
2020-02-07 18:28:12.030600: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25cead7900 next 1034 of size 28217344
2020-02-07 18:28:12.030612: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25d05c0900 next 1035 of size 27878400
2020-02-07 18:28:12.030624: I tensorflow/core/common_runtime/bfc_allocator.cc:905] InUse at 0x25d2056d00 next 18446744073709551615 of size 41358080
2020-02-07 18:28:12.030636: I tensorflow/core/common_runtime/bfc_allocator.cc:914]      Summary of in-use Chunks by size: 
2020-02-07 18:28:12.030654: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 538 Chunks of size 256 totalling 134.5KiB
2020-02-07 18:28:12.030669: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 768 totalling 768B
2020-02-07 18:28:12.030690: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 1280 totalling 1.2KiB
2020-02-07 18:28:12.030705: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 134656 totalling 131.5KiB
2020-02-07 18:28:12.030719: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 72 Chunks of size 147456 totalling 10.12MiB
2020-02-07 18:28:12.030732: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 356 Chunks of size 27878400 totalling 9.24GiB
2020-02-07 18:28:12.030745: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 9 Chunks of size 28217344 totalling 242.19MiB
2020-02-07 18:28:12.030758: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 10 Chunks of size 28558336 totalling 272.35MiB
2020-02-07 18:28:12.030771: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 16 Chunks of size 28901376 totalling 441.00MiB
2020-02-07 18:28:12.030784: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 9 Chunks of size 29593600 totalling 254.00MiB
2020-02-07 18:28:12.030798: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 30927616 totalling 29.49MiB
2020-02-07 18:28:12.030811: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 8 Chunks of size 31719424 totalling 242.00MiB
2020-02-07 18:28:12.030824: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 8 Chunks of size 34668544 totalling 264.50MiB
2020-02-07 18:28:12.030837: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 41358080 totalling 39.44MiB
2020-02-07 18:28:12.030850: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 48787200 totalling 46.53MiB
2020-02-07 18:28:12.030863: I tensorflow/core/common_runtime/bfc_allocator.cc:917] 1 Chunks of size 229125632 totalling 218.51MiB
2020-02-07 18:28:12.030876: I tensorflow/core/common_runtime/bfc_allocator.cc:921] Sum Total of in-use chunks: 11.25GiB
2020-02-07 18:28:12.030889: I tensorflow/core/common_runtime/bfc_allocator.cc:923] total_region_allocated_bytes_: 12092604416 memory_limit_: 12092604416 available bytes: 0 curr_region_allocation_bytes_: 24185208832
2020-02-07 18:28:12.030905: I tensorflow/core/common_runtime/bfc_allocator.cc:929] Stats: 
Limit:                 12092604416
InUse:                 12085207808
MaxInUse:              12085355008
NumAllocs:                    2879
MaxAllocSize:           2317877248

2020-02-07 18:28:12.030946: W tensorflow/core/common_runtime/bfc_allocator.cc:424] ****************************************************************************************************
2020-02-07 18:28:12.030979: W tensorflow/core/framework/op_kernel.cc:1622] OP_REQUIRES failed at cwise_ops_common.cc:82 : Resource exhausted: OOM when allocating tensor with shape[1,330,330,64,1] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc
Traceback (most recent call last):
  File "/oasis/projects/nsf/mia174/qingyliu/test/RESBLOCK_training_comet.py", line 457, in <module>
    callbacks = [cp_callback_all_conv])
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training.py", line 1297, in fit_generator
    steps_name='steps_per_epoch')
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_generator.py", line 265, in model_iteration
    batch_outs = batch_function(*batch_data)
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training.py", line 973, in train_on_batch
    class_weight=class_weight, reset_metrics=reset_metrics)
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_v2_utils.py", line 264, in train_on_batch
    output_loss_metrics=model._output_loss_metrics)
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_eager.py", line 311, in train_on_batch
    output_loss_metrics=output_loss_metrics))
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_eager.py", line 252, in _process_single_batch
    training=training))
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/training_eager.py", line 127, in _model_loss
    outs = model(inputs, **kwargs)
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/base_layer.py", line 891, in __call__
    outputs = self.call(cast_inputs, *args, **kwargs)
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/network.py", line 708, in call
    convert_kwargs_to_constants=base_layer_utils.call_context().saving)
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/network.py", line 860, in _run_internal_graph
    output_tensors = layer(computed_tensors, **kwargs)
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/keras/engine/base_layer.py", line 891, in __call__
    outputs = self.call(cast_inputs, *args, **kwargs)
  File "/oasis/projects/nsf/mia174/qingyliu/test/RESBLOCK_training_comet.py", line 135, in call
    input_shape)
  File "/oasis/projects/nsf/mia174/qingyliu/test/RESBLOCK_training_comet.py", line 189, in _apply_normalization
    reshaped_inputs, group_reduction_axes, keepdims=True)
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_impl.py", line 1277, in moments_v2
    return moments(x=x, axes=axes, shift=shift, name=name, keep_dims=keepdims)
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/ops/nn_impl.py", line 1230, in moments
    math_ops.squared_difference(y, array_ops.stop_gradient(mean)),
  File "/oasis/projects/nsf/mia174/qingyliu/miniconda3/lib/python3.7/site-packages/tensorflow_core/python/ops/gen_math_ops.py", line 11010, in squared_difference
    _six.raise_from(_core._status_to_exception(e.code, message), None)
  File "<string>", line 3, in raise_from
tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[1,330,330,64,1] and type float on /job:localhost/replica:0/task:0/device:GPU:0 by allocator GPU_0_bfc [Op:SquaredDifference]

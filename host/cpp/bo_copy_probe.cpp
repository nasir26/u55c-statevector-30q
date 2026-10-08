// Times xrt::bo::copy between BOs in two different HBM bank groups (device-side M2M if supported).
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>
#include <xrt/xrt_bo.h>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <vector>
int main(int argc, char **argv) {
    xrt::device dev{std::string(argv[1])};
    auto uuid = dev.load_xclbin(argv[2]);
    xrt::kernel k(dev, uuid, argv[3]);
    for (size_t mb : {16, 64, 256}) {
        size_t sz = mb << 20;
        xrt::bo a(dev, sz, k.group_id(0)), b(dev, sz, k.group_id(atoi(argv[4])));
        auto pa = a.map<unsigned *>();
        for (size_t i = 0; i < sz / 4; i++) pa[i] = (unsigned)i;
        a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        double best = 1e9;
        for (int r = 0; r < 5; r++) {
            auto t0 = std::chrono::steady_clock::now();
            b.copy(a, sz, 0, 0);
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (s < best) best = s;
        }
        b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bool ok = memcmp(b.map<unsigned *>(), pa, sz) == 0;
        printf("{\"MB\": %zu, \"copy_best_s\": %.6f, \"copy_gbps\": %.2f, \"correct\": %s}\n", mb, best, sz / best / 1e9, ok ? "true" : "false");
        fflush(stdout);
    }
}

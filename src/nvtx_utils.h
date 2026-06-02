#pragma once

#include <cstdio>
#include <string>

#if defined(ENABLE_NVTX)
#if defined(__has_include)
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#elif __has_include(<nvToolsExt.h>)
#include <nvToolsExt.h>
#else
#error "ENABLE_NVTX is defined but no NVTX header was found"
#endif
#else
#include <nvToolsExt.h>
#endif
#endif

/** RAII NVTX range; no-op when ENABLE_NVTX is undefined. */
class NvtxRange {
public:
    explicit NvtxRange(const char* name) {
#if defined(ENABLE_NVTX)
        nvtxRangePushA(name);
#else
        (void)name;
#endif
    }

    NvtxRange(const char* prefix, int index) {
#if defined(ENABLE_NVTX)
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%s_%d", prefix, index);
        owned_name_ = buf;
        nvtxRangePushA(owned_name_.c_str());
#else
        (void)prefix;
        (void)index;
#endif
    }

    ~NvtxRange() {
#if defined(ENABLE_NVTX)
        nvtxRangePop();
#endif
    }

    NvtxRange(const NvtxRange&) = delete;
    NvtxRange& operator=(const NvtxRange&) = delete;

private:
    std::string owned_name_;
};

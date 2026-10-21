#pragma once
// Thin alias header — the skin renderer uses the exact same GPU pipeline
// as android_smpl_app.  This typedef avoids renaming every symbol while
// keeping the include path self-contained within skin_rendering/.

#include "smpl_gpu_pipeline.h"

using SkinGpuPipeline = SmplGpuPipeline;

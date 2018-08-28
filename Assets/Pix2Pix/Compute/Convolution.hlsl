#ifndef CONVOLUTION_TRANSPOSE

(const uint3 tid : SV_DispatchThreadID)
{
    const uint FilterSize = 4; // We assume FilterShape.xy == (4, 4)
    const uint InputChannels = InputShape.z;

    // pos - pad = (upper left corner)
    const uint2 pos = tid.zy * 2;
    const uint pad = FilterSize / 2 - 1;

    float prod = 0;

    for (uint fy = 0; fy < FilterSize; fy++)
    {
        for (uint fx = 0; fx < FilterSize; fx++)
        {
            const uint cl = fx & 1; // Cache line selector

            // Cache the input channel values in a memory coalescing fashion.
            if (tid.x < InputChannels)
                cache[cl][tid.x] = GetInput(uint3(pos + uint2(fy, fx), tid.x), pad);

            GroupMemoryBarrierWithGroupSync();

            // Calculate the product with the filter. This is also expected to
            // run in a memory coalescing fashion.
            for (uint ic = 0; ic < InputChannels; ic++)
                prod += GetFilter(int4(fy, fx, ic, tid.x)) * cache[cl][ic];
        }
    }

    // Output with adding the bias.
    Output[OutputIndex(tid.zyx)] = prod + Bias[tid.x];
}

#else

(const uint3 tid : SV_DispatchThreadID)
{
    const uint FilterSize = 4; // We assume FilterShape.xy == (4, 4)
    const uint InputChannels = InputShape.z;
    const uint OutputChannels = OutputShape.z;

    float prod = 0;

    for (uint fy = 0; fy < FilterSize; fy += 2)
    {
        for (uint fx = 0; fx < FilterSize; fx += 2)
        {
            const uint cl = fx >> 1; // Cache line selector

            // Actually (tid.zy & 1) should be added to (fy, fx) but we avoid
            // it to prevent the loop counters depending the thread IDs. So, we
            // recalculate them here.
            const uint2 fyx = uint2(fy, fx) + (tid.zy & 1);

            // Cache the input channel values in a memory coalescing fashion.
            uint ic;
        #ifdef CONVOLUTION_TRANSPOSE_FINAL
            cache[cl][tid.x     ] = GetInput(uint3((tid.zy + fyx) / 2, tid.x     ), 1);
            cache[cl][tid.x + 32] = GetInput(uint3((tid.zy + fyx) / 2, tid.x + 32), 1);
        #else
            for (ic = 0; ic < InputChannels; ic += OutputChannels)
                cache[cl][ic + tid.x] = GetInput(uint3((tid.zy + fyx) / 2, ic + tid.x), 1);
        #endif

            GroupMemoryBarrierWithGroupSync();

            // Transposed version of fyx
            const uint2 fyx_tr = FilterSize - 1 - fyx;

            // Calculate the product with the filter. This is also expected to
            // run in a memory coalescing fashion.
        #ifdef CONVOLUTION_TRANSPOSE_FINAL
            if (tid.x < OutputChannels)
        #endif
            for (ic = 0; ic < InputChannels; ic++)
                prod += GetFilter(uint4(fyx_tr, ic, tid.x)) * cache[cl][ic];
        }
    }

    // Output with adding the bias.
#ifdef CONVOLUTION_TRANSPOSE_FINAL
    if (tid.x < OutputChannels)
#endif
    Output[OutputIndex(tid.zyx)] = prod + Bias[tid.x];
}

#endif

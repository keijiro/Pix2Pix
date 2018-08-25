#ifndef CONVOLUTION_TRANSPOSE

(uint3 tid : SV_DispatchThreadID, uint gid : SV_GroupIndex)
{
    uint2 pos = tid.zy * 2;
    uint2 pad = FilterShape.xy / 2 - 1;

    float prod = 0;

    for (uint fy = 0; fy < FilterShape.x; fy++)
    {
        for (uint fx = 0; fx < FilterShape.y; fx++)
        {
            if (gid < InputShape.z)
                cache[fx][gid] = GetInput(uint3(pos + uint2(fy, fx), gid), pad);

            GroupMemoryBarrierWithGroupSync();

            for (uint ic = 0; ic < InputShape.z; ic++)
                prod += cache[fx][ic] * GetFilter(int4(fy, fx, ic, tid.x));
        }
    }

    Output[OutputIndex(tid.zyx)] = prod + Bias[tid.x];
}

#else

(uint3 tid : SV_DispatchThreadID, uint gid : SV_GroupIndex)
{
    uint2 pos = (tid.zy + 1) / 2;
    uint2 pad = 1;

    float prod = 0;

    uint2 fd = tid.zy & 1;

    for (uint fy = 0; fy < FilterShape.x; fy += 2)
    {
        for (uint fx = 0; fx < FilterShape.y; fx += 2)
        {
            uint2 fp = uint2(fy, fx) + fd;

            cache[fx][gid] = GetInput(uint3(pos + fp / 2, gid), pad);

            GroupMemoryBarrierWithGroupSync();

    if (gid < OutputShape.z)
            for (uint ic = 0; ic < InputShape.z; ic++)
            {
                //float x = GetInput(uint3(pos + fp / 2, ic), pad);
                float x = cache[fx][ic];
                float w = GetFilter(uint4(FilterShape.xy - 1 - fp, ic, tid.x));
                prod += x * w;
            }
        }
    }

    if (gid < OutputShape.z)
        Output[OutputIndex(tid.zyx)] = prod + Bias[tid.x];
}

#endif

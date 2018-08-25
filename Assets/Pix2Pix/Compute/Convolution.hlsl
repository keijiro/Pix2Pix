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

(uint3 tid : SV_DispatchThreadID)
{
    uint2 pos = (tid.zy + 1) / 2;
    uint2 pad = 1;

    float prod = 0;

    for (uint fy = tid.z & 1; fy < FilterShape.x; fy += 2)
    {
        for (uint fx = tid.y & 1; fx < FilterShape.y; fx += 2)
        {
            for (uint ic = 0; ic < InputShape.z; ic++)
            {
                uint2 fp = uint2(fy, fx);
                float x = GetInput(uint3(pos + fp / 2, ic), pad);
                float w = GetFilter(uint4(FilterShape.xy - 1 - fp, tid.x, ic));
                prod += x * w;
            }
        }
    }

    Output[OutputIndex(tid.zyx)] = prod + Bias[tid.x];
}

#endif

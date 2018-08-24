#ifndef CONVOLUTION_TRANSPOSE

(int3 tid : SV_DispatchThreadID)
{
    int xmin = (int)tid.y * 2 - (FilterShape.y >> 1) + 1;
    int ymin = (int)tid.z * 2 - (FilterShape.x >> 1) + 1;

    float prod = 0;

    for (int fy = 0; fy < FilterShape.x; fy++)
    {
        for (int fx = 0; fx < FilterShape.y; fx++)
        {
            for (int ic = 0; ic < InputShape.z; ic++)
            {
                float pixel = GetInput(int3(ymin + fy, xmin + fx, ic));
                float weight = GetFilter(int4(fy, fx, ic, tid.x));
                prod += pixel * weight;
            }
        }
    }

    Output[OutputIndex(tid.zyx)] = prod + Bias[tid.x];
}

#else

(int3 tid : SV_DispatchThreadID)
{
    int xmin = (tid.y - 1) >> 1;
    int ymin = (tid.z - 1) >> 1;

    float prod = 0;

    for (int fy = tid.z & 1; fy < FilterShape.x; fy += 2)
    {
        for (int fx = tid.y & 1; fx < FilterShape.y; fx += 2)
        {
            for (int ic = 0; ic < InputShape.z; ic++)
            {
                float pixel = GetInput(int3(ymin + (fy >> 1), xmin + (fx >> 1), ic));
                float weight = GetFilter(int4(FilterShape.x - 1 - fy, FilterShape.y - 1 - fx, tid.x, ic));
                prod += pixel * weight;
            }
        }
    }

    Output[OutputIndex(tid.zyx)] = prod + Bias[tid.x];
}

#endif

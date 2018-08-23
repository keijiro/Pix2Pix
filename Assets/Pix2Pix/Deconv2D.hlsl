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

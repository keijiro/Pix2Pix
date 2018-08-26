(const uint tid : SV_DispatchThreadID)
{
    const uint count = InputShape.x * InputShape.y;
    const uint stride = InputShape.z;

    uint i;

    float mean = 0;
    for (i = 0; i < count; i++)
        mean += Input[i * stride + tid];
    mean /= count;

    float variance = 0;
    for (i = 0; i < count; i++)
        variance += square(Input[i * stride + tid] - mean);
    variance /= count;

    float scale = Scale[tid] / sqrt(variance + 1e-5);
    float offset = Offset[tid];

    for (i = 0; i < count; i++)
    {
        uint idx = i * stride + tid;
        Output[idx] = (Input[idx] - mean) * scale + offset;
    }
}

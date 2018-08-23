(int tid : SV_DispatchThreadID)
{
    int input_length = InputShape.x * InputShape.y * InputShape.z;
    int i;

    float mean = 0;

    for (i = tid; i < input_length; i += InputShape.z)
        mean += Input[i];

    mean /= InputShape.x * InputShape.y;

    float variance = 0;

    for (i = tid; i < input_length; i += InputShape.z)
        variance += square(Input[i] - mean);

    variance /= InputShape.x * InputShape.y;

    float sc = Filter[tid];
    float offs = Bias[tid];

    sc /= sqrt(variance + 1e-5);

    for (i = tid; i < input_length; i += InputShape.z)
        Output[i] = offs + (Input[i] - mean) * sc;
}

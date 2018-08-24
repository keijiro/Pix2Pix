(int tid : SV_DispatchThreadID)
{
    int ch1 = InputShape.z;
    int ch2 = Input2Shape.z;

    int i1 = tid.x * ch1;
    int i2 = tid.x * ch2;
    int io = tid.x * (ch1 + ch2);

    int i;

    for (i = 0; i < ch1; i ++)
        Output[io++] = Input[i1++];

    for (i = 0; i < ch2; i ++)
        Output[io++] = Input2[i2++];
}

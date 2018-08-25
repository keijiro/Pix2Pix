(uint tid : SV_DispatchThreadID)
{
    uint ch1 = Input1Shape.z;
    uint ch2 = Input2Shape.z;

    uint i1 = tid.x * ch1;
    uint i2 = tid.x * ch2;
    uint io = tid.x * (ch1 + ch2);

    uint i;
    for (i = 0; i < ch1; i ++) Output[io++] = Input1[i1++];
    for (i = 0; i < ch2; i ++) Output[io++] = Input2[i2++];
}

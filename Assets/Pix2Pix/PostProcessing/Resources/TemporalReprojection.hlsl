#include "PostProcessing/Shaders/StdLib.hlsl"

#define SAMPLE_TEX2D(name, uv) SAMPLE_TEXTURE2D(name, sampler##name, uv)
#define SAMPLE_DEPTH(name, uv) SAMPLE_DEPTH_TEXTURE(name, sampler##name, uv).x

TEXTURE2D_SAMPLER2D(_MainTex, sampler_MainTex);
TEXTURE2D_SAMPLER2D(_RemapTex, sampler_RemapTex);
TEXTURE2D_SAMPLER2D(_CameraDepthTexture, sampler_CameraDepthTexture);
TEXTURE2D_SAMPLER2D(_CameraMotionVectorsTexture, sampler_CameraMotionVectorsTexture);

struct FragmentOutput
{
    half4 color : SV_Target0;
    half2 remap : SV_Target1;
};

bool DepthMask(float depth)
{
    const float epsilon = 1e-5;
#if defined(UNITY_REVERSED_Z)
    return depth > epsilon;
#else
    return depth < 1 - epsilon;
#endif
}

FragmentOutput FragInitialize(VaryingsDefault i)
{
    FragmentOutput o;
    o.color = SAMPLE_TEX2D(_MainTex, i.texcoord);
    o.remap = i.texcoord;
    return o;
}

FragmentOutput FragUpdate(VaryingsDefault i)
{
    half2 mv = SAMPLE_TEX2D(_CameraMotionVectorsTexture, i.texcoord).xy;
    half2 remap = SAMPLE_TEX2D(_RemapTex, i.texcoord - mv).xy;

    half4 c1 = SAMPLE_TEX2D(_MainTex, i.texcoord);
    half4 c2 = SAMPLE_TEX2D(_MainTex, remap);
    bool mask = DepthMask(SAMPLE_DEPTH(_CameraDepthTexture, i.texcoord));

    FragmentOutput o;
    o.color = lerp(c1, c2, mask);
    o.remap = remap;
    return o;
}

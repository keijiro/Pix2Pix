#include "PostProcessing/Shaders/StdLib.hlsl"
#include "PostProcessing/Shaders/Colors.hlsl"

#define SAMPLE_TEX2D(name, uv) SAMPLE_TEXTURE2D(name, sampler##name, uv)
#define SAMPLE_DEPTH(name, uv) SAMPLE_DEPTH_TEXTURE(name, sampler##name, uv).x

TEXTURE2D_SAMPLER2D(_MainTex, sampler_MainTex);
TEXTURE2D_SAMPLER2D(_CameraDepthTexture, sampler_CameraDepthTexture);
TEXTURE2D_SAMPLER2D(_CameraMotionVectorsTexture, sampler_CameraMotionVectorsTexture);
TEXTURE2D_SAMPLER2D(_UVRemap, sampler_UVRemap);
TEXTURE2D_SAMPLER2D(_PrevUVRemap, sampler_PrevUVRemap);
TEXTURE2D_SAMPLER2D(_PrevMoDepth, sampler_PrevMoDepth);

float4 _CameraDepthTexture_TexelSize;

float _DepthWeight;
float _MotionWeight;
float2 _DeltaTime;

struct FragmentOutput
{
    half4 uvRemap : SV_Target0;
    half4 moDepth : SV_Target1;
};

// Converts a raw depth value to a linear 0-1 depth value.
float LinearizeDepth(float z)
{
    // A little bit complicated to take the current projection mode
    // (perspective/orthographic) into account.
    float isOrtho = unity_OrthoParams.w;
    float isPers = 1.0 - unity_OrthoParams.w;
    z *= _ZBufferParams.x;
    return (1.0 - isOrtho * z) / (isPers * z + _ZBufferParams.y);
}

float3 CompareDepth(float3 min_uvd, float2 uv)
{
    float d = SAMPLE_DEPTH(_CameraDepthTexture, uv);
    return d < min_uvd ? float3(uv, d) : min_uvd;
}

float2 SearchClosest(float2 uv)
{
    float4 duv = _CameraDepthTexture_TexelSize.xyxy * float4(1, 1, -1, 0);

    float3 min_uvd = float3(uv, SAMPLE_DEPTH(_CameraDepthTexture, uv));

    min_uvd = CompareDepth(min_uvd, uv - duv.xy);
    min_uvd = CompareDepth(min_uvd, uv - duv.wy);
    min_uvd = CompareDepth(min_uvd, uv - duv.zy);

    min_uvd = CompareDepth(min_uvd, uv + duv.zw);
    min_uvd = CompareDepth(min_uvd, uv + duv.xw);

    min_uvd = CompareDepth(min_uvd, uv + duv.zy);
    min_uvd = CompareDepth(min_uvd, uv + duv.wy);
    min_uvd = CompareDepth(min_uvd, uv + duv.xy);

    return min_uvd.xy;
}

FragmentOutput FragInitialize(VaryingsDefault i)
{
    half2 m = SAMPLE_TEX2D(_CameraMotionVectorsTexture, i.texcoord).xy;
    half d = LinearizeDepth(SAMPLE_DEPTH(_CameraDepthTexture, i.texcoord));

    FragmentOutput o;
    o.uvRemap = half4(i.texcoord, d < 0.9, 1);
    o.moDepth = half4(m, d, 0);
    return o;
}

FragmentOutput FragUpdate(VaryingsDefault i)
{
    float2 uv1 = i.texcoord;
    half2 m1 = SAMPLE_TEX2D(_CameraMotionVectorsTexture, uv1).xy;
    half d1 = LinearizeDepth(SAMPLE_DEPTH(_CameraDepthTexture, uv1));

    float2 uv0 = uv1 - m1;
    half3 md0 = SAMPLE_TEX2D(_PrevMoDepth, uv0).xyz;
    half4 c0 = SAMPLE_TEX2D(_PrevUVRemap, uv0);

    // Disocclusion test
    float docc = abs(1 - d1 / md0.z) * 20;//_DepthWeight;

    // Out of screen test
    float oscr = any(uv0 < 0) + any(uv0 > 1);

    float alpha = 1 - saturate(docc + oscr);

    FragmentOutput o;
    o.uvRemap = half4(c0.xy, d1 < 0.9, min(c0.a, alpha));
    o.moDepth = half4(m1, d1, 0);
    return o;
}

half4 FragComposite(VaryingsDefault i) : SV_Target
{
    half4 remap = SAMPLE_TEX2D(_UVRemap, i.texcoord);
    half4 c = SAMPLE_TEX2D(_MainTex, remap.xy);
    c = lerp(SAMPLE_TEX2D(_MainTex,i.texcoord), c, remap.z);
    return c;
}

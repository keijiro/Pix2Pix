// Pix2Pix post-processing effect: Temporal reprojection shader
// https://github.com/keijiro/Pix2Pix

#include "PostProcessing/Shaders/StdLib.hlsl"

#define SAMPLE_TEX2D(name, uv) SAMPLE_TEXTURE2D(name, sampler##name, uv)
#define SAMPLE_DEPTH(name, uv) SAMPLE_DEPTH_TEXTURE(name, sampler##name, uv).x

TEXTURE2D_SAMPLER2D(_MainTex, sampler_MainTex);
TEXTURE2D_SAMPLER2D(_EdgeTex, sampler_EdgeTex);
TEXTURE2D_SAMPLER2D(_RemapTex, sampler_RemapTex);
TEXTURE2D_SAMPLER2D(_CameraDepthTexture, sampler_CameraDepthTexture);
TEXTURE2D_SAMPLER2D(_CameraMotionVectorsTexture, sampler_CameraMotionVectorsTexture);

half2 _EdgeParams;

struct FragmentOutput
{
    half4 color : SV_Target0;
    half4 remap : SV_Target1;
};

half DepthMask(float2 uv)
{
    const float epsilon = 1e-5;
    float d = SAMPLE_DEPTH(_CameraDepthTexture, uv);
#if defined(UNITY_REVERSED_Z)
    return d > epsilon;
#else
    return d < 1 - epsilon;
#endif
}

FragmentOutput Frag(VaryingsDefault i)
{
    half2 mv = SAMPLE_TEX2D(_CameraMotionVectorsTexture, i.texcoord).xy;
    half4 remap = SAMPLE_TEX2D(_RemapTex, i.texcoord - mv);

#ifdef PIX2PIX_RESET_REMAP
    remap = half4(i.texcoord - mv, remap.xy);
#endif

    half4 c1 = SAMPLE_TEX2D(_MainTex, i.texcoord);
    half4 c2 = SAMPLE_TEX2D(_MainTex, remap.zw);
    half edge = SAMPLE_TEX2D(_EdgeTex, i.texcoord).x;
    half mask = DepthMask(i.texcoord);

    FragmentOutput o;
    o.color = lerp(c1, c2, mask) * lerp(1, edge, _EdgeParams.y);
    o.remap = remap;
    return o;
}

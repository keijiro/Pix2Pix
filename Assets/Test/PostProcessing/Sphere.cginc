#include "UnityCG.cginc"
#include "UnityGBuffer.cginc"
#include "UnityStandardUtils.cginc"
#include "SimplexNoise3D.hlsl"

// Cube map shadow caster; Used to render point light shadows on platforms
// that lacks depth cube map support.
#if defined(SHADOWS_CUBE) && !defined(SHADOWS_CUBE_IN_DEPTH_TEX)
#define PASS_CUBE_SHADOWCASTER
#endif

float _Seed;
float4x4 _NonJitteredVP;
float4x4 _PreviousM;
float4x4 _PreviousVP;

// Vertex input attributes
struct Attributes
{
    float4 position : POSITION;
    float3 normal : NORMAL;
    float4 tangent : TANGENT;
};

// Fragment varyings
struct Varyings
{
    float4 position : SV_POSITION;

#if defined(PASS_CUBE_SHADOWCASTER)
    // Cube map shadow caster pass
    float3 shadow : TEXCOORD0;

#elif defined(UNITY_PASS_SHADOWCASTER)
    // Default shadow caster pass

#elif defined(PASS_MOTIONVECTORS)
    // Motion vector pass
    float4 transfer0 : TEXCOORD0;
    float4 transfer1 : TEXCOORD1;

#else
    // GBuffer construction pass
    float3 normal : NORMAL;
    float4 tspace0 : TEXCOORD0;
    float4 tspace1 : TEXCOORD1;
    float4 tspace2 : TEXCOORD2;
    half3 ambient : TEXCOORD3;

#endif
};

//
// Vertex stage
//

void Vertex(inout Attributes input) {}

//
// Geometry stage
//

Varyings VertexOutput(float3 prev, float3 pos, half3 nrm, half4 otan)
{
    Varyings o;

    float3 wpos = mul(unity_ObjectToWorld, float4(pos, 1)).xyz;
    half3 wnrm = UnityObjectToWorldNormal(nrm);
    half4 wtan = half4(UnityObjectToWorldDir(otan.xyz), otan.w);

#if defined(PASS_CUBE_SHADOWCASTER)
    // Cube map shadow caster pass: Transfer the shadow vector.
    o.position = UnityWorldToClipPos(wpos);
    o.shadow = wpos - _LightPositionRange.xyz;

#elif defined(UNITY_PASS_SHADOWCASTER)
    // Default shadow caster pass: Apply the shadow bias.
    float scos = dot(wnrm, normalize(UnityWorldSpaceLightDir(wpos)));
    wpos -= wnrm * unity_LightShadowBias.z * sqrt(1 - scos * scos);
    o.position = UnityApplyLinearShadowBias(UnityWorldToClipPos(float4(wpos, 1)));

#elif defined(PASS_MOTIONVECTORS)
    // Motion vector pass
    o.position = UnityWorldToClipPos(wpos);
    o.transfer0 = mul(_PreviousVP, mul(_PreviousM, float4(prev, 1)));
    o.transfer1 = mul(_NonJitteredVP, float4(wpos, 1));

#else
    // GBuffer construction pass
    half3 bi = cross(wnrm, wtan) * wtan.w * unity_WorldTransformParams.w;
    o.position = UnityWorldToClipPos(float4(wpos, 1));
    o.normal = wnrm;
    o.tspace0 = float4(wtan.x, bi.x, wnrm.x, wpos.x);
    o.tspace1 = float4(wtan.y, bi.y, wnrm.y, wpos.y);
    o.tspace2 = float4(wtan.z, bi.z, wnrm.z, wpos.z);
    o.ambient = ShadeSHPerVertex(wnrm, 0);

#endif
    return o;
}

float3 ConstructNormal(float3 v1, float3 v2, float3 v3)
{
    return normalize(cross(v2 - v1, v3 - v1));
}

float3 Displace(float3 p, float t)
{
    float3 offs = float3(_Seed, 0, t * 1.4);
    float4 g = snoise_grad(p * 1.1 + offs);
    p *= (1 + 2 * g.w * g.w * g.w * g.w * g.w * g.w * g.w);
    p += g.xyz * 0.05;
    return p;
}

[maxvertexcount(3)]
void Geometry(
    triangle Attributes input[3], uint pid : SV_PrimitiveID,
    inout TriangleStream<Varyings> outStream
)
{
    float t0 = _Time.y - 1.0 / 60;
    float t1 = _Time.y;

    float3 v0 = input[0].position.xyz;
    float3 v1 = input[1].position.xyz;
    float3 v2 = input[2].position.xyz;

    float3 pv0 = Displace(v0, t0);
    float3 pv1 = Displace(v1, t0);
    float3 pv2 = Displace(v2, t0);

    v0 = Displace(v0, t1);
    v1 = Displace(v1, t1);
    v2 = Displace(v2, t1);

    float3 n = ConstructNormal(v0, v1, v2);
    outStream.Append(VertexOutput(pv0, v0, n, input[0].tangent));
    outStream.Append(VertexOutput(pv1, v1, n, input[1].tangent));
    outStream.Append(VertexOutput(pv2, v2, n, input[2].tangent));
    outStream.RestartStrip();
}

//
// Fragment phase
//

#if defined(PASS_CUBE_SHADOWCASTER)

// Cube map shadow caster pass
half4 Fragment(Varyings input) : SV_Target
{
    float depth = length(input.shadow) + unity_LightShadowBias.x;
    return UnityEncodeCubeShadowDepth(depth * _LightPositionRange.w);
}

#elif defined(UNITY_PASS_SHADOWCASTER)

// Default shadow caster pass
half4 Fragment() : SV_Target { return 0; }

#elif defined(PASS_MOTIONVECTORS)

// Motion vector pass
half4 Fragment(Varyings input) : SV_Target
{
    float3 hp0 = input.transfer0.xyz / input.transfer0.w;
    float3 hp1 = input.transfer1.xyz / input.transfer1.w;

    float2 vp0 = (hp0.xy + 1) / 2;
    float2 vp1 = (hp1.xy + 1) / 2;

#if UNITY_UV_STARTS_AT_TOP
    vp0.y = 1 - vp0.y;
    vp1.y = 1 - vp1.y;
#endif

    return half4(vp1 - vp0, 0, 1);
}

#else

// GBuffer construction pass
void Fragment(
    Varyings input,
    out half4 outGBuffer0 : SV_Target0,
    out half4 outGBuffer1 : SV_Target1,
    out half4 outGBuffer2 : SV_Target2,
    out half4 outEmission : SV_Target3
)
{
    half3 normal = half3(0, 0, 1);

    // Tangent space conversion (tangent space normal -> world space normal)
    float3 wn = normalize(float3(
        dot(input.tspace0.xyz, normal),
        dot(input.tspace1.xyz, normal),
        dot(input.tspace2.xyz, normal)
    ));

    // Update the GBuffer.
    UnityStandardData data;
    data.diffuseColor = 1;
    data.occlusion = 1;
    data.specularColor = 0;
    data.smoothness = 0;
    data.normalWorld = wn;
    UnityStandardDataToGbuffer(data, outGBuffer0, outGBuffer1, outGBuffer2);

    // Calculate ambient lighting and output to the emission buffer.
    float3 wp = float3(input.tspace0.w, input.tspace1.w, input.tspace2.w);
    half3 sh = ShadeSHPerPixel(data.normalWorld, input.ambient, wp);
    outEmission = half4(sh, 1);
}

#endif

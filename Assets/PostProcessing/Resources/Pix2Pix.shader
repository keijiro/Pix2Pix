Shader "Hidden/Pix2Pix/PostProcessing"
{
    HLSLINCLUDE

    #include "PostProcessing/Shaders/StdLib.hlsl"
    #include "PostProcessing/Shaders/Colors.hlsl"

    TEXTURE2D_SAMPLER2D(_MainTex, sampler_MainTex);
    TEXTURE2D_SAMPLER2D(_EdgeTex, sampler_EdgeTex);

    half2 _EdgeParams;

    half4 FragEdge(VaryingsDefault i) : SV_Target
    {
        float2 uv = i.texcoord;
        uv = (floor(uv * 256) + 0.5) / 256;
        float3 duv = float3(1, 1, 0) / 256;

        // Source color
        half4 c0 = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, uv);

        // Four sample points of the roberts cross operator
        float2 uv0 = uv;          // TL
        float2 uv1 = uv + duv.xy; // BR
        float2 uv2 = uv + duv.xz; // TR
        float2 uv3 = uv + duv.zy; // BL

        // Color samples
        half3 c1 = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, uv1).rgb;
        half3 c2 = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, uv2).rgb;
        half3 c3 = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, uv3).rgb;

        // Roberts cross operator
        half3 g1 = c1 - c0.rgb;
        half3 g2 = c3 - c2;
        half g = sqrt(dot(g1, g1) + dot(g2, g2));

        // Contrast
        half thresh = _EdgeParams.x / 2;
        half edge = smoothstep(thresh, 1 - thresh, g);

        return 1 - edge;
    }

    half4 FragCompo(VaryingsDefault i) : SV_Target
    {
        half4 c0 = SAMPLE_TEXTURE2D(_MainTex, sampler_MainTex, i.texcoord);
        half4 c1 = SAMPLE_TEXTURE2D(_EdgeTex, sampler_EdgeTex, i.texcoord);
        return c0 * lerp(1, c1, _EdgeParams.y);
    }

    ENDHLSL
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            HLSLPROGRAM
            #pragma vertex VertDefault
            #pragma fragment FragEdge
            ENDHLSL
        }
        Pass
        {
            HLSLPROGRAM
            #pragma vertex VertDefault
            #pragma fragment FragCompo
            ENDHLSL
        }
    }
}

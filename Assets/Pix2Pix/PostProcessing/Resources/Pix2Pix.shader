Shader "Hidden/Pix2Pix/PostProcessing"
{
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            HLSLPROGRAM
            #pragma vertex VertDefault
            #pragma fragment Frag
            #include "EdgeDetection.hlsl"
            ENDHLSL
        }
        Pass
        {
            HLSLPROGRAM
            #pragma vertex VertDefault
            #pragma fragment Frag
            #define PIX2PIX_RESET_REMAP
            #include "TemporalReprojection.hlsl"
            ENDHLSL
        }
        Pass
        {
            HLSLPROGRAM
            #pragma vertex VertDefault
            #pragma fragment Frag
            #include "TemporalReprojection.hlsl"
            ENDHLSL
        }
    }
}

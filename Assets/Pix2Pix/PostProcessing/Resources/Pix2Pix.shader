Shader "Hidden/Pix2Pix/PostProcessing"
{
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            HLSLPROGRAM
            #pragma vertex VertDefault
            #pragma fragment FragEdge
            #include "EdgeDetection.hlsl"
            ENDHLSL
        }
        Pass
        {
            HLSLPROGRAM
            #pragma vertex VertDefault
            #pragma fragment FragInitialize
            #include "TemporalReprojection.hlsl"
            ENDHLSL
        }
        Pass
        {
            HLSLPROGRAM
            #pragma vertex VertDefault
            #pragma fragment FragUpdate
            #include "TemporalReprojection.hlsl"
            ENDHLSL
        }
    }
}

Shader "Pix2Pix/Sphere"
{
    Properties
    {
        _Seed("Seed", Float) = 0
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        Pass
        {
            Tags { "LightMode"="Deferred" }
            CGPROGRAM
            #pragma target 4.0
            #pragma vertex Vertex
            #pragma geometry Geometry
            #pragma fragment Fragment
            #pragma multi_compile_prepassfinal noshadowmask nodynlightmap nodirlightmap nolightmap
            #include "Sphere.cginc"
            ENDCG
        }
        Pass
        {
            Tags { "LightMode"="ShadowCaster" }
            CGPROGRAM
            #pragma target 4.0
            #pragma vertex Vertex
            #pragma geometry Geometry
            #pragma fragment Fragment
            #pragma multi_compile_shadowcaster noshadowmask nodynlightmap nodirlightmap nolightmap
            #include "Sphere.cginc"
            ENDCG
        }
        Pass
        {
            Tags { "LightMode" = "MotionVectors" }
            ZWrite Off
            CGPROGRAM
            #pragma target 4.0
            #pragma vertex Vertex
            #pragma geometry Geometry
            #pragma fragment Fragment
            #define PASS_MOTIONVECTORS
            #include "Sphere.cginc"
            ENDCG
        }
    }
}

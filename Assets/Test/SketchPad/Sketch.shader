Shader "Hidden/Pix2Pix/Sketch"
{
    Properties
    {
        _Color("", Color) = (0, 0, 0, 0)
    }
    SubShader
    {
        Pass
        {
            Cull Off
            ZTest Always
            ZWrite Off

            CGPROGRAM

            #pragma vertex Vertex
            #pragma fragment Fragment

            #include "UnityCG.cginc"

            fixed4 _Color;

            float4 Vertex(float4 position : POSITION) : SV_Position
            {
                return position;
            }

            fixed4 Fragment(float4 position : SV_Position) : SV_Target
            {
                return _Color;
            }

            ENDCG
        }
    }
}

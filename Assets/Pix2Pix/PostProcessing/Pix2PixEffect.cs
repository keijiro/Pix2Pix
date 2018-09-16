using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.PostProcessing;

namespace Pix2Pix.PostProcessing
{
    #region Effect settings

    [System.Serializable]
    [PostProcess(typeof(Pix2PixRenderer), PostProcessEvent.BeforeStack, "Pix2Pix")]
    public sealed class Pix2PixEffect : PostProcessEffectSettings
    {
        [Range(0, 2)] public FloatParameter edgeThreshold = new FloatParameter { value = 0.8f };
        [Range(0, 1)] public FloatParameter edgeOpacity = new FloatParameter { value = 0 };
    }

    #endregion

    #region Effect renderer

    sealed class Pix2PixRenderer : PostProcessEffectRenderer<Pix2PixEffect>
    {
        static class ShaderIDs
        {
            internal static readonly int EdgeParams = Shader.PropertyToID("_EdgeParams");
            internal static readonly int UVRemap      = Shader.PropertyToID("_UVRemap");
            internal static readonly int PrevUVRemap  = Shader.PropertyToID("_PrevUVRemap");
            internal static readonly int PrevMoDepth  = Shader.PropertyToID("_PrevMoDepth");
            internal static readonly int DeltaTime    = Shader.PropertyToID("_DeltaTime");
        }

        Dictionary<string, Tensor> _weightTable;
        Generator _generator;

        Shader _shader;
        RenderTargetIdentifier[] _mrt = new RenderTargetIdentifier[2];

        RenderTexture _source;
        RenderTexture _result;

        RenderTexture _prevUVRemap;
        RenderTexture _prevMoDepth;

        float _prevDeltaTime;
        int _frameCount;

        public override void Init()
        {
            base.Init();

            var filePath = System.IO.Path.Combine
                (Application.streamingAssetsPath, "edges2cats_AtoB.pict");

            _shader = Shader.Find("Hidden/Pix2Pix/PostProcessing");

            _weightTable = WeightReader.ReadFromFile(filePath);
            _generator = new Generator(_weightTable);

            _source = new RenderTexture(256, 256, 0);
            _result = new RenderTexture(256, 256, 0);
            _result.enableRandomWrite = true;

            _source.hideFlags = HideFlags.DontSave;
            _result.hideFlags = HideFlags.DontSave;

            _source.Create();
            _result.Create();
        }

        public override void Release()
        {
            if (_generator != null)
            {
                _generator.Dispose();
                _generator = null;
            }

            if (_weightTable != null)
            {
                WeightReader.DisposeTable(_weightTable);
                _weightTable = null;
            }

            RuntimeUtilities.Destroy(_source);
            RuntimeUtilities.Destroy(_result);

            if (_prevUVRemap != null) RenderTexture.ReleaseTemporary(_prevUVRemap);
            if (_prevMoDepth != null) RenderTexture.ReleaseTemporary(_prevMoDepth);

            base.Release();
        }

        public override DepthTextureMode GetCameraFlags()
        {
            return DepthTextureMode.MotionVectors | DepthTextureMode.Depth;
        }

        public override void Render(PostProcessRenderContext context)
        {
            var cmd = context.command;
            cmd.BeginSample("Pix2Pix");

            var sheet = context.propertySheets.Get(_shader);
            var props = sheet.properties;

            // Edge detection pass
            props.SetVector(
                ShaderIDs.EdgeParams,
                new Vector2(settings.edgeThreshold, settings.edgeOpacity)
            );

            cmd.BlitFullscreenTriangle(context.source, _source, sheet, 0);

            // Pix2Pix generator pass
            var budget = Application.isPlaying ? 180 : 1200;
            var update = false;

            for (var cost = 0.0f; cost < budget;)
            {
                if (!_generator.Running) _generator.Start(_source);

                cost += _generator.Step();

                if (!_generator.Running)
                {
                    _generator.GetResult(_result);
                    update = true;
                }
            }

            // Temporal reprojection pass
            if (_prevUVRemap != null) props.SetTexture(ShaderIDs.PrevUVRemap, _prevUVRemap);
            if (_prevMoDepth != null) props.SetTexture(ShaderIDs.PrevMoDepth, _prevMoDepth);

            props.SetVector(
                ShaderIDs.DeltaTime,
                new Vector2(Time.deltaTime, _prevDeltaTime)
            );

            var uvRemap = RenderTexture.GetTemporary
                (context.width, context.height, 0, RenderTextureFormat.ARGBHalf);
            var moDepth = RenderTexture.GetTemporary
                (context.width, context.height, 0, RenderTextureFormat.ARGBHalf);

            _mrt[0] = uvRemap.colorBuffer;
            _mrt[1] = moDepth.colorBuffer;

            cmd.BlitFullscreenTriangle
                (context.source, _mrt, uvRemap.depthBuffer, sheet, update ? 1 : 2);

            // Composition pass
            props.SetTexture(ShaderIDs.UVRemap, uvRemap);
            cmd.BlitFullscreenTriangle(_result, context.destination, sheet, 3);

            // Discard the previous frame state.
            if (_prevUVRemap != null) RenderTexture.ReleaseTemporary(_prevUVRemap);
            if (_prevMoDepth != null) RenderTexture.ReleaseTemporary(_prevMoDepth);

            // Update the internal state.
            _prevUVRemap = uvRemap;
            _prevMoDepth = moDepth;
            _prevDeltaTime = Time.deltaTime;

            cmd.EndSample("Pix2Pix");
        }
    }

    #endregion
}

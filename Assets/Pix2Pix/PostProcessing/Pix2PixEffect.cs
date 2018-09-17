// Pix2Pix post-processing effect
// https://github.com/keijiro/Pix2Pix

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
            internal static readonly int EdgeTex = Shader.PropertyToID("_EdgeTex");
            internal static readonly int RemapTex = Shader.PropertyToID("_RemapTex");
        }

        Dictionary<string, Tensor> _weightTable;
        Generator _generator;

        Shader _shader;
        RenderTargetIdentifier[] _mrt = new RenderTargetIdentifier[2];

        RenderTexture _source;
        RenderTexture _result;
        RenderTexture _remap;

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

            if (_remap != null) RenderTexture.ReleaseTemporary(_remap);

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
                if (!_generator.Running)
                {
                    _generator.Start(_source);
                    update = true;
                }

                cost += _generator.Step();

                if (!_generator.Running) _generator.GetResult(_result);
            }

            // Temporal reprojection pass
            props.SetTexture(ShaderIDs.EdgeTex, _source);
            if (_remap != null) props.SetTexture(ShaderIDs.RemapTex, _remap);

            var newRemap = RenderTexture.GetTemporary
                (context.width, context.height, 0, RenderTextureFormat.ARGBHalf);

            _mrt[0] = context.destination;
            _mrt[1] = newRemap.colorBuffer;

            cmd.BlitFullscreenTriangle
                (_result, _mrt, newRemap.depthBuffer, sheet, update ? 1 : 2);

            // Update the internal state.
            if (_remap != null) RenderTexture.ReleaseTemporary(_remap);
            _remap = newRemap;

            cmd.EndSample("Pix2Pix");
        }
    }

    #endregion
}

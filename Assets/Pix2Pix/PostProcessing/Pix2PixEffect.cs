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
        [Range(0, 2)] public FloatParameter edgeThreshold = new FloatParameter { value = 0.5f };
        [Range(0, 1)] public FloatParameter edgeIntensity = new FloatParameter { value = 0.5f };
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
        }

        Dictionary<string, Tensor> _weightTable;
        Generator _generator;

        Shader _shader;

        RenderTexture _sourceRT;
        RenderTexture _resultRT;

        public override void Init()
        {
            base.Init();

            var filePath = System.IO.Path.Combine
                (Application.streamingAssetsPath, "edges2cats_AtoB.pict");

            _shader = Shader.Find("Hidden/Pix2Pix/PostProcessing");

            _weightTable = WeightReader.ReadFromFile(filePath);
            _generator = new Generator(_weightTable);

            _sourceRT = new RenderTexture(256, 256, 0);
            _resultRT = new RenderTexture(256, 256, 0);
            _resultRT.enableRandomWrite = true;

            _sourceRT.hideFlags = HideFlags.DontSave;
            _resultRT.hideFlags = HideFlags.DontSave;

            _sourceRT.Create();
            _resultRT.Create();
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

            RuntimeUtilities.Destroy(_sourceRT);
            RuntimeUtilities.Destroy(_resultRT);

            base.Release();
        }

        public override void Render(PostProcessRenderContext context)
        {
            var cmd = context.command;
            cmd.BeginSample("Pix2Pix");

            var sheet = context.propertySheets.Get(_shader);
            var props = sheet.properties;

            props.SetVector(ShaderIDs.EdgeParams, new Vector3(
                settings.edgeThreshold,
                settings.edgeIntensity,
                settings.edgeOpacity
            ));

            // Edge detection pass
            cmd.BlitFullscreenTriangle(context.source, _sourceRT, sheet, 0);

            // Pix2Pix generator pass
            GpuBackend.UseCommandBuffer(cmd);
            _generator.Start(_sourceRT);
            _generator.Step();
            while (_generator.Running) _generator.Step();
            _generator.GetResult(_resultRT);
            GpuBackend.ResetToDefaultCommandBuffer();

            // Composite pass
            props.SetTexture(ShaderIDs.EdgeTex, _sourceRT);
            cmd.BlitFullscreenTriangle(_resultRT, context.destination, sheet, 1);

            cmd.EndSample("Pix2Pix");
        }
    }

    #endregion
}

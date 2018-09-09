using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering.PostProcessing;

namespace Pix2Pix.PostProcessing
{
    #region Effect settings

    [System.Serializable]
    [PostProcess(typeof(Pix2PixRenderer), PostProcessEvent.BeforeStack, "Pix2Pix")]
    public sealed class Pix2Pix : PostProcessEffectSettings
    {
        [Range(0, 1)] public FloatParameter edgeContrast = new FloatParameter { value = 0.5f };
        [Range(0, 1)] public FloatParameter edgeOpacity = new FloatParameter { value = 0 };
    }

    #endregion

    #region Effect renderer

    sealed class Pix2PixRenderer : PostProcessEffectRenderer<Pix2Pix>
    {
        static class ShaderIDs
        {
            internal static readonly int EdgeParams = Shader.PropertyToID("_EdgeParams");
            internal static readonly int EdgeTex = Shader.PropertyToID("_EdgeTex");
        }

        Dictionary<string, Tensor> _weightTable;
        Generator _generator;

        RenderTexture _source;
        RenderTexture _result;

        public override void Init()
        {
            base.Init();

            var filePath = System.IO.Path.Combine(Application.streamingAssetsPath, "edges2cats_AtoB.pict");
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

            base.Release();
        }

        public override void Render(PostProcessRenderContext context)
        {
            var cmd = context.command;
            cmd.BeginSample("Pix2Pix");

            var sheet = context.propertySheets.Get(Shader.Find("Hidden/Pix2Pix/PostProcessing"));

            sheet.properties.SetVector(ShaderIDs.EdgeParams, new Vector2(
                settings.edgeContrast, settings.edgeOpacity
            ));
            sheet.properties.SetTexture(ShaderIDs.EdgeTex, _source);

            cmd.BlitFullscreenTriangle(context.source, _source, sheet, 0);
            cmd.BlitFullscreenTriangle(_result, context.destination, sheet, 1);

            for (var cost = 0.0f; cost < (Application.isPlaying ? 400 : 1200);)
            {
                if (!_generator.Running) _generator.Start(_source);
                cost += _generator.Step();
                if (!_generator.Running) _generator.GetResult(_result);
            }

            cmd.EndSample("Pix2Pix");
        }
    }

    #endregion
}

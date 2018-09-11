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
            internal static readonly int PrevTex = Shader.PropertyToID("_PrevTex");

            internal static readonly int UVRemap      = Shader.PropertyToID("_UVRemap");
            internal static readonly int PrevUVRemap  = Shader.PropertyToID("_PrevUVRemap");
            internal static readonly int PrevMoDepth  = Shader.PropertyToID("_PrevMoDepth");
            internal static readonly int DeltaTime    = Shader.PropertyToID("_DeltaTime");
        }

        Dictionary<string, Tensor> _weightTable;
        Generator _generator;

        RenderTexture _source;
        RenderTexture _result;
        RenderTexture _history;

        RenderTexture _lastFrame;
        RenderTexture _prevUVRemap;
        RenderTexture _prevMoDepth;

        RenderTargetIdentifier[] _mrt = new RenderTargetIdentifier[2];

        float _prevDeltaTime;
        int _frameCount;

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

            if (_history != null) RenderTexture.ReleaseTemporary(_history);

            if (_lastFrame != null) RenderTexture.ReleaseTemporary(_lastFrame);
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

            var sheet = context.propertySheets.Get(Shader.Find("Hidden/Pix2Pix/PostProcessing"));
            var reset = false;

            sheet.properties.SetVector(ShaderIDs.EdgeParams, new Vector2(
                settings.edgeThreshold, settings.edgeOpacity
            ));
            cmd.BlitFullscreenTriangle(context.source, _source, sheet, 0);

            for (var cost = 0.0f; cost < (Application.isPlaying ? 180 : 1200);)
            {
                if (!_generator.Running) _generator.Start(_source);
                cost += _generator.Step();
                if (!_generator.Running)
                {
                    _generator.GetResult(_result);
                    reset = true;
                }
            }

            // Allocate RTs for storing the next frame state.
            var uvRemap = RenderTexture.GetTemporary(context.width, context.height, 0, RenderTextureFormat.ARGBHalf);
            var moDepth = RenderTexture.GetTemporary(context.width, context.height, 0, RenderTextureFormat.ARGBHalf);
            _mrt[0] = uvRemap.colorBuffer;
            _mrt[1] = moDepth.colorBuffer;

            // Set the shader uniforms.
            sheet = context.propertySheets.Get(Shader.Find("Hidden/Pix2Pix/TemporalReprojection"));
            if (_prevUVRemap != null) sheet.properties.SetTexture(ShaderIDs.PrevUVRemap, _prevUVRemap);
            if (_prevMoDepth != null) sheet.properties.SetTexture(ShaderIDs.PrevMoDepth, _prevMoDepth);
            sheet.properties.SetVector(ShaderIDs.DeltaTime, new Vector2(Time.deltaTime, _prevDeltaTime));

            if (reset)
            {
                // Update the last frame store.
                if (_lastFrame != null) RenderTexture.ReleaseTemporary(_lastFrame);
                _lastFrame = RenderTexture.GetTemporary(context.width, context.height, 0, RenderTextureFormat.ARGBHalf);
                cmd.BlitFullscreenTriangle(_result, _lastFrame);

                // Reset pass
                cmd.BlitFullscreenTriangle(context.source, _mrt, uvRemap.depthBuffer, sheet, 0);
            }
            else
            {
                // Temporal reprojection pass
                cmd.BlitFullscreenTriangle(context.source, _mrt, uvRemap.depthBuffer, sheet, 1);
            }

        // Second pass: Composition
        var newHistory = RenderTexture.GetTemporary(context.width, context.height, 0, RenderTextureFormat.ARGBHalf);
        if (_history != null) sheet.properties.SetTexture(ShaderIDs.PrevTex, _history);
        sheet.properties.SetTexture(ShaderIDs.UVRemap, uvRemap);
        cmd.BlitFullscreenTriangle(_lastFrame, newHistory, sheet, 2);
        cmd.BlitFullscreenTriangle(newHistory, context.destination);

        if (_history != null) RenderTexture.ReleaseTemporary(_history);
        _history = newHistory;

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

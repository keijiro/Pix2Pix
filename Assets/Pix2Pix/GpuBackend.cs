// GPU backend interface
// https://github.com/keijiro/Pix2Pix

using UnityEngine;
using UnityEngine.Rendering;
using System.Collections.Generic;

namespace Pix2Pix
{
    [ExecuteInEditMode]
    sealed class GpuBackend : MonoBehaviour
    {
        #region MonoBehaviour implementation as a singleton-like class

        [SerializeField, HideInInspector] ComputeAssets _computeAssets = null;

        static GpuBackend _instance;

        void OnEnable()
        {
            _instance = this;
        }

        void OnDisable()
        {
            ReleaseBufferPool();
            if (_commandBuffer != null) _commandBuffer.Dispose();
            _instance = null;
        }

        #endregion

        #region Command buffer management

        CommandBuffer _commandBuffer;
        CommandBuffer _overrideCommandBuffer;

        public static CommandBuffer SharedCommandBuffer {
            get {
                if (_instance._overrideCommandBuffer != null)
                    return _instance._overrideCommandBuffer;

                if (_instance._commandBuffer == null)
                    _instance._commandBuffer = new CommandBuffer();

                return _instance._commandBuffer;
            }
        }

        public static void ClearCommandBuffer()
        {
            SharedCommandBuffer.Clear();
        }

        public static void ExecuteCommandBuffer()
        {
            Graphics.ExecuteCommandBuffer(SharedCommandBuffer);
        }

        public static void ExecuteAndClearCommandBuffer()
        {
            ExecuteCommandBuffer();
            ClearCommandBuffer();
        }

        public static void UseCommandBuffer(CommandBuffer buffer)
        {
            _instance._overrideCommandBuffer = buffer;
        }

        public static void ResetToDefaultCommandBuffer()
        {
            _instance._overrideCommandBuffer = null;
        }

        #endregion

        #region Compute buffer management

        List<ComputeBuffer> _bufferPool = new List<ComputeBuffer>();

        internal static ComputeBuffer AllocateBuffer(int size)
        {
            foreach (var buffer in _instance._bufferPool)
            {
                if (buffer.count == size)
                {
                    _instance._bufferPool.Remove(buffer);
                    return buffer;
                }
            }
            return new ComputeBuffer(size, sizeof(float));
        }

        internal static void ReleaseBuffer(ComputeBuffer buffer)
        {
            if (_instance != null)
                _instance._bufferPool.Add(buffer);
            else
                buffer.Dispose();
        }

        void ReleaseBufferPool()
        {
            foreach (var buffer in _bufferPool) buffer.Dispose();
            _bufferPool.Clear();
        }

        #endregion

        #region Compute kernel invocation methods
        
        internal static void InvokeActivation
            (string name, Tensor input, float alpha, Tensor output)
        {
            Debug.Assert(input.Shape[0] == output.Shape[0]);
            Debug.Assert(input.Shape[1] == output.Shape[1]);
            Debug.Assert(input.Shape[2] == output.Shape[2]);

            var compute = _instance._computeAssets.Activation;
            var kernel = compute.FindKernel(name);
            var threadCount = compute.GetThreadGroupSizeVector(kernel);

            var length = input.Buffer.count;
            Debug.Assert(length % threadCount.x == 0);

            var cb = SharedCommandBuffer;
            cb.SetComputeFloatParam(compute, "Alpha", alpha);
            cb.SetComputeBufferParam(compute, kernel, "Input" , input .Buffer);
            cb.SetComputeBufferParam(compute, kernel, "Output", output.Buffer);

            cb.DispatchCompute(compute, kernel, length / threadCount.x, 1, 1);
        }

        internal static void InvokeConcat
            (Tensor input1, Tensor input2, Tensor output)
        {
            var height   = input1.Shape[0];
            var width    = input1.Shape[1];
            var channels = input1.Shape[2];

            Debug.Assert(input2.Shape[0] == height);
            Debug.Assert(input2.Shape[1] == width);
            Debug.Assert(input2.Shape[2] == channels);

            Debug.Assert(output.Shape[0] == height);
            Debug.Assert(output.Shape[1] == width);
            Debug.Assert(output.Shape[2] == channels * 2);

            var compute = _instance._computeAssets.Concat;
            var kernel = compute.FindKernel("Concat");
            var threadCount = compute.GetThreadGroupSizeVector(kernel);

            Debug.Assert(channels % threadCount.x == 0);
            Debug.Assert((width * height) % threadCount.y == 0);

            var cb = SharedCommandBuffer;
            cb.SetComputeShapeAsIntParams(compute, "InputShape", input1.Shape);
            cb.SetComputeBufferParam(compute, kernel, "Input1", input1.Buffer);
            cb.SetComputeBufferParam(compute, kernel, "Input2", input2.Buffer);
            cb.SetComputeBufferParam(compute, kernel, "Output", output.Buffer);

            cb.DispatchCompute(
                compute, kernel,
                channels / threadCount.x, width * height / threadCount.y, 1
            );
        }

        internal static void InvokeBatchNorm
            (Tensor input, Tensor scale, Tensor offset, Tensor output)
        {
            Debug.Assert(input.Shape[0] == output.Shape[0]);
            Debug.Assert(input.Shape[1] == output.Shape[1]);
            Debug.Assert(input.Shape[2] == output.Shape[2]);

            var channels = input.Shape[2];
            var elements = input.Buffer.count / channels;
            var nestable = (elements % 16 == 0);

            Debug.Assert(channels == scale .Buffer.count);
            Debug.Assert(channels == offset.Buffer.count);

            var compute = _instance._computeAssets.BatchNorm;
            var kernel = compute.FindKernel(nestable ? "BatchNormNested" : "BatchNorm");
            var threadCount = compute.GetThreadGroupSizeVector(kernel);

            Debug.Assert(channels % threadCount.x == 0);

            var cb = SharedCommandBuffer;
            cb.SetComputeShapeAsIntParams(compute, "InputShape", input.Shape);
            cb.SetComputeBufferParam(compute, kernel, "Input" , input .Buffer);
            cb.SetComputeBufferParam(compute, kernel, "Scale" , scale .Buffer);
            cb.SetComputeBufferParam(compute, kernel, "Offset", offset.Buffer);
            cb.SetComputeBufferParam(compute, kernel, "Output", output.Buffer);

            cb.DispatchCompute(compute, kernel, channels / threadCount.x, 1, 1);
        }

        static string GetConv2DKernelName(int outChannels)
        {
            switch (outChannels)
            {
                case 64  : return "Conv2D_64";
                case 128 : return "Conv2D_128";
                case 256 : return "Conv2D_256";
                case 512 : return "Conv2D_512";
            }
            Debug.LogError("Invalid channel count.");
            return "";
        }

        static string GetDeconv2DKernelName(int outChannels)
        {
            switch (outChannels)
            {
                case 3   : return "TransConv2D_final";
                case 64  : return "TransConv2D_64";
                case 128 : return "TransConv2D_128";
                case 256 : return "TransConv2D_256";
                case 512 : return "TransConv2D_512";
            }
            Debug.LogError("Invalid channel count.");
            return "";
        }

        internal static void InvokeConv2D
            (Tensor input, Tensor filter, Tensor bias, Tensor output)
        {
            Debug.Assert(output.Shape[0] == input.Shape[0] / 2);
            Debug.Assert(output.Shape[1] == input.Shape[1] / 2);

            var outChannels = filter.Shape[3];
            Debug.Assert(output.Shape[2] == outChannels);

            var kernelName = GetConv2DKernelName(outChannels);
            InvokeConv2DInternal(kernelName, input, filter, bias, output);
        }

        internal static void InvokeDeconv2D
            (Tensor input, Tensor filter, Tensor bias, Tensor output)
        {
            Debug.Assert(output.Shape[0] == input.Shape[0] * 2);
            Debug.Assert(output.Shape[1] == input.Shape[1] * 2);

            var outChannels = filter.Shape[3];
            Debug.Assert(output.Shape[2] == outChannels);

            var kernelName = GetDeconv2DKernelName(outChannels);
            InvokeConv2DInternal(kernelName, input, filter, bias, output);
        }

        static void InvokeConv2DInternal
            (string kernelName, Tensor input, Tensor filter, Tensor bias, Tensor output)
        {
            Debug.Assert(filter.Shape[0] == 4);
            Debug.Assert(filter.Shape[1] == 4);
            Debug.Assert(filter.Shape[2] == input.Shape[2]);

            var outChannels = filter.Shape[3];
            Debug.Assert(bias.Shape[0] == outChannels);

            var compute = _instance._computeAssets.Convolution;
            var kernel = compute.FindKernel(kernelName);
            var threadCount = compute.GetThreadGroupSizeVector(kernel);

            Debug.Assert(outChannels % threadCount.x == 0 || outChannels == 3);

            var cb = SharedCommandBuffer;
            cb.SetComputeShapeAsIntParams(compute,  "InputShape", input .Shape);
            cb.SetComputeShapeAsIntParams(compute, "FilterShape", filter.Shape);
            cb.SetComputeShapeAsIntParams(compute, "OutputShape", output.Shape);

            cb.SetComputeIntParams(compute,  "InputIndexer", GpuBackendHelper.Indexer(input .Shape));
            cb.SetComputeIntParams(compute, "FilterIndexer", GpuBackendHelper.Indexer(filter.Shape));
            cb.SetComputeIntParams(compute, "OutputIndexer", GpuBackendHelper.Indexer(output.Shape));

            cb.SetComputeBufferParam(compute, kernel, "Input" , input .Buffer);
            cb.SetComputeBufferParam(compute, kernel, "Filter", filter.Buffer);
            cb.SetComputeBufferParam(compute, kernel, "Bias"  , bias  .Buffer);
            cb.SetComputeBufferParam(compute, kernel, "Output", output.Buffer);

            var groupCount = Mathf.Max(1, outChannels / threadCount.x);
            cb.DispatchCompute(compute, kernel, groupCount, output.Shape[1], output.Shape[0]);
        }

        internal static void InvokeReorderWeights(Tensor input, Tensor output)
        {
            var compute = _instance._computeAssets.Setup;
            var kernel = compute.FindKernel("ReorderWeights");

            Debug.Assert(input.Shape[0] == output.Shape[0]);
            Debug.Assert(input.Shape[1] == output.Shape[1]);
            Debug.Assert(input.Shape[2] == output.Shape[3]);
            Debug.Assert(input.Shape[3] == output.Shape[2]);

            var cb = SharedCommandBuffer;
            cb.SetComputeShapeAsIntParams(compute, "InputShape" , input .Shape);
            cb.SetComputeShapeAsIntParams(compute, "OutputShape", output.Shape);

            cb.SetComputeIntParams(compute, "InputIndexer" , GpuBackendHelper.Indexer(input .Shape));
            cb.SetComputeIntParams(compute, "OutputIndexer", GpuBackendHelper.Indexer(output.Shape));

            cb.SetComputeBufferParam(compute, kernel, "Input" , input .Buffer);
            cb.SetComputeBufferParam(compute, kernel, "Output", output.Buffer);

            cb.DispatchCompute(compute, kernel, input.Shape[0], input.Shape[1], 1);
        }

        internal static void InvokeImageToTensor(Texture input, Tensor output)
        {
            Debug.Assert(output.Shape[0] == input.height);
            Debug.Assert(output.Shape[1] == input.width);
            Debug.Assert(output.Shape[2] == 3);

            var compute = _instance._computeAssets.Image;
            var kernel = compute.FindKernel("ImageToTensor");
            var threadCount = compute.GetThreadGroupSizeVector(kernel);

            Debug.Assert(input.width  % threadCount.x == 0);
            Debug.Assert(input.height % threadCount.y == 0);

            var cb = SharedCommandBuffer;
            cb.SetComputeShapeAsIntParams(compute, "Shape", output.Shape);
            cb.SetComputeTextureParam(compute, kernel, "InputImage", input);
            cb.SetComputeBufferParam(compute, kernel, "OutputTensor", output.Buffer);

            cb.DispatchCompute(
                compute, kernel,
                input.width  / threadCount.x, input.height / threadCount.y, 1
            );
        }

        internal static void InvokeTensorToImage(Tensor input, RenderTexture output)
        {
            Debug.Assert(output.height == input.Shape[0]);
            Debug.Assert(output.width  == input.Shape[1]);
            Debug.Assert(input.Shape[2] == 3);

            var compute = _instance._computeAssets.Image;
            var kernel = compute.FindKernel("TensorToImage");
            var threadCount = compute.GetThreadGroupSizeVector(kernel);

            Debug.Assert(input.Shape[1] % threadCount.x == 0);
            Debug.Assert(input.Shape[0] % threadCount.y == 0);
            Debug.Assert(output.enableRandomWrite);

            var cb = SharedCommandBuffer;
            cb.SetComputeShapeAsIntParams(compute, "Shape", input.Shape);
            cb.SetComputeBufferParam(compute, kernel, "InputTensor", input.Buffer);
            cb.SetComputeTextureParam(compute, kernel, "OutputImage", output);

            cb.DispatchCompute(
                compute, kernel,
                output.width  / threadCount.x, output.height / threadCount.y, 1
            );
        }

        #endregion
    }

    #region Internal helper

    static class GpuBackendHelper
    {
        internal static Vector3Int
            GetThreadGroupSizeVector(this ComputeShader self, int kernel)
        {
            uint tgs_x, tgs_y, tgs_z;
            self.GetKernelThreadGroupSizes(kernel, out tgs_x, out tgs_y, out tgs_z);
            return new Vector3Int((int)tgs_x, (int)tgs_y, (int)tgs_z);
        }

        static int[] _tempArray = new int[4];

        public static void SetComputeShapeAsIntParams
            (this CommandBuffer cb, ComputeShader compute, string name, Shape shape)
        {
            _tempArray[0] = shape.Dim1;
            _tempArray[1] = shape.Dim2;
            _tempArray[2] = shape.Dim3;
            _tempArray[3] = shape.Dim4;
            cb.SetComputeIntParams(compute, name, _tempArray);
        }

        internal static int[] Indexer(Shape shape)
        {
            if (shape.Order == 4)
            {
                _tempArray[0] = shape[1] * shape[2] * shape[3];
                _tempArray[1] = shape[2] * shape[3];
                _tempArray[2] = shape[3];
                _tempArray[3] = 1;
            }
            else
            {
                _tempArray[0] = shape[1] * shape[2];
                _tempArray[1] = shape[2];
                _tempArray[2] = 1;
                _tempArray[3] = 0;
            }
            return _tempArray;
        }
    }

    #endregion
}

// GPU backend interface
// https://github.com/keijiro/Pix2Pix

using UnityEngine;
using System.Collections.Generic;

namespace Pix2Pix
{
    sealed class GpuBackend : MonoBehaviour
    {
        #region MonoBehaviour implementation as a singleton-like class

        [SerializeField, HideInInspector] ComputeAssets _computeAssets;

        static GpuBackend _instance;

        void OnEnable()
        {
            _instance = this;
        }

        void OnDestroy()
        {
            ReleaseAllBuffers();
        }

        #endregion

        #region Compute buffer management

        List<ComputeBuffer> _buffers = new List<ComputeBuffer>();

        internal static ComputeBuffer AllocateBuffer(int size)
        {
            var buffer = new ComputeBuffer(size, sizeof(float));
            _instance._buffers.Add(buffer);
            return buffer;
        }

        internal static void ReleaseBuffer(ComputeBuffer buffer)
        {
            _instance._buffers.Remove(buffer);
            buffer.Release();
        }

        void ReleaseAllBuffers()
        {
            foreach (var buffer in _buffers) buffer.Release();
            _buffers.Clear();
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

            compute.SetFloat("Alpha", alpha);
            compute.SetBuffer(kernel, "Input" , input .Buffer);
            compute.SetBuffer(kernel, "Output", output.Buffer);

            compute.Dispatch(kernel, length / threadCount.x, 1, 1);
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

            compute.SetShapeAsInts("InputShape", input1.Shape);
            compute.SetBuffer(kernel, "Input1", input1.Buffer);
            compute.SetBuffer(kernel, "Input2", input2.Buffer);
            compute.SetBuffer(kernel, "Output", output.Buffer);

            compute.Dispatch(
                kernel,
                channels / threadCount.x,
                width * height / threadCount.y,
                1
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

            compute.SetShapeAsInts("InputShape", input.Shape);
            compute.SetBuffer(kernel, "Input" , input .Buffer);
            compute.SetBuffer(kernel, "Scale" , scale .Buffer);
            compute.SetBuffer(kernel, "Offset", offset.Buffer);
            compute.SetBuffer(kernel, "Output", output.Buffer);

            compute.Dispatch(kernel, channels / threadCount.x, 1, 1);
        }

        internal static void InvokeConv2D
            (Tensor input, Tensor filter, Tensor bias, Tensor output)
        {
            Debug.Assert(output.Shape[0] == input.Shape[0] / 2);
            Debug.Assert(output.Shape[1] == input.Shape[1] / 2);

            var outChannels = filter.Shape[3];
            Debug.Assert(output.Shape[2] == outChannels);

            var kernelName = "Conv2D_" + outChannels;
            InvokeConv2DInternal(kernelName, input, filter, bias, output);
        }

        internal static void InvokeDeconv2D
            (Tensor input, Tensor filter, Tensor bias, Tensor output)
        {
            Debug.Assert(output.Shape[0] == input.Shape[0] * 2);
            Debug.Assert(output.Shape[1] == input.Shape[1] * 2);

            var outChannels = filter.Shape[3];
            Debug.Assert(output.Shape[2] == outChannels);

            var kernelName = "TransConv2D_" +
                (outChannels == 3 ? "final" : outChannels.ToString());
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

            compute.SetShapeAsInts( "InputShape", input .Shape);
            compute.SetShapeAsInts("FilterShape", filter.Shape);
            compute.SetShapeAsInts("OutputShape", output.Shape);

            compute.SetInts( "InputIndexer", GpuBackendHelper.Indexer(input .Shape));
            compute.SetInts("FilterIndexer", GpuBackendHelper.Indexer(filter.Shape));
            compute.SetInts("OutputIndexer", GpuBackendHelper.Indexer(output.Shape));

            compute.SetBuffer(kernel, "Input" , input .Buffer);
            compute.SetBuffer(kernel, "Filter", filter.Buffer);
            compute.SetBuffer(kernel, "Bias"  , bias  .Buffer);
            compute.SetBuffer(kernel, "Output", output.Buffer);

            var groupCount = Mathf.Max(1, outChannels / threadCount.x);
            compute.Dispatch(kernel, groupCount, output.Shape[1], output.Shape[0]);
        }

        internal static void InvokeReorderWeights(Tensor input, Tensor output)
        {
            var compute = _instance._computeAssets.Setup;
            var kernel = compute.FindKernel("ReorderWeights");

            Debug.Assert(input.Shape[0] == output.Shape[0]);
            Debug.Assert(input.Shape[1] == output.Shape[1]);
            Debug.Assert(input.Shape[2] == output.Shape[3]);
            Debug.Assert(input.Shape[3] == output.Shape[2]);

            compute.SetShapeAsInts("InputShape" , input .Shape);
            compute.SetShapeAsInts("OutputShape", output.Shape);

            compute.SetInts("InputIndexer" , GpuBackendHelper.Indexer(input .Shape));
            compute.SetInts("OutputIndexer", GpuBackendHelper.Indexer(output.Shape));

            compute.SetBuffer(kernel, "Input" , input .Buffer);
            compute.SetBuffer(kernel, "Output", output.Buffer);

            compute.Dispatch(kernel, input.Shape[0], input.Shape[1], 1);
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

            compute.SetShapeAsInts("Shape", output.Shape);
            compute.SetTexture(kernel, "InputImage", input);
            compute.SetBuffer(kernel, "OutputTensor", output.Buffer);

            compute.Dispatch(
                kernel,
                input.width  / threadCount.x,
                input.height / threadCount.y,
                1
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

            compute.SetShapeAsInts("Shape", input.Shape);
            compute.SetBuffer(kernel, "InputTensor", input.Buffer);
            compute.SetTexture(kernel, "OutputImage", output);

            compute.Dispatch(
                kernel,
                output.width  / threadCount.x,
                output.height / threadCount.y,
                1
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

        public static void SetShapeAsInts
            (this ComputeShader shader, string name, Shape shape)
        {
            _tempArray[0] = shape.Dim1;
            _tempArray[1] = shape.Dim2;
            _tempArray[2] = shape.Dim3;
            _tempArray[3] = shape.Dim4;
            shader.SetInts(name, _tempArray);
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

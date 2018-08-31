using UnityEngine;
using System.Collections.Generic;

namespace Pix2Pix
{
    static class GpuHelper
    {
        #region Compute buffer management

        static List<ComputeBuffer> _buffers = new List<ComputeBuffer>();

        public static ComputeBuffer AllocateBuffer(int size)
        {
            var buffer = new ComputeBuffer(size, sizeof(float));
            _buffers.Add(buffer);
            return buffer;
        }

        public static void ReleaseBuffer(ComputeBuffer buffer)
        {
            _buffers.Remove(buffer);
            buffer.Release();
        }

        public static void ReleaseAllBuffers()
        {
            foreach (var buffer in _buffers) buffer.Release();
            _buffers.Clear();
        }

        #endregion

        #region Compute kernel invocation methods
        
        public static void InvokeActivation(string name, Tensor input, Tensor output)
        {
            Debug.Assert(input.Shape[0] == output.Shape[0]);
            Debug.Assert(input.Shape[1] == output.Shape[1]);
            Debug.Assert(input.Shape[2] == output.Shape[2]);

            var compute = ComputeAssets.Activation;
            var kernel = compute.FindKernel(name);
            var threadCount = compute.GetKernelThreadGroupSizeVector(kernel);

            var length = input.Buffer.count;
            Debug.Assert(length % threadCount.x == 0);

            compute.SetBuffer(kernel, "Input" , input .Buffer);
            compute.SetBuffer(kernel, "Output", output.Buffer);
            compute.Dispatch(kernel, length / threadCount.x, 1, 1);
        }

        public static void InvokeConcat(Tensor input1, Tensor input2, Tensor output)
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

            var compute = ComputeAssets.Concat;
            var kernel = compute.FindKernel("Concat");
            var threadCount = compute.GetKernelThreadGroupSizeVector(kernel);

            Debug.Assert(channels % threadCount.x == 0);
            Debug.Assert((width * height) % threadCount.y == 0);

            compute.SetInts("InputShape", input1.Shape);
            compute.SetBuffer(kernel, "Input1", input1.Buffer);
            compute.SetBuffer(kernel, "Input2", input2.Buffer);
            compute.SetBuffer(kernel, "Output", output.Buffer);
            compute.Dispatch(kernel, channels / threadCount.x, width * height / threadCount.y, 1);
        }

        public static void InvokeBatchNorm(Tensor input, Tensor scale, Tensor offset, Tensor output)
        {
            Debug.Assert(input.Shape[0] == output.Shape[0]);
            Debug.Assert(input.Shape[1] == output.Shape[1]);
            Debug.Assert(input.Shape[2] == output.Shape[2]);

            var channels = input.Shape[2];
            var elements = input.Buffer.count / channels;
            var nestable = (elements % 16 == 0);

            Debug.Assert(channels == scale .Buffer.count);
            Debug.Assert(channels == offset.Buffer.count);

            var compute = ComputeAssets.BatchNorm;
            var kernel = compute.FindKernel(nestable ? "BatchNormNested" : "BatchNorm");
            var threadCount = compute.GetKernelThreadGroupSizeVector(kernel);

            Debug.Assert(channels % threadCount.x == 0);

            compute.SetInts("InputShape", input.Shape);
            compute.SetBuffer(kernel, "Input" , input .Buffer);
            compute.SetBuffer(kernel, "Scale" , scale .Buffer);
            compute.SetBuffer(kernel, "Offset", offset.Buffer);
            compute.SetBuffer(kernel, "Output", output.Buffer);
            compute.Dispatch(kernel, channels / threadCount.x, 1, 1);
        }

        public static void InvokeConv2D(Tensor input, Tensor filter, Tensor bias, Tensor output)
        {
            Debug.Assert(output.Shape[0] == input.Shape[0] / 2);
            Debug.Assert(output.Shape[1] == input.Shape[1] / 2);

            var outChannels = filter.Shape[3];
            Debug.Assert(output.Shape[2] == outChannels);

            var kernelName = "Conv2D_" + outChannels;
            InvokeConv2DInternal(kernelName, input, filter, bias, output);
        }

        public static void InvokeDeconv2D(Tensor input, Tensor filter, Tensor bias, Tensor output)
        {
            Debug.Assert(output.Shape[0] == input.Shape[0] * 2);
            Debug.Assert(output.Shape[1] == input.Shape[1] * 2);

            var outChannels = filter.Shape[3];
            Debug.Assert(output.Shape[2] == outChannels);

            var kernelName = "TransConv2D_" + (outChannels == 3 ? "final" : outChannels.ToString());
            InvokeConv2DInternal(kernelName, input, filter, bias, output);
        }

        static void InvokeConv2DInternal(
            string kernelName, Tensor input, Tensor filter, Tensor bias, Tensor output
        )
        {
            Debug.Assert(filter.Shape[0] == 4);
            Debug.Assert(filter.Shape[1] == 4);
            Debug.Assert(filter.Shape[2] == input.Shape[2]);

            var outChannels = filter.Shape[3];
            Debug.Assert(bias.Shape[0] == outChannels);

            var compute = ComputeAssets.Convolution;
            var kernel = compute.FindKernel(kernelName);
            var threadCount = compute.GetKernelThreadGroupSizeVector(kernel);

            Debug.Assert(outChannels % threadCount.x == 0 || outChannels == 3);

            compute.SetInts( "InputShape", input .Shape);
            compute.SetInts("FilterShape", filter.Shape);
            compute.SetInts("OutputShape", output.Shape);

            compute.SetInts( "InputIndexer", CalculateIndexer(input .Shape));
            compute.SetInts("FilterIndexer", CalculateIndexer(filter.Shape));
            compute.SetInts("OutputIndexer", CalculateIndexer(output.Shape));

            compute.SetBuffer(kernel, "Input" , input .Buffer);
            compute.SetBuffer(kernel, "Filter", filter.Buffer);
            compute.SetBuffer(kernel, "Bias"  , bias  .Buffer);
            compute.SetBuffer(kernel, "Output", output.Buffer);

            var groupCount = Mathf.Max(1, outChannels / threadCount.x);
            compute.Dispatch(kernel, groupCount, output.Shape[1], output.Shape[0]);
        }

        public static void InvokeReorderWeights(Tensor input, Tensor output)
        {
            var compute = ComputeAssets.Setup;
            var kernel = compute.FindKernel("ReorderWeights");

            Debug.Assert(input.Shape[0] == output.Shape[0]);
            Debug.Assert(input.Shape[1] == output.Shape[1]);
            Debug.Assert(input.Shape[2] == output.Shape[3]);
            Debug.Assert(input.Shape[3] == output.Shape[2]);

            compute.SetInts("InputShape" , input .Shape);
            compute.SetInts("OutputShape", output.Shape);

            compute.SetInts("InputIndexer" , CalculateIndexer(input .Shape));
            compute.SetInts("OutputIndexer", CalculateIndexer(output.Shape));

            compute.SetBuffer(kernel, "Input" , input .Buffer);
            compute.SetBuffer(kernel, "Output", output.Buffer);

            compute.Dispatch(kernel, input.Shape[0], input.Shape[1], 1);
        }

        #endregion

        #region Internal helpers

        static Vector3Int GetKernelThreadGroupSizeVector(this ComputeShader self, int kernel)
        {
            uint tgs_x, tgs_y, tgs_z;
            self.GetKernelThreadGroupSizes(kernel, out tgs_x, out tgs_y, out tgs_z);
            return new Vector3Int((int)tgs_x, (int)tgs_y, (int)tgs_z);
        }

        static int[] _tempIndexVector = new int[4];

        static int[] CalculateIndexer(int[] shape)
        {
            if (shape.Length == 4)
            {
                _tempIndexVector[0] = shape[1] * shape[2] * shape[3];
                _tempIndexVector[1] = shape[2] * shape[3];
                _tempIndexVector[2] = shape[3];
                _tempIndexVector[3] = 1;
            }
            else
            {
                _tempIndexVector[0] = shape[1] * shape[2];
                _tempIndexVector[1] = shape[2];
                _tempIndexVector[2] = 1;
                _tempIndexVector[3] = 0;
            }
            return _tempIndexVector;
        }

        #endregion
    }
}

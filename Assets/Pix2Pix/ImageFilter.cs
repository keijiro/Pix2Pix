using UnityEngine;

namespace Pix2Pix
{
    static class ImageFilter
    {
        public static Tensor Preprocess(Texture2D source)
        {
            var compute = ComputeAssets.Image;
            var kernel = compute.FindKernel("ImageToTensor");

            uint tgn_x, tgn_y, tgn_z;
            compute.GetKernelThreadGroupSizes(kernel, out tgn_x, out tgn_y, out tgn_z);

            var width  = source.width;
            var height = source.height;

            Debug.Assert(width  % tgn_x == 0);
            Debug.Assert(height % tgn_y == 0);

            var tensor = new Tensor(new [] {height, width, 3});
            var buffer = new ComputeBuffer(tensor.Data.Length, sizeof(float));

            compute.SetInts("Shape", tensor.Shape);
            compute.SetTexture(kernel, "InputImage", source);
            compute.SetBuffer(kernel, "OutputTensor", buffer);
            compute.Dispatch(kernel, width / (int)tgn_x, height / (int)tgn_y, 1);
            buffer.GetData(tensor.Data);

            buffer.Dispose();

            return tensor;

            /*
            var w = source.width;
            var h = source.height;

            var data = new float[h * w * 3];
            var i = 0;

            for (var y = 0; y < h; y++)
            {
                for (var x = 0; x < w; x++)
                {
                    var p = source.GetPixel(x, y);
                    data[i++] = p.r * 2 - 1;
                    data[i++] = p.g * 2 - 1;
                    data[i++] = p.b * 2 - 1;
                }
            }

            return new Tensor(new[]{h, w, 3}, data);
            */
        }

        public static Texture Deprocess(Tensor source)
        {
            var compute = ComputeAssets.Image;
            var kernel = compute.FindKernel("TensorToImage");

            uint tgn_x, tgn_y, tgn_z;
            compute.GetKernelThreadGroupSizes(kernel, out tgn_x, out tgn_y, out tgn_z);

            var width  = source.Shape[1];
            var height = source.Shape[0];

            Debug.Assert(width  % tgn_x == 0);
            Debug.Assert(height % tgn_y == 0);

            var buffer = new ComputeBuffer(source.Data.Length, sizeof(float));
            buffer.SetData(source.Data);

            var texture = new RenderTexture(width, height, 0);
            texture.enableRandomWrite = true;
            texture.Create();

            compute.SetInts("Shape", source.Shape);
            compute.SetBuffer(kernel, "InputTensor", buffer);
            compute.SetTexture(kernel, "OutputImage", texture);
            compute.Dispatch(kernel, width / (int)tgn_x, height / (int)tgn_y, 1);

            buffer.Dispose();

            return texture;

            /*
            var w = source.Shape[1];
            var h = source.Shape[0];

            var tex = new Texture2D(w, h);
            var i = 0;

            for (var y = 0; y < h; y++)
            {
                for (var x = 0; x < w; x++)
                {
                    var r = (source.Data[i++] + 1) / 2;
                    var g = (source.Data[i++] + 1) / 2;
                    var b = (source.Data[i++] + 1) / 2;
                    tex.SetPixel(x, y, new Color(r, g, b));
                }
            }

            tex.Apply();

            return tex;
            */
        }
    }
}

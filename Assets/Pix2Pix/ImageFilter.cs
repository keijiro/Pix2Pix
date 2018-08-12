using UnityEngine;

namespace Pix2Pix
{
    static class ImageFilter
    {
        public static Tensor Preprocess(Texture2D source)
        {
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
        }

        public static Texture2D Deprocess(Tensor source)
        {
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
        }
    }
}

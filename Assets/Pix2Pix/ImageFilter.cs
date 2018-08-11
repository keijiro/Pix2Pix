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

            return new Tensor(new[]{2, 2, 3}, data);
        }
    }
}

// .pict weight file reader
// https://github.com/keijiro/Pix2Pix

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace Pix2Pix
{
    public static class WeightReader
    {
        #region JSON deserialization

        [Serializable] struct ShapeInfo
        {
            public string name;
            public int[] shape;
        }

        [Serializable] struct ShapeList
        {
            public ShapeInfo[] shapes;
        }

        static ShapeInfo[] ReadShapeInfoJson(Byte[] data)
        {
            var json = "{\"shapes\":" + Encoding.UTF8.GetString(data) + "}";
            return UnityEngine.JsonUtility.FromJson<ShapeList>(json).shapes;
        }

        #endregion

        #region Public method

        public static Dictionary<string, Tensor> ReadFromFile(string filename)
        {
            var table = new Dictionary<string, Tensor>();
            var reader = new BinaryReader(File.Open(filename, FileMode.Open));

            // Read the shape list.
            var length = reader.ReadBEInt();
            var shapes = ReadShapeInfoJson(reader.ReadBytes(length));

            // Read the value table.
            length = reader.ReadBEInt();
            var values = new float[length];
            for (var i = 0; i < length / 4; i++) values[i] = reader.ReadSingle();

            // Read and decode the weight table.
            length = reader.ReadBEInt(); // not used
            for (var i = 0; i < shapes.Length; i++)
            {
                var info = shapes[i];
                length = info.shape.Aggregate(1, (acc, x) => acc * x);

                var data = new float[length];
                for (var j = 0; j < length; j++) data[j] = values[reader.ReadByte()];

                var tensor = new Tensor(new Shape(info.shape), data);

                if (info.name.Contains("conv2d_transpose/kernel"))
                {
                    var temp = new Tensor(new Shape(
                        info.shape[0], info.shape[1], info.shape[3], info.shape[2]
                    ));
                    GpuBackend.InvokeReorderWeights(tensor, temp);
                    tensor.Dispose();
                    tensor = temp;
                }

                table[info.name] = tensor;
            }

            return table;
        }

        public static void DisposeTable(Dictionary<string, Tensor> table)
        {
            foreach (var pair in table) pair.Value.Dispose();
        }

        #endregion
    }

    static class BinaryReaderExtension
    {
        public static int ReadBEInt(this BinaryReader reader)
        {
            var b1 = reader.ReadByte();
            var b2 = reader.ReadByte();
            var b3 = reader.ReadByte();
            var b4 = reader.ReadByte();
            return (b1 << 24) | (b2 << 16) | (b3 << 8) | b4;
        }
    }
}

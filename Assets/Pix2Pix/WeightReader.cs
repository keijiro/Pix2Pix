using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using System.Linq;

namespace Pix2Pix
{
    static class WeightReader
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

                table[info.name] = new Tensor(info.shape, data);

                if (info.name.Contains("conv2d_transpose/kernel"))
                {
                    var t = table[info.name];
                    table[info.name] = GpuHelper.SwapFilter(t);
                    t.Dispose();
                }
            }

            return table;
        }

        public static void DisposeTable(Dictionary<string, Tensor> table)
        {
            foreach (var pair in table) pair.Value.Dispose();
        }

        #endregion
    }
}

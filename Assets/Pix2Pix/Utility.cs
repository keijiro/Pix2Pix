using System.IO;

namespace Pix2Pix
{
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

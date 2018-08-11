using UnityEngine;
using System.IO;

namespace Pix2Pix
{
    public class Test : MonoBehaviour
    {
        [SerializeField] string _weightFileName;
        [SerializeField] Texture2D _sourceTexture;

        void Start()
        {
            var path = Path.Combine(Application.streamingAssetsPath, _weightFileName);
            var weights = WeightReader.ReadFromFile(path);
            var source = ImageFilter.Preprocess(_sourceTexture);
            Generator.Apply(source, weights);
        }
    }
}

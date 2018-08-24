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
            var fullPathWeights = Path.Combine(Application.streamingAssetsPath, _weightFileName);
            var weights = WeightReader.ReadFromFile(fullPathWeights);

            var stopwatch = new System.Diagnostics.Stopwatch();
            stopwatch.Start();

            var source = ImageFilter.Preprocess(_sourceTexture);
            var generated = Generator.Apply(source, weights);
            var result = ImageFilter.Deprocess(generated);

            stopwatch.Stop();
            Debug.Log("Done. Total inference time is " + stopwatch.Elapsed);

            source.Dispose();
            generated.Dispose();
            WeightReader.DisposeTable(weights);

            GetComponent<MeshRenderer>().material.mainTexture = result;
        }
    }
}

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
            var weights = Path.Combine(Application.streamingAssetsPath, _weightFileName);

            var stopwatch = new System.Diagnostics.Stopwatch();
            stopwatch.Start();

            var result = ImageFilter.Deprocess(
                Generator.Apply(
                    ImageFilter.Preprocess(_sourceTexture),
                    WeightReader.ReadFromFile(weights)
                )
            );

            stopwatch.Stop();
            Debug.Log("Done. Total inference time is " + stopwatch.Elapsed);

            GetComponent<MeshRenderer>().material.mainTexture = result;
        }
    }
}

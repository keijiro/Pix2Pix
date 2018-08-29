using UnityEngine;
using UnityEngine.UI;
using System.Collections;

class FpsCounter : MonoBehaviour
{
    [SerializeField] Text _text;

    float _frameRate;

    IEnumerator Start()
    {
        yield return new WaitForSeconds(0.5f);

        while (true)
        {
            var start = Time.time;
            for (var i = 0; i < 10; i++) yield return null;
            var end = Time.time;
            _frameRate = 10 / (end - start);
        }
    }

    void Update()
    {
        _text.text = _frameRate.ToString("00.0") + " FPS";
    }
}

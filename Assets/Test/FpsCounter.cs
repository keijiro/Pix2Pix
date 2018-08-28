using UnityEngine;
using UnityEngine.UI;

class FpsCounter : MonoBehaviour
{
    [SerializeField] Text _text;

    void Update()
    {
        _text.text = (1 / Time.smoothDeltaTime).ToString("00.0") + " FPS";
    }
}

Pix2Pix
=======

![screenshot](https://i.imgur.com/wnR1jmCm.jpg)
![screenshot](https://i.imgur.com/b8I1cOWm.jpg)

This is a naïve (unoptimized) implementation of [pix2pix] in Unity.

This implementation supports the weight data format used in Christopher Hesse's
[interactive demo]. Pick one of the [pre-trained models], or you can train your
own model with using [pix2pix-tensorflow].

At the moment, it's implemented in naïve C# and hasn't been optimized by any
mean. It takes about 5 minutes to inference a single image. Personally, I'm
optimistic about performance because there are many established approaches to
optimize neural network inference.

[pix2pix]: https://github.com/phillipi/pix2pix
[interactive demo]: https://affinelayer.com/pixsrv/
[pre-trained models]: https://github.com/affinelayer/pix2pix-tensorflow-models
[pix2pix-tensorflow]: https://github.com/affinelayer/pix2pix-tensorflow

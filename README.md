# Towards Extreme Image Compression with Latent Feature Guidance and Diffusion Prior

> Zhiyuan Li, Yanhui Zhou, Hao Wei, Chenyang Ge, Jingwen Jiang<br>
> 
> <p style="text-align: justify;"> Image compression at extremely low bitrates (below 0.1 bits per pixel (bpp)) is a significant challenge due to substantial information loss. In this work, we propose a novel two-stage extreme image compression framework that exploits the powerful generative capability of pre-trained diffusion models to achieve realistic image reconstruction at extremely low bitrates. In the first stage, we treat the latent representation of images in the diffusion space as guidance, employing a VAE-based compression approach to compress images and initially decode the compressed information into content variables. The second stage leverages pre-trained stable diffusion to reconstruct images under the guidance of content variables. Specifically, we introduce a small control module to inject content information while keeping the stable diffusion model fixed to maintain its generative capability. Furthermore, we design a space alignment loss to force the content variables to align with the diffusion space and provide the necessary constraints for optimization. Extensive experiments demonstrate that our method significantly outperforms state-of-the-art approaches in terms of visual performance at extremely low bitrates.
</p>

<p style="text-align: justify;">
这是一个示例段落。我们希望这个段落的文字能够两端对齐。通过在 Markdown 中嵌入 HTML 标签，可以实现我们想要的效果。Markdown 本身并不支持这样的格式化，但 HTML 标签可以弥补这一点。
</p>

## :memo:TODE
- [ ] Release code and pretrained models

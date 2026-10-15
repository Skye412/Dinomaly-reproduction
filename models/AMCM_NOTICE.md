# SCRWKV AMCM attribution

`amcm_official.py` is an unmodified copy of `models/AMCM.py` from
https://github.com/zhxhzy/SCRWKV at commit
`f3a1063f6043c769ef828c2ae95edf1029d76c40`.
Downloaded source SHA256:
`2b40458a9f59b2a501de27fef61f2287e847188ed1b55631a2044d7eceeb6e5d`.

Authors credited by upstream: Hanxu Zhang, Chen Jia, Hui Liu, Xu Cheng,
Fan Shi and Shengyong Chen.
Paper: SCRWKV: Ultra-Compact Structure-Calibrated Vision-RWKV for Topological
Crack Segmentation, https://arxiv.org/abs/2605.14926.
Upstream README declares Apache License 2.0; the standard license text is
retained in `licenses/SCRWKV-Apache-2.0.txt`. Upstream's pinned root does not
contain a separate LICENSE file. The original Dinomaly LICENSE is preserved.

`amcm_tokens.py` is our interface adapter, not upstream code. It applies one
default AMCM to the 28x28 patch grid after the existing noisy bottleneck,
passing the CLS/register prefix through unchanged. Official AMCM initialization
is retained; no extra outer residual or projection is added.

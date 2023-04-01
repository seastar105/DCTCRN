# DCTCRN

This is the unofficial implementation of DCTCRN, from paper Real-time Monaural Speech Enhancement With Short-time Discrete Cosine Transform.

Arxiv: https://arxiv.org/abs/2102.04629

## Result

Model is trained on 28spk version of voicebank+demand dataset from [here](https://datashare.ed.ac.uk/handle/10283/2791)

Evaluation is has done on test set of voicebank+demand dataset.

| MODEL | PESQ | STOI | SI-SNR |
| --- | --- | --- | --- |
| DCTCRN-T | 2.550 | 0.932 | 17.233 |
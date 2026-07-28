# PatchGuard Lacks Robustness: Breaking State-Of-The-Art Defect Detection Under New Norm Attacks

We provide code for evaluating the adversarial robustness of PatchGuard trained on MVTec AD and VisA anomaly detection datasets.
All attacks provided here are white-box attacks implemented in PyTorch and evaluated on both clean-trained and adversarially-trained PatchGuard models.

We provide attack code for  PGD-Linf, APGD-L1, APGD-L2, APGD-Linf, and PGD-L0 along with model architecture, training, and evaluation files.
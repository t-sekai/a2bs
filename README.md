# Audio to Face Blendshape Model

**Author: Tse-kai Chan, R&D Intern @ Qualcomm Institute**

This repo implementes a co-speech gesture generation model that predict the arkit facial blendshape from audio, emotion, text, and speaker ID. The model is modified from the CaMN model in the [BEAT paper](https://arxiv.org/abs/2203.05297). Blendshapes can be sent to animate metahumans in Unreal Engine through an OSC server. You can refer to this [video](https://www.youtube.com/watch?v=y1wVykdmJNM) for tutorial.

To get started, navigate to the `test.ipynb` file and walk through the annotated cells to run the model on the BEAT dataset. * Note that this repo does not contain all the files needed to train the a2bs model. The BEAT dataset need to be downloaded and referenced in the jupyter notebook. 

This is an in progress work. Recent development has been done in a separate private repo, which will be made public when completed.

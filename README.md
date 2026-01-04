# What is it?

This repository roughly follows the metholodgy outlined in (Arditi et. al. 2024)[https://arxiv.org/abs/2406.11717] and the code implementation from FailSpy.

The main thing I did was to stitch together key parts of FailSpy's code into a single function `compute_directions`, can be easily called to find refusal directions within the model.

I also made it easier to find the refusal directions, foregoing the more complex (and in my experience, not very useful) objectives discussed in the paper and implemented by FailSpy.

Specifically, I created a review function to allow humans to manually find the layer with the strongest refusal signal, and I additionally use the WildGuard classifier from AllenAI to automatically determine model output harmfulness with different MMD vector interventions.

Finally, I set up it up to have a CLI interface (an example of use can be found in the `.pbs` file) and save the calculated MMD vectors, the best MMD vector, and the model responses with different interventions for spot checking.

Empirically, I found the code to work quite well for a number of different language models, except Llama 2-13B. I think that model is just a bit of an oddball because it was way overfit on safety data. 

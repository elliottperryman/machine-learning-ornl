# data_generator

Generate synthetic pileup for machine learning code.
I put a lot of work into this but it looks like a pretty simple thing.

I sped up compuation by about a 100X using the following:
* bit level logic instead of arithmetic
* I calculated the best way to do FFTs for the computer I was working on and saved the methodology
* I batch all the computations together to reduce context switching and use the whole noise spectrum
* I #DEFINE everything I can and unroll all the loops I can

This took a lot of time and work, profiling, optimizing, and profiling again, but I don't think I use this that much since Gaussian noise is a pretty good approximation. Either way, I think I learned a lot about optimizing code and why it's better to optimize from the top down.


- `embedding_l2_costy`: Did a run with weight = .0001 and that seemed too strong. Mean l1 weight of embeddings was .01, max=.4. Mean l2 norm = .005
- increasing prod emb size from 32->64 didn't seem to help (and maybe hurt generalization?). But that was with an absurd weight cost.
- rnn size seems really important. Decreasing from 64 to 32 hurt perf a lot. Increasing to 128 gave the biggest perf improvement.
- grad clipping (1.0) results seem worse
- dropout seems to help generalization
- layer norm barely seemed to make a difference, and slowed down training more than 2x (significantly slower than run where rnn size was doubled)

I'm basing this all on 10k batches. All models were definitely still learning at that point, so it's possible that some setting that seemed bad here could pull ahead after more epochs?

Some default params to start with for next runs:

- rnn_size = 256
- dropout = 1
- p_emb_size = 32 (but also try 16)

Other things to try:
- bigger/smaller starting learning rates
- sketch-rnn's lstm implementation vs. vanilla tf (which doesn't have ortho init, or dropout w/o memory loss). particularly interesting to see if there are perf differences (since layernorm rnn was so slow)

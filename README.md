# Policy Learning Using Weak Supervision

[[`Paper`](https://arxiv.org/pdf/2010.01748.pdf)]
[[`arXiv`](https://arxiv.org/abs/2010.01748)]
[[`Project Page`](http://www.cs.toronto.edu/~wangjk/publications/peerpl.html)]

> [Policy Learning Using Weak Supervision]()  
> Jingkang Wang, Hongyi Guo, Zhaowei Zhu, Yang Liu \
> NeurIPS 2021  

<div align="center">
    <img src="imgs/weak-policy-learning.png" alt><br>
    Weak supervision signals are everywhere! We provide a unified formulation of the weakly supervised policy learning problems. We also propose PeerPL, a new way to perform policy evaluation under weak supervision.
</div>

## Reproduce the Results 
Our code is based on popular RL/BC frameworks: [`keras-rl`](https://github.com/keras-rl/keras-rl), [`stable-baselines`](https://github.com/hill-a/stable-baselines) and [`rl-baselines-zoo`](https://github.com/araffin/rl-baselines-zoo). Please refer to README.md at `rl-noisy-reward` and `bc-cotrain` folders to set-up the environments and launch the experiments. 

## Citation
If you find our code or paper useful, please consider citing
```bibtex
@inproceedings{wang2021policy,
    title={Policy Learning Using Weak Supervision},
    author={Wang, Jingkang and Guo, Hongyi and Zhu, Zhaowei  and Liu, Yang},
    booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
    year={2021}
}
```

## Questions/Bugs
Please submit a Github issue or contact wangjk@cs.toronto.edu and hongyiguo2025@u.northwestern.edu if you have any questions or find any bugs.

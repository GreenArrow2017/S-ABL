This repository is the implementation for the IJCNN2022 accepted paper:
> Yangkai Lin, Suixin Ou
> [Exploit Domain Knowledge: Smarter Abductive Learning and Its Application to Math Word Problem]
> IJCNN 2022

## prime100epoch_math23k_sota.log
The log of training process. 

## src/tool.py
The code of the math property, simplify and filter.

## src/train_and_evaluate.py
line 425-486, the main inplementation.

## env
The environment is the same as the Learning by fixing(https://github.com/evelinehong/LBF.git), and our code is based on them.

## run
nohup python -u run_seq2tree.py --model ma-fix --nstep 50 --name ma-fix >> prime100epoch_math23k_sota.log 2>&1 &. The running order is the same as the LBF.

## Citation
@inproceedings{hong2021weakly,
  title     = {Learning by Fixing: Solving Math Word Problems with Weak Supervision},
  author    = {Hong, Yining and Li, Qing and Ciao, Daniel and Huang, Siyuan and Zhu, Song-Chun},
  booktitle = {Proceedings of the Thirty-Fifth AAAI Conference on Artificial Intelligence, {AAAI-21}},            
  year      = {2021}
}

@inproceedings{lin2021weakly,
  title     = {Exploit Domain Knowledge: Smarter Abductive Learning and Its Application to Math Word Problem},
  author    = {Yangkai Lin, Suixin Ou},
  booktitle = {International Joint Conference on Neural Networks, {IJCNN-22}},            
  year      = {2022}
}


# python createVocab.py --file /home/tejas/dlops_project/CoRPG/our_data/story.txt \
#                       --save_path /home/tejas/dlops_project/CoRPG/our_data/vocab.share

# python preprocess.py --source /home/tejas/dlops_project/CoRPG/data/news-commentary/data/bpe/train.pesu.comb.bpe \
#                      --graph /home/tejas/dlops_project/CoRPG/data/news-commentary/data/train.pesu.graph \
#                      --target /home/tejas/dlops_project/CoRPG/data/news-commentary/data/bpe/train.comb.bpe \
#                      --vocab ./data/vocab/vocab.share \
#                      --save_file ./data/para.pair

python eval/coherence.py --inference \
						 --pretrain_model albert-base-v2 \
						 --save_file /home/tejas/dlops_project/CoRPG/our_data \
			 			 --text_file /home/tejas/dlops_project/CoRPG/our_data/story.txt \

python preprocess.py --source /home/tejas/dlops_project/CoRPG/our_data/story.txt \
                     --graph /home/tejas/dlops_project/CoRPG/our_data/story.txt.graph \
                     --vocab /home/tejas/dlops_project/CoRPG/data/vocab.share \
                     --save_file /home/tejas/dlops_project/CoRPG/our_data/sent.pt \


# python train.py --cuda_num 0 1 2 3\
#                 --vocab ./data_roc/vocab/vocab.share\
#                 --train_file ./data_roc/para.pair\
#                 --checkpoint_path ./data_roc/model_2100 \
#                 --restore_file ./data_roc/checkpoint97.pkl \
#                 --batch_print_info 200 \
#                 --grad_accum 1 \
#                 --graph_eps 0.5 \
#                 --max_tokens 5000

# 5000/


python generator.py --cuda_num 0 \
                 --file  /home/tejas/dlops_project/CoRPG/our_data/sent.pt \
                 --max_tokens 10000 \
                 --vocab /home/tejas/dlops_project/CoRPG/data/vocab.share \
                 --decode_method greedy \
                 --beam 5 \
                 --model_path /home/tejas/dlops_project/CoRPG/data/model.pkl \
                 --output_path /home/tejas/dlops_project/CoRPG/our_data/output \
                 --max_length 300

# de2en

# python avg_param.py --input ./data/en2de/model/checkpoint89.pkl \
#                             ./data/en2de/model/checkpoint85.pkl \
#                             ./data/en2de/model/checkpoint86.pkl \
#                             ./data/en2de/model/checkpoint87.pkl \
#                             ./data/en2de/model/checkpoint88.pkl \
#                     --outputata/en2de/model/checkpoint.pkl


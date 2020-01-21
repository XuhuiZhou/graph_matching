python train.py \
--lr=0.00001 \
--batch_size=32 \
--train_set='data/cite_acl/train_cite_full.csv' \
--test_set='data/cite_acl/dev_cite_full.csv' \
--word2vec_path='data/word_embedding/glove.6B.50d.txt' \
--max_sent_words="5,5" \
--graph=0 \
--tune=1 \
--model_name='model_han_' \
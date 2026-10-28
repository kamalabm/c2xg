from c2xg import C2xG
import os

cxg = C2xG(nickname = "ubc_novels", 
                language = "eng", 
                max_words = 10000000, 
                max_sentence_length = 100, 
                cbow_file = "training_corpus.01.cbow.bin", 
                sg_file = "training_corpus.01.sg.bin",
                normalization = True,
                )

cxg.learn(input_data = "ubc_data.txt", npmi_threshold = 0.70, starting_index = 1, min_count = 5, 
          max_vocab = 50000, cbow_range = False, sg_range = False, get_examples = True, 
          increments = 25000, learning_rounds = 1, forgetting_rounds = 25, cluster_only = False)

cxg.package_model()
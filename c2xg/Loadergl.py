import os
import pickle
import codecs
import gzip
import time
import cytoolz as ct
import pandas as pd
import numpy as np
from cleantext import clean
from gensim.models.phrases import Phrases
import statsmodels.formula.api as smf
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances

#The loader object handles all file access
class Loader(object):

    def __init__(self, in_dir = None, out_dir = None,
                    nickname = "", max_words = False, 
                    phrases = False, sg_model = False, cbow_model = False,
                    max_sentence_length = 50):
    
        self.max_words = max_words
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.phrases = phrases
        self.nickname = nickname
        self.cbow = False
        self.sg = False
        self.sg_model = sg_model
        self.cbow_model = cbow_model
        self.cbow_centroids = False
        self.sg_centroids = False
        self.clips = None
        self.starting_index = 0
        self.max_sentence_length = max_sentence_length

        #Check that directories exist
        if in_dir != None:
            
            if os.path.isdir(self.in_dir) == False:
                os.makedirs(self.in_dir)
                print("Creating input folder")
            
        if out_dir != None:
            if os.path.isdir(self.out_dir) == False:
                os.makedirs(self.out_dir)
                print("Creating output folder")
            
    #---------------------------------------------------------------#
    
    def save_file(self, file, filename):
        
        print("\t\tSaving " + filename)

        #Write file to disk
        try:
            with open(os.path.join(self.out_dir, filename), "wb") as handle:
                pickle.dump(file, handle, protocol = 3)
                    
        except:
            time.sleep(100)
            with open(os.path.join(self.out_dir, filename), "wb") as handle:
                pickle.dump(file, handle, protocol = 3)
                 
    #---------------------------------------------------------------#
    
    def load_file(self, filename):
    
        try:
            with open(os.path.join(self.out_dir, filename), "rb") as handle:
                return_file = pickle.load(handle)
        except Exception as e:
            print(filename, e)
                
            try:
                with open(os.path.join(self.out_dir, filename), "rb") as handle:
                    return_file = pickle.load(handle)
            except:
                return_file = False
                
        return return_file
    
    #---------------------------------------------------------------#
    
    def read_file(self, file, iterating = False):
    
        max_counter = 0
        starting_counter = 0
        clean_lines = []

        #Read lines from uncompressed text file
        if file.endswith(".txt"):
            with codecs.open(os.path.join(self.in_dir, file), "rb") as fo:
                lines = fo.readlines()
        #Read lines from compressed text file
        elif file.endswith(".gz"):
            with gzip.open(os.path.join(self.in_dir, file), "rb") as fo:
                lines = fo.readlines()

        #Get data up to max words
        if iterating == False:

            for line in lines:
                #Ensure utf-8 input
                line = line.decode("utf-8", errors = "replace")
                starting_counter += len(line.split())

                if starting_counter > self.starting_index:
                    #Control the amount of input data
                    if self.max_words != False:
                        if max_counter < self.max_words:
                            if len(line) > 2:
                                max_counter += len(line.split())
                                clean_lines.append(line)
                    else:
                        clean_lines.append(line)

        #Get data up to max words
        elif iterating != False:
            #Define number of words to discard first and then the stopping point
            start = iterating[0]
            stop = iterating[1]

            #Iterate over lines
            for line in lines:
                #Ensure utf-8 input
                line = line.decode("utf-8", errors = "replace")
                starting_counter += len(line.split())
                
                if starting_counter > self.starting_index:
                    if len(line) > 2:
                        max_counter += len(line.split())
                    
                        #Start after existing data is passed
                        #In between, accumulate data
                        if max_counter > start and max_counter < stop:
                            clean_lines.append(line)
                        
        return clean_lines
                
    #---------------------------------------------------------------#
    def get_lexicon(self, input_data, npmi_threshold = 0.75, min_count = 1, max_vocab = None):

        if isinstance(input_data, str):
            
            #Get data
            data = self.read_file(input_data)
            data = [self.clean(line, encode = False) for line in data]

            #Find phrases, then freeze
            phrase_model = Phrases(data, min_count = min_count, threshold = npmi_threshold, scoring = "npmi", delimiter = " ")
            phrases = phrase_model.freeze()

            #Get full lexicon without potential phrases
            full_lexicon = phrase_model.vocab
            threshold = lambda x: " " not in x
            full_lexicon = ct.keyfilter(threshold, full_lexicon)

            #glfix: ensure correct format of full_lexicon_df
            full_lexicon_df = pd.DataFrame(list(full_lexicon.items()), columns=["Word", "Frequency"])  #glfix
            full_lexicon_df.sort_values("Frequency", inplace=True, ascending=False)  #glfix
            full_lexicon_df["Rank"] = full_lexicon_df["Frequency"].rank(method="average", ascending=False)  #glfix
            full_lexicon_df.to_csv(os.path.join(self.out_dir, self.nickname+".full_lexicon.csv"), index=False)  #glfix

            #Get a list of words which account for over 1% of the total data
            total_size = sum(full_lexicon.values())
            threshold = total_size/1000
            unique_words = full_lexicon_df[full_lexicon_df["Frequency"] > threshold]

            #Prepare to estimate Zipfian distribution
            full_lexicon_df["Actual_Frequency"] = full_lexicon_df["Frequency"]  #glfix
            full_lexicon_df["Frequency"] = np.log10(full_lexicon_df["Frequency"])  #glfix
            full_lexicon_df["Rank"] = np.log10(full_lexicon_df["Rank"])  #glfix

            #Do regression and plot against actual data
            regr = smf.ols(formula = 'Frequency ~ Rank', data = full_lexicon_df)  #glfix
            res = regr.fit()

            pred_ols = res.get_prediction()
            iv_l = pred_ols.summary_frame()["obs_ci_lower"]
            iv_u = pred_ols.summary_frame()["obs_ci_upper"]
            fig, ax = plt.subplots(figsize=(8, 6))
            y = full_lexicon_df["Frequency"]
            x = full_lexicon_df["Rank"]
            ax.plot(x, y, "o", label="Data")
            ax.plot(x, res.fittedvalues, "r--.", alpha=0.25, label="Model")
            ax.plot(x, iv_u, "r--", alpha = 0.25)
            ax.plot(x, iv_l, "r--", alpha = 0.25)
            ax.legend(loc="best")
            plt.xlabel("Rank (Log)")
            plt.ylabel("Frequency (Log)")
            plt.savefig(os.path.join(self.out_dir, self.nickname+".lexicon_distribution.png"), dpi=300, bbox_inches = "tight")
            
            #Reduce lexicon to min_count
            threshold = lambda x: x >= min_count
            lexicon = ct.valfilter(threshold, phrase_model.vocab)
            
            #Remove unkept phrases from lexicon
            remove_list = []
            for key in lexicon:
                if " " in key and key not in phrases.phrasegrams:
                    remove_list.append(key)
            for key in remove_list:
                lexicon.pop(key)
                
            #Reduce to max vocab
            if max_vocab != None:
                lexicon = dict(sorted(lexicon.items(), key=lambda x: x[1], reverse=True)[:max_vocab])
                print("REDUCED ", len(lexicon))
             
            return lexicon, phrases, unique_words, full_lexicon_df

    #---------------------------------------------------------------------------#


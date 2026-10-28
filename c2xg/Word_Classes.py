import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from gensim.models import FastText
from gensim.models.fasttext import save_facebook_model
from scipy.stats import mode # -> commented out on line 173?
import kmedoids
from sklearn.metrics import pairwise_distances
from collections import defaultdict

def _kv(model):
    return getattr(model, "wv", model)

def _get_mean_vector(model, keys, weights=None, pre_normalize=True, post_normalize=True):
    import numpy as np
    kv = _kv(model)
    if weights is None:
        weights = [1.0]*len(keys)
    vecs, ws = [], []
    for w, wt in zip(keys, weights):
        try:
            if hasattr(kv, "key_to_index") and w not in kv.key_to_index:
                continue
            v = kv[w]
        except Exception:
            continue
        v = v.astype("float32", copy=False)
        if pre_normalize:
            n = float((v*v).sum()) ** 0.5
            if n: v = v / n
        vecs.append(v)
        try:
            ws.append(float(wt))
        except Exception:
            ws.append(1.0)
    if not vecs:
        dim = getattr(getattr(kv, "vectors", None), "shape", (1,100))[1]
        return np.zeros(dim, dtype="float32")
    arr = np.vstack(vecs)
    ww  = np.asarray(ws, dtype="float32")
    denom = float(ww.sum()) if float(ww.sum()) != 0.0 else 1.0
    mean = (arr * ww[:, None]).sum(axis=0) / denom
    if post_normalize:
        n = float((mean*mean).sum()) ** 0.5
        if n: mean = mean / n
    return mean.astype("float32", copy=False)



# ---- auto-inserted helper: clamp k to valid range ----
def _clamp_k(_N, k=None, medoids=None):
    # accept either k=... or medoids=... (as seen in c2xg codepaths)
    if k is None:
        k = medoids
    try:
        k = int(k)
    except Exception:
        k = 1 if not isinstance(N, int) or N <= 0 else N
    try:
        N = int(N)
    except Exception:
        N = 1
    if N <= 1:
        return max(1, N)
    return max(1, min(k, N))
# ---- end helper ----

try:
    from sklearnex import patch_sklearn
    patch_sklearn()

except:
    print("UNABLE TO ACCELERATE SKLEARN")


class Word_Classes(object):

    def __init__(self, Load):

        self.Load = Load

    #----------------------------------------------------------------------------------#
    def learn_embeddings(self, input_data, model_type, name="embeddings"):

        if model_type == "cbow":
            ws = 1
            sg = 0

        elif model_type == "sg":
            ws = 5
            sg = 1

        #Pre-load text
        data = [x for x in self.Load.load(input_data)]

        model = FastText(vector_size=100, window=ws, sg=sg, hs=0, negative=100, sorted_vocab=1, alpha=0.01, min_count=5, min_n=3, max_n=6)
        model.build_vocab(corpus_iterable=data)
        model.train(corpus_iterable=data, total_examples=len(data), epochs=20)

        #Save
        save_name = os.path.join(self.Load.out_dir, name+"."+model_type+".bin")
        print("Saving " + save_name)
        save_facebook_model(model, save_name)

        return model

    #----------------------------------------------------------------------------------#

    def learn_categories(self, model, vocab, unique_words=None, variety="cbow", top_range=None):
        """
        Cluster words into categories for CBOW/SG using k-medoids over cosine distances.
        Returns (cluster_df, mean_dict)
        """
        import numpy as np
        import pandas as pd
        import multiprocessing as mp
        from sklearn.metrics import pairwise_distances
        import kmedoids

        kv = getattr(model, "wv", model)

        # filter to tokens present in embeddings
        words_all = list(vocab.keys())
        if hasattr(kv, "key_to_index"):
            word_list = [w for w in words_all if w in kv.key_to_index]
        else:
            word_list = [w for w in words_all if w in kv]
        if not word_list:
            return pd.DataFrame(columns=["Rank","Word","Category","Category_Name"]), {}

        X = np.vstack([kv[w] for w in word_list]).astype(np.float64, copy=False)
        distances = pairwise_distances(X, metric="cosine", n_jobs=1)
        np.fill_diagonal(distances, 0.0)
        distances = np.ascontiguousarray(distances, dtype=np.float64)
        N = len(word_list)

        def _default_range(n):
            if n >= 250:  return range(250, 200, -5)
            if n >= 100:  return range(min(100, n), max(10, n//10), -5)
            if n >= 20:   return range(min(20,  n), max(2,  n//5),  -1)
            return range(1, min(2, n)+1)

        # normalize top_range into an iterable of ints
        if top_range is None or isinstance(top_range, bool):
            cluster_range = _default_range(N)
        elif isinstance(top_range, int):
            cluster_range = range(min(top_range, N), max(1, min(10, N//10)) - 1, -1)
        else:
            cluster_range = [k for k in top_range if isinstance(k, int) and 1 <= k <= N] or list(_default_range(N))

        best_sh = -1.0
        best_labels = [0]*N

        for k in cluster_range:
            if k > N:
                continue
            km = kmedoids.fasterpam(distances, medoids=k, max_iter=100000, init='build', n_cpu=mp.cpu_count())
            meds = getattr(km, 'medoids', None)
            meds = np.asarray(meds) if meds is not None else np.array([], dtype=int)
            if meds.size == 0:
                continue

            sh = float(kmedoids.medoid_silhouette(diss=distances, meds=meds)[0])
            print("\t {} : With {} iterations and {} swaps. Loss: {}  Silhouette: {}".format(
                k, getattr(km, 'n_iter', 'NA'), getattr(km, 'n_swap', 'NA'), getattr(km, 'loss', 'NA'), sh))

            if sh > (best_sh + (abs(sh) * 0.001)):
                best_sh = sh
                best_labels = np.argmin(distances[:, meds], axis=1).astype(int).tolist()

        if not isinstance(best_labels, list) or len(best_labels) != N:
            best_labels = [0] * N

        cluster_df = pd.DataFrame({"Word": word_list, "Cluster": best_labels})

        # per-cluster ranking & mean vectors
        from collections import defaultdict
        complete = []
        mean_dict = {}
        name_dict = {}

        for cat, cat_df in cluster_df.groupby("Cluster"):
            tokens = cat_df["Word"].astype(str).tolist()
            if hasattr(kv, "key_to_index"):
                tokens = [w for w in tokens if w in kv.key_to_index]
            else:
                tokens = [w for w in tokens if w in kv]
            if not tokens:
                continue

            if hasattr(model, "rank_by_centrality"):
                ranks_list = model.rank_by_centrality(words=tokens, use_norm=True)
            else:
                ranks_list = [(1.0, w) for w in tokens]
            ranks = pd.DataFrame(ranks_list, columns=["Rank","Word"])
            if ranks.empty:
                continue
            ranks.loc[:, "Category"] = cat

            mean_vector = _get_mean_vector(model, keys=ranks["Word"], weights=ranks["Rank"],
                                           pre_normalize=True, post_normalize=True)
            if mean_vector is None or getattr(mean_vector, "size", 0) == 0:
                continue
            mean_dict[cat] = mean_vector

            if hasattr(kv, "similar_by_vector"):
                ex = [w for (w,_s) in kv.similar_by_vector(mean_vector, topn=10000) if w in tokens]
            else:
                ex = ranks.sort_values("Rank", ascending=False)["Word"].tolist()
            if len(ex) < 2:
                ex = tokens[:2]
            name_dict[cat] = ex[:2]
            ranks.loc[:, "Category_Name"] = "_".join(name_dict[cat])
            complete.append(ranks)

        if not complete:
            return pd.DataFrame(columns=["Rank","Word","Category","Category_Name"]), {}

        merged = pd.concat(complete, ignore_index=True)

        # optional phrase assignment
        phrase_results = []
        phrase_list = []
        try:
            phrase_list = list(getattr(self, "phrase_list", []))
        except Exception:
            pass
        if phrase_list and mean_dict:
            centroids = np.vstack([mean_dict[c] for c in sorted(mean_dict.keys())]).astype(np.float64, copy=False)
            from sklearn.metrics import pairwise_distances as _pwd
            for ph in phrase_list:
                try:
                    v = kv[ph]
                except Exception:
                    continue
                d = _pwd(v.reshape(1, -1), centroids, metric="cosine", n_jobs=1)
                cl = int(np.argmin(d))
                cname = "_".join(name_dict.get(cl, ["cluster", str(cl)]))
                phrase_results.append([0.0, ph, cl, cname])
        if phrase_results:
            phrase_df = pd.DataFrame(phrase_results, columns=["Rank","Word","Category","Category_Name"])
            merged = pd.concat([merged, phrase_df], ignore_index=True)

        # CBOW unique tokens as their own categories
        if (variety == "cbow") and (unique_words is not None):
            try:
                if not unique_words.empty:
                    start_cat = int(merged["Category"].max()) + 1 if "Category" in merged.columns and not merged.empty else 0
                    uw_rows = []
                    for w in unique_words["Word"].astype(str).tolist():
                        uw_rows.append([0.0, w, start_cat, "unique_"+w]); start_cat += 1
                    if uw_rows:
                        uw_df = pd.DataFrame(uw_rows, columns=["Rank","Word","Category","Category_Name"])
                        merged = pd.concat([merged, uw_df], ignore_index=True)
            except Exception:
                pass

        merged = merged.sort_values(by=["Category","Rank"], ascending=False, kind="mergesort").reset_index(drop=True)
        return merged, mean_dict

    def learn_construction_categories(self, grammar, similarity_matrix):
        """
        Cluster constructions using k-medoids. Accepts a similarity (1.0 on diag) or
        a distance matrix. Returns DataFrame(Chunk, Cluster).
        """
        import numpy as np
        import pandas as pd
        import multiprocessing as mp
        import kmedoids

        grammar = [eval(g) if isinstance(g, str) else g for g in grammar]
        if not grammar:
            return pd.DataFrame(columns=["Chunk","Cluster"])

        mat = similarity_matrix
        if isinstance(mat, pd.DataFrame):
            mat = mat.values
        mat = np.asarray(mat, dtype=np.float64)

        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError(f"Construction matrix must be square, got {mat.shape}")

        if mat.min() >= 0.0 and mat.max() <= 1.0 and np.allclose(np.diag(mat), 1.0, atol=1e-6):
            dist = 1.0 - mat
        else:
            dist = mat
        dist = np.nan_to_num(dist, nan=0.0, posinf=1.0, neginf=0.0)
        np.fill_diagonal(dist, 0.0)
        dist = np.ascontiguousarray(dist, dtype=np.float64)

        N = dist.shape[0]
        if N != len(grammar):
            M = min(N, len(grammar))
            dist = np.ascontiguousarray(dist[:M, :M], dtype=np.float64)
            grammar = grammar[:M]

        def _sweep(n):
            if n >= 40: return range(40, 10, -5)
            if n >= 15: return range(min(15, n), max(3, n//3), -2)
            if n >= 4:  return range(min(4, n), 1, -1)
            return [1]

        best_sh = -1.0
        best_labels = [0]*len(grammar)

        for k in _sweep(len(grammar)):
            if k > len(grammar):
                continue
            
            km = kmedoids.fasterpam(dist, medoids=k, max_iter=100000, init='build', n_cpu=mp.cpu_count())
            
            meds = getattr(km, "medoids", [])
            meds = None if meds is None else np.asarray(meds)
            if meds is None or getattr(meds, 'size', len(meds)) == 0:
                continue

            sh = float(kmedoids.medoid_silhouette(diss=dist, meds=meds)[0])
            print("     {} : With {} iterations and {} swaps. Loss: {}  Silhouette: {}".format(
                k,getattr(km, 'n_iter', 'NA'), getattr(km, 'n_swap', 'NA'), getattr(km, 'loss', 'NA'), sh))

            if sh > (best_sh + (abs(sh)*0.001)):
                best_sh = sh
                best_labels = np.argmin(dist[:, meds], axis=1).astype(int).tolist()

        if not best_labels or len(best_labels) != len(grammar):
            best_labels = [0]*len(grammar)

        return pd.DataFrame({"Chunk": [str(g) for g in grammar], "Cluster": best_labels})



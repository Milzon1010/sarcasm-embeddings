import numpy as np, tensorflow as tf, json, pickle
from pathlib import Path

MAX_LEN=32
Path("projector/word").mkdir(parents=True, exist_ok=True)
Path("projector/sent").mkdir(parents=True, exist_ok=True)

# reload model
model=tf.keras.models.load_model("artifacts/sarcasm_embed_model.keras")

# rebuild vectorizer from saved vocab
vocab=pickle.load(open("artifacts/vectorizer_vocab.pkl","rb"))
vectorizer=tf.keras.layers.TextVectorization(output_mode="int", output_sequence_length=MAX_LEN)
vectorizer.set_vocabulary(vocab)

# WORD projection (search kata)
emb_layer=next(l for l in model.layers if isinstance(l, tf.keras.layers.Embedding))
E=emb_layer.get_weights()[0]; vocab_eff=min(len(vocab), E.shape[0])
START=2; N=min(5000, max(0, vocab_eff-START)); idx=np.arange(START, START+N)
np.savetxt("projector/word/vectors.tsv", E[idx], delimiter="\t")
with open("projector/word/metadata.tsv","w",encoding="utf-8") as f:
    f.write("token\trank\n")
    for r,i in enumerate(idx): f.write(f"{vocab[i].replace('\t',' ').replace('\n',' ')}\t{r}\n")

# SENTENCE projection (warnai label)
URL="https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json"
data=Path("data/sarcasm.json")
if not data.exists():
    import urllib.request; data.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve(URL, data.as_posix())
ds=json.load(open(data))
texts=[d["headline"] for d in ds]; labels=np.array([d["is_sarcastic"] for d in ds], np.int32)
M=min(2000, len(texts))
X=vectorizer(tf.constant(texts[:M])); y=labels[:M]

encoder=tf.keras.Sequential(model.layers[:-1]); _=encoder(tf.zeros((1,MAX_LEN),dtype=tf.int32))
Z=encoder.predict(tf.cast(X, tf.int32), batch_size=512, verbose=0)
np.savetxt("projector/sent/vectors.tsv", Z, delimiter="\t")
with open("projector/sent/metadata.tsv","w") as f:
    f.write("label\n")
    for i in range(M): f.write(f"{int(y[i])}\n")
print("Exported projector files.")

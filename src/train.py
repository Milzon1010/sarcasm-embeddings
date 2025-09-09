import os, json, numpy as np, tensorflow as tf
from pathlib import Path
from sklearn.metrics import classification_report

SEED=42; np.random.seed(SEED); tf.random.set_seed(SEED)
VOCAB_SIZE=20000; MAX_LEN=32; EMBED_DIM=16; BATCH=128; EPOCHS=6
DATA=Path("data/sarcasm.json"); Path("figs").mkdir(exist_ok=True); Path("artifacts").mkdir(exist_ok=True)

URL="https://storage.googleapis.com/tensorflow-1-public/course3/sarcasm.json"
if not DATA.exists():
    import urllib.request
    DATA.parent.mkdir(exist_ok=True, parents=True)
    urllib.request.urlretrieve(URL, DATA.as_posix())

with open(DATA) as f: ds=json.load(f)
texts=[d["headline"] for d in ds]; labels=np.array([d["is_sarcastic"] for d in ds], np.int32)
N=len(texts); idx=np.arange(N); np.random.shuffle(idx); cut=int(.9*N)
tr_text=[texts[i] for i in idx[:cut]]; va_text=[texts[i] for i in idx[cut:]]
tr_y=labels[idx[:cut]]; va_y=labels[idx[cut:]]

vectorizer=tf.keras.layers.TextVectorization(
    max_tokens=VOCAB_SIZE, standardize="lower_and_strip_punctuation",
    output_mode="int", output_sequence_length=MAX_LEN)
vectorizer.adapt(tf.data.Dataset.from_tensor_slices(tr_text).batch(1024))

def ds(x,y,train=False):
    d=tf.data.Dataset.from_tensor_slices((vectorizer(tf.constant(x)), y))
    if train: d=d.shuffle(2048)
    return d.batch(BATCH).prefetch(tf.data.AUTOTUNE)

train_ds=ds(tr_text,tr_y,True); val_ds=ds(va_text,va_y,False)

model=tf.keras.Sequential([
    tf.keras.layers.Input(shape=(MAX_LEN,), dtype="int32"),
    tf.keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM),
    tf.keras.layers.Conv1D(128,5,activation="relu"),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer=tf.keras.optimizers.Adam(1e-3), metrics=["accuracy"])
hist=model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)

# plots
import matplotlib.pyplot as plt
plt.figure(); plt.plot(hist.history["accuracy"]); plt.plot(hist.history["val_accuracy"]); plt.legend(["train","val"]); plt.title("Accuracy"); plt.savefig("figs/training_curves.png", dpi=180); plt.close()

# eval quick
pred=(model.predict(val_ds, verbose=0).ravel()>0.5).astype(int)
print(classification_report(va_y[:len(pred)], pred, digits=3))

# save
model.save("artifacts/sarcasm_embed_model.keras")
# sample preds
samples=["Great, another delay…","Thank you so much 🙏","Yeah right, that meeting was super useful"]
probs=model.predict(vectorizer(tf.constant(samples)), verbose=0).ravel()
with open("figs/sample_preds.txt","w") as f:
    for s,p in zip(samples, probs): f.write(f"{s}\tSarcasm p={p:.2f}\n")
# save vectorizer config for reuse
import pickle; pickle.dump(vectorizer.get_vocabulary(), open("artifacts/vectorizer_vocab.pkl","wb"))

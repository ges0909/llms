# Word Embeddings & Vector Spaces

## Was sind Embeddings?

Nach der Tokenisierung liegt Text als diskrete Zahlenfolge vor:

```
"Hallo Welt" → [15496, 16896]
```

Modelle können mit reinen IDs nicht rechnen: ID 10 ist mathematisch nicht "doppelt so viel"
wie ID 5 und hat keine inhärente Ähnlichkeit zu ID 11.

**Embeddings wandeln diskrete Token-IDs in dichte Vektoren (Floats) um.**
Hier wird Sprache zur Geometrie: Semantisch ähnliche Begriffe liegen im Vektorraum nah beieinander.

```
Token-ID 15496 ("Hallo") → [-0.14, +0.82, +0.33, ..., -0.05] (d_model Dimensionen)
```

| Begriff           | Erklärung                                                       |
| ----------------- | --------------------------------------------------------------- |
| **Embedding**     | Ein dichter Zahlenvektor, der ein Token im Raum repräsentiert   |
| **d_model**       | Dimension des Vektorraums (z. B. 768 bei GPT-2, 4096 bei Llama) |
| **Vokabular (V)** | Anzahl aller bekannten Tokens (Zeilen der Embedding-Matrix)     |
| **Lookup**        | Schneller Tabellenzugriff `W_E[token_id]` in O(1)               |

---

## One-Hot vs. Dense Embeddings

| Eigenschaft           | One-Hot-Encoding                     | Dense Embeddings                      |
| --------------------- | ------------------------------------ | ------------------------------------- |
| **Vektorgröße**       | V (z. B. 50.000 oder 128.000 Zahlen) | d_model (z. B. 768 oder 4.096 Zahlen) |
| **Werte**             | Nur Nullen und genau eine Eins       | Kontinuierliche Fließkommazahlen      |
| **Speichereffizienz** | ❌ Katastrophal (99.99% Nullen)      | ✅ Kompakt                            |
| **Semantische Nähe**  | ❌ Keine (alle Paare orthogonal)     | ✅ Vektorabstand spiegelt Bedeutung   |
| **Lernbar**           | ❌ Fest vorgegeben                   | ✅ Wird im Training optimiert         |

---

## Die Embedding-Matrix als Lookup

Mathematisch ist das Nachschlagen eine Matrixmultiplikation:

$$\vec{x}_{\text{one-hot}} \cdot W_E = W_E[\text{token\_id}]$$

Da der One-Hot-Vektor nur eine `1` an Stelle $i$ hat, filtert er exakt die $i$-te Zeile der Matrix
heraus.

In der Praxis wird diese Multiplikation nicht ausgeführt:

- **Effizienter Lookup**: Der Transformer greift per Index direkt auf `W_E[token_id]` zu ($O(1)$).
- **Sequenzen**: Eine Sequenz von $T$ Token-IDs ergibt eine Matrix der Form $(T, d_{model})$.

---

## Vektorgeometrie & Semantik

## Wie Embeddings gelernt werden

Am Anfang sind Embedding-Vektoren zufällig initialisiert. Beim Training bekommt das Modell eine
Aufgabe und einen Fehlerwert. Hier lautet die Aufgabe: Aus dem aktuellen Wort das nächste Wort
vorhersagen. Gradientenabstieg passt die Embeddings und die Ausgabeschicht so an, dass der
Vorhersagefehler sinkt.

Die Demo nutzt einen winzigen Korpus und ein Bigram-Modell. Wörter mit ähnlichen Kontexten können
dabei ähnliche Vorhersagen unterstützen. Das garantiert keine allgemeine semantische Ähnlichkeit:
Die Vektoren werden nur für diese konkrete Aufgabe und diesen Korpus optimiert.

## Vektorgeometrie & Semantik

Im Vektorraum drückt sich Bedeutung durch Winkel und Abstände aus:

### 1. Cosine-Similarity (Kosinus-Ähnlichkeit)

Misst den Winkel zwischen zwei Vektoren, unabhängig von deren Länge:

$$\cos(\theta) = \frac{\vec{a} \cdot \vec{b}}{\|\vec{a}\|_2 \|\vec{b}\|_2}$$

- `+1.0`: Identische Richtung (maximale Ähnlichkeit)
- ` 0.0`: Orthogonal (keine semantische Beziehung)
- `-1.0`: Entgegengesetzte Richtung

### 2. Vektorarithmetik (Analogien)

Semantische Beziehungen lassen sich durch Vektoraddition und -subtraktion abbilden:

$$\vec{v}_{\text{König}} - \vec{v}_{\text{Mann}} + \vec{v}_{\text{Frau}} \approx \vec{v}_{\text{Königin}}$$

$$\vec{v}_{\text{Welpe}} - \vec{v}_{\text{Hund}} + \vec{v}_{\text{Katze}} \approx \vec{v}_{\text{Kätzchen}}$$

---

## Embeddings in modernen Transformern

### 1. Skalierung mit $\sqrt{d_{model}}$

Im ursprünglichen Transformer (_Attention Is All You Need_) werden Token-Embeddings mit
$\sqrt{d_{model}}$ multipliziert:

$$X = X_{\text{token}} \cdot \sqrt{d_{model}} + X_{\text{position}}$$

Dies verhindert, dass die Embedding-Werte bei Addition mit den Positionscodierungen untergehen.

### 2. Das Polysemie-Problem (Statisch vs. Kontextuell)

Reine Token-Embeddings sind **statisch**:

- _"Ich sitze auf der Bank"_
- _"Ich überweise Geld an die Bank"_

In beiden Sätzen erhält das Token `" Bank"` vor dem Eintritt in den Transformer exakt denselben
Vektor.

Erst nachfolgende **Attention-Schichten (Topic 4)** erlauben es den Tokens, Informationen
untereinander auszutauschen und kontextuelle Repräsentationen zu formen.

### 3. Weight Tying

Viele Architekturen (GPT-2, Gemma) teilen sich die Gewichte zwischen Input-Embedding-Matrix und
dem finalen Output-Projektionslayer:

$$\text{Logits} = H_{\text{final}} \cdot W_E^T$$

Das spart Millionen lernbare Parameter ein.

---

## Reale Modelle im Vergleich

| Modell        | Vokabular (V) | Dimension ($d_{model}$) | Embedding-Parameter | Anteil am Modell |
| ------------- | ------------- | ----------------------- | ------------------- | ---------------- |
| GPT-2 (Small) | 50.257        | 768                     | ~38,6 Mio.          | ~31,1%           |
| GPT-2 (XL)    | 50.257        | 1.600                   | ~80,4 Mio.          | ~5,2%            |
| Llama 2 (7B)  | 32.000        | 4.096                   | ~131,1 Mio.         | ~1,9%            |
| Llama 3 (8B)  | 128.256       | 4.096                   | ~525,3 Mio.         | ~6,5%            |
| GPT-3 (175B)  | 50.257        | 12.288                  | ~617,6 Mio.         | ~0,4%            |

---

## Lektionen & Ausführen

| Datei                                                    | Inhalt                               | Befehl                     |
| -------------------------------------------------------- | ------------------------------------ | -------------------------- |
| [`embedding_lookup.py`](embedding_lookup.py)             | One-Hot vs. Dense Lookup             | `uv run embed-lookup`      |
| [`vector_similarity.py`](vector_similarity.py)           | Cosine-Similarity & Analogien        | `uv run embed-similarity`  |
| [`transformer_embeddings.py`](transformer_embeddings.py) | LLM-Layer, Skalierung & Kontext      | `uv run embed-transformer` |
| [`embedding_training.py`](embedding_training.py)         | Embeddings per Vorhersage trainieren | `uv run embed-training`    |

---

## Ressourcen

- [3Blue1Brown: Visualizing Attention and Embeddings](https://www.youtube.com/watch?v=eMlx5fFNoYc)
- [Jay Alammar: The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/)
- [Vaswani et al. (2017): Attention Is All You Need](https://arxiv.org/abs/1706.03762)

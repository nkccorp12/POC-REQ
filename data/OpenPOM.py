import numpy as np
import pandas as pd
import deepchem as dc
from openpom.feat.graph_featurizer import GraphFeaturizer
from openpom.models.mpnn_pom import MPNNPOMModel

# 1. CSV laden
df = pd.read_csv('molecules.csv')

# 2. Featurizer und Modell initialisieren
featurizer = GraphFeaturizer()
model = MPNNPOMModel(n_tasks=55, pretrained=True)  # lädt das vortrainierte POM-Modell

# 3. Moleküle in Graph-Features umwandeln
graphs = featurizer.featurize(df['nonStereoSMILES'])

# 4. Dataset erstellen 
dataset = dc.data.NumpyDataset(X=graphs)

# 5. Embeddings berechnen (Penultimate Layer)
embeddings = model.predict(dataset)  # Form: (4983, 256)

# 6. Als .npy speichern
np.save('embeddings.npy', embeddings)

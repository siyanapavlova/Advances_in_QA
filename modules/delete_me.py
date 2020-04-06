from modules.Encoder import Encoder
from modules.FusionBlock import FusionBlock
from modules.EntityGraph import EntityGraph
import utils

graph = EntityGraph()
encoder = Encoder()

q_ids, c_ids = encoder.encode()

q_emb = encoder.predict(c_ids, q_ids)
c_emb = encoder.predict(q_ids, c_ids)


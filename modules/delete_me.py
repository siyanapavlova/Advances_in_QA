from modules.Encoder import Encoder
from modules.FusionBlock import FusionBlock
from modules.EntityGraph import EntityGraph
import utils

graph = EntityGraph()
encoder = Encoder()

q_ids, c_ids = encoder.token_ids()
q_emb = encoder(c_ids, q_ids)
c_emb = encoder(q_ids, c_ids)

fusionblock = FusionBlock(300)

# these will be shortcut by calling the forward() or fusionblock
#fusionblock.entity_embs = fusionblock.tok2ent()
#fusionblock.entity_embs = fusionblock.entity_embs.unsqueeze(2)
#updated_entity_embs = fusionblock.graph_attention()
#
#fusionblock.query_emb = fusionblock.bidaf(updated_entity_embs, fusionblock.query_emb)
#Ct = fusionblock.graph2doc(updated_entity_embs)
Ct = fusionblock(c_emb, q_emb, graph)



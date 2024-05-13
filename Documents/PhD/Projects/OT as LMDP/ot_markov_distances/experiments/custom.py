import torch
from ot_markov_distances import discounted_wl_k
from ot_markov_distances.sinkhorn import sinkhorn_internal, sinkhorn
from ot_markov_distances.utils import markov_measure
import time

MX = torch.tensor([[0.5,0.25,0.25],
             [0.25,0.25, 0.5],
                   [0.5,0.25,0.25]]).unsqueeze(0)

MY = torch.tensor([[0.5,0.25,0.25],
             [0.25,0.5, 0.25],
                   [0.,0.25,0.75]]).unsqueeze(0)

cost_matrix = torch.tensor([[1,1,0],
                            [1,0,1],
                            [0,1,0]]).unsqueeze(0)


muX = markov_measure(MX)
muY = markov_measure(MY)
st = time.time()
print(sinkhorn(muX, muY, cost_matrix, .01, return_has_converged=True))
et = time.time()
elapsed_time = et - st
print(elapsed_time)

#discounted_wl_k(MX, MY, cost_matrix, k = 5)